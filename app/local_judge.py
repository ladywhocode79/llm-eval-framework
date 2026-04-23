"""
Local LLM judge using Ollama.

Plugs into deepeval as a drop-in replacement for the default OpenAI judge.
Runs 100% locally — no API cost.

Prerequisites
-------------
1. Install the ollama Python package:
       pip install -r requirements.txt

2. Install the Ollama application:
       Mac:   brew install ollama
       Other: https://ollama.com/download

3. Pull a model (one-time, ~2 GB):
       ollama pull llama3.2

4. Start the Ollama server (keep this running in a separate terminal):
       ollama serve

Usage
-----
    from app.local_judge import OllamaJudge
    judge = OllamaJudge()                # uses llama3.2 by default
    judge = OllamaJudge(model="mistral") # any other pulled model

    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
"""

import json
import re
from typing import Tuple

try:
    import ollama as _ollama_lib
except ImportError:
    raise ImportError(
        "\n\n"
        "  [OllamaJudge] The 'ollama' Python package is not installed.\n"
        "  Fix: run the following command and try again:\n\n"
        "      pip install -r requirements.txt\n\n"
        "  Or install directly:\n\n"
        "      pip install ollama\n"
    )

from deepeval.models.base_model import DeepEvalBaseLLM


class OllamaJudge(DeepEvalBaseLLM):
    """deepeval-compatible judge backed by a local Ollama model."""

    def __init__(self, model: str = "llama3.2"):
        self.model_name = model

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> Tuple[str, float]:
        """
        Call the local Ollama model and return a clean JSON string.

        Two-layer defence against malformed JSON from local models:
          1. format="json"  — Ollama grammar-based constraint forces valid JSON tokens
          2. _extract_json  — post-processes the text to strip markdown fences,
                              extra prose, bad escapes, and trailing garbage
        """
        # Reinforce JSON-only output at the prompt level for local models
        json_prompt = (
            prompt
            + "\n\nIMPORTANT: Respond with a single valid JSON object only. "
            "No markdown, no code fences (```), no explanation text. "
            "Start your response with { and end with }."
        )

        try:
            response = _ollama_lib.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": json_prompt}],
                format="json",   # Ollama JSON mode — constrains output to valid JSON tokens
            )
            text = response.message.content
            return self._extract_json(text), 0.0   # cost = 0, local is free

        except Exception as e:
            error_str = str(e).lower()

            if "not found" in error_str or "404" in error_str:
                raise RuntimeError(
                    f"\n\n"
                    f"  [OllamaJudge] Model '{self.model_name}' is not available locally.\n"
                    f"  Fix: pull the model first (one-time download ~2 GB):\n\n"
                    f"      ollama pull {self.model_name}\n\n"
                    f"  Available alternatives:\n"
                    f"      ollama pull llama3.2:1b   # lightest (~1 GB)\n"
                    f"      ollama pull mistral        # highest quality (~4 GB)\n"
                ) from e

            if "connection" in error_str or "refused" in error_str or "connect" in error_str:
                raise RuntimeError(
                    "\n\n"
                    "  [OllamaJudge] Cannot connect to the Ollama server.\n"
                    "  Fix: start Ollama in a separate terminal and keep it running:\n\n"
                    "      ollama serve\n\n"
                    "  If Ollama is not installed:\n"
                    "      Mac:   brew install ollama\n"
                    "      Other: https://ollama.com/download\n"
                ) from e

            raise RuntimeError(
                f"\n\n"
                f"  [OllamaJudge] Unexpected error calling model '{self.model_name}':\n"
                f"  {e}\n"
            ) from e

    async def a_generate(self, prompt: str) -> Tuple[str, float]:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"

    # ── JSON cleaning ─────────────────────────────────────────────────────────

    def _extract_json(self, text: str) -> str:
        """
        Extract a valid JSON object from model output that may contain:
          - Markdown code fences  (```json ... ```)
          - Prose before/after the JSON
          - Invalid escape sequences  (e.g. \H, \C)
          - Single quotes instead of double quotes
          - Trailing commas
        """
        # 1. Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text.strip())
        text = text.strip()

        # 2. Try the cleaned text as-is
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # 3. Extract first complete JSON object/array by bracket matching
        for open_ch, close_ch in [('{', '}'), ('[', ']')]:
            candidate = self._bracket_extract(text, open_ch, close_ch)
            if candidate:
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    # Try fixing common issues on the extracted candidate
                    fixed = self._fix_json_string(candidate)
                    try:
                        json.loads(fixed)
                        return fixed
                    except json.JSONDecodeError:
                        pass

        # 4. Last resort: apply fixes to the full text and try again
        fixed = self._fix_json_string(text)
        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            pass

        # Return original — let deepeval surface its own error with full context
        return text

    def _bracket_extract(self, text: str, open_ch: str, close_ch: str) -> str | None:
        """Find and return the first complete bracket-balanced substring."""
        start = text.find(open_ch)
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return None

    def _fix_json_string(self, text: str) -> str:
        """Apply heuristic fixes to common JSON errors from local models."""
        # Replace single-quoted strings with double-quoted
        text = re.sub(r"'([^']*)'", r'"\1"', text)

        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Fix invalid escape sequences: \H \C \P etc. → \\H \\C \\P
        def fix_escape(m):
            char = m.group(1)
            valid = set('"\\\/bfnrtu')
            return m.group(0) if char in valid else "\\\\" + char

        text = re.sub(r'\\(.)', fix_escape, text)

        return text
