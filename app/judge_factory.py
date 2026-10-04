"""
Model-agnostic judge selection for deepeval metrics.

Instead of hardcoding a single provider as the LLM-as-judge, this module
picks one at runtime:

    JUDGE_BACKEND=ollama     -> force local Ollama
    JUDGE_BACKEND=gemini     -> force Gemini (requires GEMINI_API_KEY)
    JUDGE_BACKEND=anthropic  -> force Claude (requires ANTHROPIC_API_KEY)
    JUDGE_BACKEND=auto       -> (default) prefer local Ollama when it's
                                 actually running and the model is pulled;
                                 otherwise fall back to whichever cloud key
                                 is present in .env (GEMINI_API_KEY, then
                                 ANTHROPIC_API_KEY)

Usage:
    from app.judge_factory import get_judge_model
    judge_model = get_judge_model()
    metric = FaithfulnessMetric(threshold=0.85, model=judge_model)
"""

import os

from anthropic import Anthropic, AsyncAnthropic, BadRequestError
from deepeval.models import DeepEvalBaseLLM


class ClaudeLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="claude-haiku-4-5-20251001"):
        self.model_name = model_name
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)
        self._send_temperature = True

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def _kwargs(self, prompt: str) -> dict:
        kwargs = dict(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        # Pin temperature=0 for deterministic judging, unless the model rejected it.
        if self._send_temperature:
            kwargs["temperature"] = 0
        return kwargs

    @staticmethod
    def _temperature_rejected(err: BadRequestError) -> bool:
        return "temperature" in str(err).lower()

    def generate(self, prompt: str) -> str:
        try:
            res = self.client.messages.create(**self._kwargs(prompt))
        except BadRequestError as e:
            if not (self._send_temperature and self._temperature_rejected(e)):
                raise
            self._send_temperature = False  # newer models deprecate `temperature`
            res = self.client.messages.create(**self._kwargs(prompt))
        return res.content[0].text

    async def a_generate(self, prompt: str) -> str:
        try:
            res = await self.async_client.messages.create(**self._kwargs(prompt))
        except BadRequestError as e:
            if not (self._send_temperature and self._temperature_rejected(e)):
                raise
            self._send_temperature = False
            res = await self.async_client.messages.create(**self._kwargs(prompt))
        return res.content[0].text


class GeminiLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="gemini-2.5-flash"):
        from google import genai

        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str) -> str:
        res = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"temperature": 0},
        )
        return res.text

    async def a_generate(self, prompt: str) -> str:
        res = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"temperature": 0},
        )
        return res.text


def _ollama_available(model: str) -> bool:
    """Non-fatal check: is Ollama installed, running, and is `model` pulled?"""
    try:
        import ollama
    except ImportError:
        return False

    try:
        pulled_models = [m.model for m in ollama.list().models]
    except Exception:
        return False

    model_base = model.split(":")[0]
    return any(m.split(":")[0] == model_base for m in pulled_models)


def get_judge_model() -> DeepEvalBaseLLM:
    """
    Resolve a deepeval-compatible judge model without hardcoding a provider.

    Resolution order:
      1. EVAL_JUDGE_BACKEND env var, if explicitly set to
         ollama/gemini/anthropic, forces that backend. Falls back to the
         shared JUDGE_BACKEND var (used by conftest.py's ollama/openai judge
         fixture for the other eval files) so an existing JUDGE_BACKEND=ollama
         keeps working, but conftest-only values like "openai" won't leak in
         here by accident — use EVAL_JUDGE_BACKEND to target this factory
         specifically without touching the other test files.
      2. Otherwise ("auto", or unset): prefer a local Ollama model when it's
         actually reachable and pulled (free, no API cost, no data leaves
         the machine) — falling back to whichever cloud API key is present
         in .env (GEMINI_API_KEY, then ANTHROPIC_API_KEY).
    """
    backend = os.getenv("EVAL_JUDGE_BACKEND", os.getenv("JUDGE_BACKEND", "auto")).lower()
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

    if backend == "auto":
        if _ollama_available(ollama_model):
            backend = "ollama"
        elif os.getenv("GEMINI_API_KEY"):
            backend = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            backend = "anthropic"
        else:
            raise RuntimeError(
                "\n\n"
                "  [judge_factory] No judge model available.\n"
                "  Fix: either run a local judge —\n\n"
                f"      ollama pull {ollama_model}\n"
                "      ollama serve\n\n"
                "  — or set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env.\n"
            )

    if backend == "ollama":
        from app.local_judge import OllamaJudge
        return OllamaJudge(model=ollama_model)

    if backend == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("EVAL_JUDGE_BACKEND=gemini requires GEMINI_API_KEY in .env.")
        return GeminiLLM(model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    if backend == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("EVAL_JUDGE_BACKEND=anthropic requires ANTHROPIC_API_KEY in .env.")
        return ClaudeLLM(model_name=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"))

    raise RuntimeError(
        f"Unknown judge backend '{backend}'. Use one of: auto, ollama, gemini, anthropic "
        "(set via EVAL_JUDGE_BACKEND, or JUDGE_BACKEND if EVAL_JUDGE_BACKEND is unset)."
    )
