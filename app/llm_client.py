"""
Claude API wrapper — thin client used by the QA pipeline.
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, system: str = "", max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Send a prompt and return the text response."""
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text
