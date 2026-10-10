"""
llm_client.py - the ONLY file that talks to the LLM provider.

Groq exposes an OpenAI-compatible API, so we use the `openai` package with
Groq's base_url. To switch provider later, only this file changes.
The API key is read from the .env file in the FailScope folder (never hard-code it).
"""
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
BASE_URL = "https://api.groq.com/openai/v1"


class LLMClient:
    def __init__(self, model=None):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not found. Put it in the .env file in the FailScope folder.")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=key, base_url=BASE_URL, timeout=60.0, max_retries=1)   # fail fast instead of hanging

    def complete_json(self, system, user):
        """Ask the model for a JSON answer and return it as a Python dict."""
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return json.loads(resp.choices[0].message.content)