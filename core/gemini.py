"""The only file that calls Google: Gemini reads what was retrieved and writes the answer.

Embeddings run on your own computer (core/embeddings.py), so searching costs nothing. Gemini is
called only to read: answers, reranking, describing a figure once, and reading facts once.
"""

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv(Path(__file__).parents[1] / ".env")

CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gemini-3.6-flash")
# A model's speed changes through the day. If the first one is slow or busy, the next one answers.
FALLBACK_MODEL = os.getenv("RAG_FALLBACK_MODEL", "gemini-2.5-flash")

# Used by the agent (core/agent.py): try again a few times, waiting longer each time.
RETRY = types.HttpRetryOptions(attempts=4, initial_delay=2, max_delay=15)

# One try of the first model, at most 15 seconds; then up to three tries of the fallback.
PATIENCE = {CHAT_MODEL: types.HttpOptions(timeout=15_000, retry_options=types.HttpRetryOptions(attempts=1)),
            FALLBACK_MODEL: types.HttpOptions(timeout=20_000, retry_options=types.HttpRetryOptions(attempts=3))}

client = genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                      location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))


def part(data: bytes, mime: str) -> types.Part:
    return types.Part.from_bytes(data=data, mime_type=mime)


def ask(parts, schema=None):
    """Text and pictures in; text out, or parsed JSON when a schema is given."""
    for model, patience in PATIENCE.items():
        config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0), temperature=0,
                                             http_options=patience)
        if schema:
            config.response_mime_type, config.response_schema = "application/json", schema
        try:
            reply = client.models.generate_content(model=model, contents=parts, config=config)
        except (errors.APIError, httpx.HTTPError):
            if model == FALLBACK_MODEL:
                raise
            continue
        return json.loads(reply.text) if schema else reply.text.strip()
