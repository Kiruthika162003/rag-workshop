"""The only file that calls Google: Gemini reads what was retrieved and writes the answer.

Embeddings run on your own computer (core/embeddings.py), so searching costs nothing. Gemini is
called only to read: answers, reranking, describing a figure once, and reading facts once.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parents[1] / ".env")

CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gemini-3.6-flash")

# A busy model answers 429 or 5xx. Try again a few times, waiting longer each time, then give up
# with a clear error instead of hanging: a workshop room cannot wait minutes for one answer.
RETRY = types.HttpRetryOptions(attempts=4, initial_delay=2, max_delay=15)

client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    http_options=types.HttpOptions(timeout=45_000, retry_options=RETRY),
)


def part(data: bytes, mime: str) -> types.Part:
    return types.Part.from_bytes(data=data, mime_type=mime)


def ask(parts, schema=None):
    """Text and pictures in; text out, or parsed JSON when a schema is given."""
    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0), temperature=0)
    if schema:
        config.response_mime_type, config.response_schema = "application/json", schema
    reply = client.models.generate_content(model=CHAT_MODEL, contents=parts, config=config)
    return json.loads(reply.text) if schema else reply.text.strip()
