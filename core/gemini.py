"""Every call to Google goes through this file: one client, one embedding model, one chat model."""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parents[1] / ".env")

CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gemini-3.8-flash")
# One embedding space for text, pictures, audio and video: a typed question can find a chart or a voicemail.
EMBED_MODEL = "gemini-embedding-2"
DIMENSIONS = 768

# A busy model answers 429 or 5xx. Wait and try again, longer each time, up to six tries.
RETRY = types.HttpRetryOptions(attempts=6, initial_delay=3, max_delay=40)

client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    http_options=types.HttpOptions(timeout=90_000, retry_options=RETRY),
)


def part(data: bytes, mime: str) -> types.Part:
    return types.Part.from_bytes(data=data, mime_type=mime)


def embed(item, mime: str = "") -> np.ndarray:
    """Text, or the bytes of a picture, recording or video, as a unit-length vector."""
    content = part(item, mime) if isinstance(item, bytes) else item
    reply = client.models.embed_content(model=EMBED_MODEL, contents=[content],
                                        config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS))
    vector = np.array(reply.embeddings[0].values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def ask(parts, schema=None):
    """Text, pictures, audio or video in; text out, or parsed JSON when a schema is given."""
    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0), temperature=0)
    if schema:
        config.response_mime_type, config.response_schema = "application/json", schema
    reply = client.models.generate_content(model=CHAT_MODEL, contents=parts, config=config)
    return json.loads(reply.text) if schema else reply.text.strip()
