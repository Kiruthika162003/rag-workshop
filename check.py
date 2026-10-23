"""Check your setup one thing at a time. It stops at the first problem and says how to fix it.

Run: python check.py
"""

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent


def ok(what):
    print(f"  [ ok ]  {what}")


def stop(what, fix):
    print(f"  [ FIX ] {what}\n          {fix}")
    sys.exit(1)


print("\nChecking your setup\n")

if sys.version_info < (3, 11):
    stop(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old.", "Install Python 3.11 or newer from python.org/downloads.")
ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")

try:
    import core.rag  # noqa: F401
    import flask  # noqa: F401
    import google.adk  # noqa: F401
except ImportError as missing:
    stop(f"A package is missing: {missing.name}.", "Switch on the virtual environment, then run: pip install -r requirements.txt")
ok("The packages are installed")

if not (HERE / ".env").exists():
    stop("There is no .env file.", "Copy .env.example to .env and put your Project ID in it.")
from core.gemini import CHAT_MODEL, EMBED_MODEL, ask, embed  # reads .env

if os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id") == "your-project-id":
    stop(".env still has the placeholder project.", "Open .env and set GOOGLE_CLOUD_PROJECT to your Project ID.")
ok(f"The settings file points at project {os.getenv('GOOGLE_CLOUD_PROJECT')}")

if not shutil.which("gcloud"):
    stop("The gcloud command was not found.", "Install the Google Cloud CLI, then close this terminal and open a new one.")
ok("gcloud is installed")

import google.auth
from google.auth.exceptions import DefaultCredentialsError

try:
    google.auth.default()
except DefaultCredentialsError:
    stop("Programs on this computer are not signed in to Google Cloud.", "Run: gcloud auth application-default login")
ok("Programs on this computer are signed in")

try:
    reply = ask("Reply with the single word: ready")
except Exception as problem:
    text = str(problem)
    if "PERMISSION_DENIED" in text:
        stop("Gemini refused: this account has no access to the project.",
             "Run gcloud auth application-default login again and pick the account that owns the project.")
    if "billing" in text.lower():
        stop("Gemini refused: billing is off for this project.", "Switch on billing for the project in the Cloud console.")
    if "has not been used" in text or "is disabled" in text:
        stop("Vertex AI is not switched on for this project.", "Run: gcloud services enable aiplatform.googleapis.com")
    stop("Gemini did not answer.", text.splitlines()[0][:200])
ok(f"Gemini answered: {reply}   ({CHAT_MODEL})")

vector = embed("a bike share in Troy, New York")
ok(f"Embeddings work: {len(vector)} numbers per passage   ({EMBED_MODEL})")

if not (HERE / "index" / "graph.json").exists():
    stop("The index has not been built yet.", "Run: python build_index.py   (about four minutes)")
ok("The index is built")

print("\nEverything works. Start with: python learn/1_read_the_files.py\n")
