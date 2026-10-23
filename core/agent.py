"""Agentic RAG: an agent that decides which retrieval to use, with which filters, and when it has enough."""

import asyncio
import re
import threading
import uuid
from datetime import date

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from core import rag
from core.gemini import CHAT_MODEL, RETRY, ask, part
from core.store import INDEX


def brief(record: dict) -> dict:
    return {"ref": record["id"], "title": record["title"], "date": record["date"], "page": record["page"],
            "type": record["type"], "text": record["text"][:700]}


def search_documents(query: str, kind: str = "", after: str = "", newest_version_only: bool = True) -> list[dict]:
    """Search the text of all company documents.

    Args:
        query: what to look for, in plain words.
        kind: only documents of this kind: Policy, Report, Memo, Minutes, Directory, Incident. Empty for all.
        after: only documents dated on or after this day, YYYY-MM-DD. Empty for any date.
        newest_version_only: skip older versions of a document that has been replaced.
    """
    found = rag.find_text(query, kinds=[kind] if kind else None, after=after or None, latest_only=newest_version_only)
    return [brief(r) for r in rag.rerank(query, found)]


def search_media(query: str) -> list[dict]:
    """Search pictures, charts, scanned pages, voicemails and videos. Returns a description or a
    timestamped transcript of each match. Use look_at when you need a detail the description misses."""
    return [brief(r) for r in rag.find_media(query, k=3)]


def look_at(ref: str, question: str) -> str:
    """Look at one picture, recording or video found by search_media, and answer a question about it.

    Args:
        ref: the ref of the media, like "m0".
        question: what to find out from it.
    """
    media = rag.record(ref)
    return ask([part((INDEX / media["media"]).read_bytes(), media["mime"]), question])


def search_graph(question: str) -> dict:
    """Follow connections across documents: who manages what, who to call, what happened where.
    Use it when the answer needs facts from more than one document."""
    _, facts = rag.graph_facts(question)
    return {"facts": [f"{f['subject']} | {f['relation']} | {f['object']}{' | since ' + f['since'] if f['since'] else ''} [{f['source']}]"
                      for f in facts]}


root_agent = Agent(
    name="research_agent",
    model=Gemini(model=CHAT_MODEL, retry_options=RETRY),
    generate_content_config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)),
    description="Answers questions about Collar City Bikes from its documents, pictures, recordings and videos.",
    instruction=(
        f"Today is {date.today().isoformat()}. "
        "You answer questions about Collar City Bikes using only what your tools return.\n"
        "Pick the tools the question needs. Numbers that live in charts or tables, scanned pages, "
        "voicemails and videos are found with search_media. Policies and reports with search_documents. "
        "Questions that join facts from several documents with search_graph, then check the key fact "
        "with search_documents. Call several tools at once when they are independent.\n"
        "Prefer the most recent document when sources disagree, and say so.\n"
        "Answer in two to four sentences. After each fact put the ref it came from in square brackets, "
        "like [t14] or [m4, 0:23]. If the tools do not have the answer, say so. Never guess."
    ),
    tools=[search_documents, search_media, look_at, search_graph],
)


runner = Runner(agent=root_agent, app_name="rag", session_service=InMemorySessionService(), auto_create_session=True)

# One event loop for the agent, in a side thread, so callers from any thread share it.
loop = asyncio.new_event_loop()
threading.Thread(target=loop.run_forever, daemon=True).start()


async def run(question: str) -> dict:
    message = types.Content(role="user", parts=[types.Part(text=question)])
    steps, text = [], ""
    async for event in runner.run_async(user_id="you", session_id=uuid.uuid4().hex, new_message=message):
        for piece in event.content.parts if event.content else []:
            if piece.function_call:
                steps.append({"tool": piece.function_call.name, "args": dict(piece.function_call.args)})
            elif piece.text and not piece.thought:
                text = piece.text
    cited = dict.fromkeys(re.findall(r"\b([tm]\d+)\b", text))
    return {"answer": text, "steps": steps, "sources": [rag.shown(rag.record(ref)) for ref in cited]}


def ask_agent(question: str) -> dict:
    return asyncio.run_coroutine_threadsafe(run(question), loop).result()
