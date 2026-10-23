"""Agentic RAG: an agent that decides which retrieval to use, with which filters, and when it has enough."""

import asyncio
import re
import threading
import uuid

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from core import prompts, rag
from core.answer import shown
from core.gemini import CHAT_MODEL, RETRY, ask, part
from core.rerank import rerank
from core.search import hybrid, pictures
from core.store import INDEX


def brief(record: dict) -> dict:
    return {"ref": record["id"], "page": record["page"], "section": record["section"],
            "type": record["type"], "text": record["text"][:700]}


def search_text(query: str, section: str = "") -> list[dict]:
    """Search the text of the paper.

    Args:
        query: what to look for, in plain words.
        section: only this section and the ones inside it, by number or name, like "3.2" or "Training". Empty for all.
    """
    return [brief(r) for r in rerank(query, hybrid(query, section=section or None))]


def search_figures(query: str) -> list[dict]:
    """Search the figures, diagrams, attention plots and tables. Returns a description of each match.
    Use look_at when you need a detail the description misses."""
    return [brief(r) for r in pictures(query, k=3)]


def look_at(ref: str, question: str) -> str:
    """Look at one figure or table found by search_figures, and answer a question about it.

    Args:
        ref: the ref of the figure, like "p0".
        question: what to find out from it.
    """
    item = rag.record(ref)
    return ask([part((INDEX / item["media"]).read_bytes(), item["mime"]), question])


def search_graph(question: str) -> dict:
    """Follow connections across the paper: what a part of the model contains, what was trained on what,
    which model scored what. Use it when the answer joins facts from different sections."""
    _, facts = rag.graph_facts(question)
    return {"facts": [f"{f['subject']} | {f['relation']} | {f['object']} [{f['source']}]" for f in facts]}


root_agent = Agent(
    name="research_agent",
    model=Gemini(model=CHAT_MODEL, retry_options=RETRY),
    generate_content_config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)),
    description="Answers questions about the paper Attention Is All You Need, from its text, figures and tables.",
    instruction=prompts.AGENT,
    tools=[search_text, search_figures, look_at, search_graph],
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
    cited = dict.fromkeys(re.findall(r"\b([tp]\d+)\b", text))
    return {"answer": text, "steps": steps, "sources": [shown(rag.record(ref)) for ref in cited]}


def ask_agent(question: str, wait: float | None = None) -> dict:
    """The agent's answer, or a TimeoutError after wait seconds, so a caller never hangs."""
    return asyncio.run_coroutine_threadsafe(run(question), loop).result(timeout=wait)
