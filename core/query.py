"""Work on the question before searching: self-query filters, multi-query, HyDE."""

from core import prompts
from core.embeddings import embed_passages
from core.gemini import ask
from core.search import by_meaning, fuse
from core.store import store


def filters_from(question: str) -> dict:
    sections = sorted({record["section"].split(" / ")[0] for record in store("text").records})
    reply = ask(prompts.FILTERS.format(sections="; ".join(sections), question=question), schema=prompts.FILTERS_REPLY)
    return {"query": reply["query"] or question, "section": reply["section"] or None,
            "pages": (reply["first_page"], reply["last_page"]) if reply["first_page"] else None}


def rewrites(question: str, n: int = 3) -> list[str]:
    return ask(prompts.REWRITE.format(n=n, question=question), schema=prompts.REWRITE_REPLY)["searches"][:n]


def multi_query(question: str, k: int = 12, **filters) -> list[dict]:
    searches = [question, *rewrites(question)]
    return fuse(*(by_meaning(search, k, **filters) for search in searches))[:k]


def hypothetical_answer(question: str) -> str:
    return ask(prompts.HYPOTHETICAL.format(question=question))


def hyde(question: str, k: int = 12, **filters) -> list[dict]:
    # The made-up answer is a passage, so it is embedded the way passages are, not the way questions are.
    return store("text").search(embed_passages([hypothetical_answer(question)])[0], k, **filters)
