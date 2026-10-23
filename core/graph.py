"""A knowledge graph: facts as (subject, relation, object) links, each one pointing back to its chunk.

Building it: Gemini reads every chunk and writes down the facts it states.
Using it: find where to start, then walk the links. Three hops go from the incident,
to its station, to the station's manager, to the manager's phone number.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from google.genai import types

from core.gemini import ask

INDEX = Path(__file__).parents[1] / "index"

FACTS = types.Schema(type="ARRAY", items=types.Schema(
    type="OBJECT",
    properties={
        "subject": types.Schema(type="STRING"),
        "relation": types.Schema(type="STRING"),
        "object": types.Schema(type="STRING"),
        "since": types.Schema(type="STRING", description="YYYY-MM-DD the fact starts being true, if the text says so, else empty"),
    },
    required=["subject", "relation", "object", "since"],
))

READ_FACTS = """Write down every fact this passage states, as subject, relation, object.
- Use people's full names, and station names without the word "station".
- Relations are short and lower case, like "manages", "has phone", "happened at", "replaces".
- Put a date in "since" only when the passage says when the fact starts.
- Only facts the passage states. Nothing you know from elsewhere.

Passage, from "{title}" dated {date}:
{text}"""


def key(name: str) -> str:
    """One spelling per thing: 'Monument Square station' and 'monument square' are the same node."""
    name = re.sub(r"\bstation\b", "", name.lower())
    return " ".join(re.sub(r"[^\w\s-]", " ", name).split())


def read_facts(chunk: dict) -> list[dict]:
    facts = ask(READ_FACTS.format(**chunk), schema=FACTS)
    return [{**fact, "source": chunk["id"]} for fact in facts if fact["subject"] and fact["object"]]


class Graph:
    def __init__(self):
        self.facts = json.loads((INDEX / "graph.json").read_text(encoding="utf-8"))
        self.links = defaultdict(list)                 # node -> facts that touch it
        for fact in self.facts:
            self.links[key(fact["subject"])].append(fact)
            self.links[key(fact["object"])].append(fact)

    @staticmethod
    def save(facts: list[dict]):
        (INDEX / "graph.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")

    def named_in(self, text: str) -> set[str]:
        """Nodes whose name appears in the text, like 'Monument Square' in a question."""
        text = key(text)
        return {node for node in self.links if len(node) > 3 and re.search(rf"\b{re.escape(node)}\b", text)}

    def from_chunks(self, chunk_ids: set[str]) -> set[str]:
        """Nodes mentioned in the chunks a vector search found: the way in when a question names nothing."""
        names = {f[side] for f in self.facts if f["source"] in chunk_ids for side in ("subject", "object")}
        # Start from names, not from words like "dock" or "manager" that link to everything.
        return {key(name) for name in names if len(name) > 4 and name[0].isupper() and not name[:4].isdigit()}

    def walk(self, start: set[str], hops: int = 3, limit: int = 40) -> list[dict]:
        """Every fact within a few links of the starting nodes, nearest first."""
        seen_nodes, seen_facts, found, frontier = set(start), set(), [], set(start)
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for fact in self.links.get(node, []):
                    marker = (fact["subject"], fact["relation"], fact["object"], fact["source"])
                    if marker in seen_facts:
                        continue
                    seen_facts.add(marker)
                    found.append(fact)
                    for side in (key(fact["subject"]), key(fact["object"])):
                        if side not in seen_nodes:
                            seen_nodes.add(side)
                            next_frontier.add(side)
            frontier = next_frontier
        return found[:limit]
