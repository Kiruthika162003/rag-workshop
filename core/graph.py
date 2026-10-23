"""A knowledge graph: facts as (subject, relation, object) links, each one pointing back to its chunk.

Building it: Gemini reads every chunk and writes down the facts it states.
Using it: find where to start, then walk the links. Three hops go from the incident,
to its station, to the station's manager, to the manager's phone number.
"""

import json
import re
from collections import defaultdict
from functools import cache, cached_property

from core import prompts
from core.gemini import ask
from core.store import INDEX

HUB = 20  # a node with more facts than this is a hub


def key(name: str) -> str:
    """One spelling per thing: 'Monument Square station' and 'monument square' are the same node."""
    name = re.sub(r"\bstation\b", "", name.lower())
    return " ".join(re.sub(r"[^\w\s-]", " ", name).split())


def is_name(text: str) -> bool:
    """Names like "Daniel Kim" or "Monument Square", not words like "dock" or "manager" that link to everything."""
    return len(text) > 4 and text[0].isupper() and not text[:4].isdigit()


def is_proper(text: str) -> bool:
    """Proper names of two words or more, every word capitalised: "Monument Square", "Daniel Kim"."""
    words = text.split()
    return len(words) > 1 and all(word[0].isupper() or word[0].isdigit() for word in words)


def read_facts(chunk: dict) -> list[dict]:
    facts = ask(prompts.READ_FACTS.format(**chunk), schema=prompts.READ_FACTS_REPLY)
    return [{**fact, "source": chunk["id"]} for fact in facts if fact["subject"] and fact["object"]]


class Graph:
    def __init__(self):
        self.facts = json.loads((INDEX / "graph.json").read_text(encoding="utf-8"))
        self.links = defaultdict(list)                 # node -> facts that touch it
        for fact in self.facts:
            self.links[key(fact["subject"])].append(fact)
            self.links[key(fact["object"])].append(fact)
        # Entity resolution: the model may write "Monument Square, dock 7" in one fact and "Monument Square"
        # in another. A node whose name contains a proper name is linked to that name as well.
        names = {key(n) for f in self.facts for n in (f["subject"], f["object"]) if is_proper(n)}
        self.part_of = {node: {n for n in names if n != node and re.search(rf"\b{re.escape(n)}\b", node)}
                        for node in self.links}

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
        return {key(name) for name in names if is_name(name)}

    def ends(self, fact: dict) -> set[str]:
        """The two nodes a fact links, and the known names inside them."""
        ends = {key(fact["subject"]), key(fact["object"])}
        return ends.union(*(self.part_of[end] for end in ends))

    @cached_property
    def hubs(self) -> set[str]:
        """Nodes linked to almost everything, like "we" or "transformer". Walking through one reaches
        every fact in the paper, so the walk may start at a hub but never passes through one."""
        return {node for node, facts in self.links.items() if len(facts) > HUB}

    def walk(self, start: set[str], hops: int = 3, limit: int = 60) -> list[dict]:
        """Every fact within a few links of the starting nodes, nearest first."""
        seen_nodes, seen_facts, found, frontier = set(start), set(), [], set(start)
        for hop in range(hops):
            next_frontier = set()
            for node in frontier if hop == 0 else frontier - self.hubs:
                for fact in self.links.get(node, []):
                    marker = (fact["subject"], fact["relation"], fact["object"], fact["source"])
                    if marker in seen_facts:
                        continue
                    seen_facts.add(marker)
                    found.append(fact)
                    next_frontier |= self.ends(fact) - seen_nodes
                    seen_nodes |= self.ends(fact)
            frontier = next_frontier
        return found[:limit]


@cache
def graph() -> Graph:
    return Graph()
