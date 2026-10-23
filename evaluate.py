"""The scorecard: every kind of RAG answers every test question, and we count who got it right.

Run: python evaluate.py                       all kinds, all questions (about 5 minutes)
     python evaluate.py --kinds naive text    only some kinds
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core import rag
from core.agent import ask_agent
from core.questions import QUESTIONS, correct

KINDS = {"naive": rag.naive, "text": rag.text, "multimodal": rag.multimodal, "graph": rag.graph_rag, "agent": ask_agent}


def score(kind: str, question: dict) -> dict:
    started = time.time()
    try:
        reply = KINDS[kind](question["ask"])["answer"]
    except Exception as problem:
        reply = f"error: {problem}"
    return {"kind": kind, "id": question["id"], "answer": reply, "right": correct(question, reply),
            "seconds": round(time.time() - started, 1)}


def main():
    options = argparse.ArgumentParser()
    options.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    kinds = options.parse_args().kinds

    jobs = [(kind, question) for question in QUESTIONS for kind in kinds]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(lambda job: score(*job), jobs))

    # Running only some kinds keeps the saved results of the others.
    saved = Path("index/scorecard.json")
    earlier = [r for r in json.loads(saved.read_text(encoding="utf-8")) if r["kind"] not in kinds] if saved.exists() else []
    saved.write_text(json.dumps(earlier + results, indent=1), encoding="utf-8")

    right = {(r["kind"], r["id"]): r["right"] for r in results}
    print(f"\n{'question':16}{'needs':26}" + "".join(f"{kind:>12}" for kind in kinds))
    for question in QUESTIONS:
        marks = "".join(f"{'yes' if right[kind, question['id']] else '-':>12}" for kind in kinds)
        print(f"{question['id']:16}{question['needs']:26}{marks}")
    totals = "".join(f"{sum(right[kind, q['id']] for q in QUESTIONS):>9} / {len(QUESTIONS)}" for kind in kinds)
    print(f"{'right':42}{totals}\n\nEvery answer is in index/scorecard.json")


if __name__ == "__main__":
    main()
