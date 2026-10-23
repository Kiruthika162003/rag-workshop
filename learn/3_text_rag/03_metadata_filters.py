"""Metadata filters: keep only chunks from a section or pages, before ranking.

Say: Filter by where a chunk is, not by what it says. "Dropout" appears all over the paper;
     filtered to section 5, only the training setup can answer. The filter runs before ranking,
     so a ruled-out chunk can never win. Every vector database has this.

Code: Store.passes() and Store.top() in core/store.py

Run: python learn/3_text_rag/03_metadata_filters.py
     python learn/3_text_rag/03_metadata_filters.py "What does Table 3 vary?" --section "6.2"
     python learn/3_text_rag/03_metadata_filters.py "attention" --pages 13 15
"""

import argparse

from core.search import by_meaning
from core.show import chunks, step
from core.store import store

options = argparse.ArgumentParser()
options.add_argument("question", nargs="*", default=["What dropout rate is used?"])
options.add_argument("--section", default="5", help="a section number or name, like 5.4 or Training")
options.add_argument("--pages", nargs=2, type=int, help="first and last page")
args = options.parse_args()
asked = " ".join(args.question)
print(f"\nQuestion: {asked}")

step("No filter: the closest five chunks anywhere in the paper.")
chunks(by_meaning(asked, k=5))

filters = {"section": args.section, "pages": tuple(args.pages) if args.pages else None}
allowed = sum(store("text").passes(record, **filters) for record in store("text").records)
step(f"Filtered to section={args.section}" + (f", pages={args.pages}" if args.pages else "") +
     f": {allowed} of {len(store('text').records)} chunks may compete.")
chunks(by_meaning(asked, k=5, **filters))
