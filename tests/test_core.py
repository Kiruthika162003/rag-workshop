"""Fast checks that need no Gemini calls: run with  python -m pytest"""

import numpy as np

from core.chunking import fixed_chunks, semantic_chunks
from core.documents import library, pieces
from core.graph import Graph, key
from core.questions import QUESTIONS, correct
from core.store import Store


def test_every_file_has_metadata():
    files = library()
    assert len(files) == 10
    assert all(f["title"] and f["kind"] and f["date"] for f in files)


def test_table_rows_keep_their_column_names():
    rows = [p["text"] for p in pieces("station-operations-memo-2026.pdf") if p["type"] == "row"]
    assert "Station: Monument Square; Station manager: Priya Raman" in rows


def test_scanned_flyer_is_one_picture():
    assert [p["type"] for p in pieces("winter-hours-flyer.pdf")] == ["picture"]


def test_fixed_chunks_cut_every_n_characters():
    assert fixed_chunks("abcdefghij", size=4) == ["abcd", "efgh", "ij"]


def test_semantic_chunks_start_at_headings():
    parts = [{"page": 1, "type": "heading", "text": "A"}, {"page": 1, "type": "sentence", "text": "one"},
             {"page": 1, "type": "heading", "text": "B"}, {"page": 1, "type": "sentence", "text": "two"}]
    vectors = [np.ones(3) / np.sqrt(3)] * 4
    assert [[p["text"] for p in chunk] for chunk in semantic_chunks(parts, vectors)] == [["A", "one"], ["B", "two"]]


def test_newest_version_only_drops_the_2024_policy():
    store = Store("text")
    kept = {r["doc"] for r in store.records if store.passes(r, None, None, latest_only=True)}
    assert "refund-policy-2026.pdf" in kept and "refund-policy-2024.pdf" not in kept


def test_kind_and_date_filters():
    store = Store("text")
    kept = [r for r in store.records if store.passes(r, ["Minutes"], "2026-04-01", False)]
    assert kept and all(r["kind"] == "Minutes" for r in kept)


def test_graph_links_monument_square_to_its_new_manager():
    graph = Graph()
    assert key("Monument Square station") == "monument square"
    facts = graph.walk({"monument square"}, hops=1)
    assert any(f["subject"] == "Daniel Kim" and f["since"] == "2026-05-01" for f in facts)


def test_answer_checker():
    truck = next(q for q in QUESTIONS if q["id"] == "truck")
    assert correct(truck, "Monument Square, then RPI Student Union, Lansingburgh, Riverfront Park")
    assert not correct(truck, "Riverfront Park, then RPI Student Union, Lansingburgh, Monument Square")
