"""Fast checks that need no Gemini calls: run with  python -m pytest"""

import numpy as np

from core.chunking import fixed_chunks, semantic_chunks
from core.documents import library, pieces
from core.graph import is_proper, key
from core.keywords import Keywords, words
from core.questions import QUESTIONS, correct
from core.search import fuse, with_neighbours
from core.store import Store

PAPER = "attention-is-all-you-need.pdf"


def test_the_paper_has_metadata():
    assert library()[0]["title"] == "Attention Is All You Need"


def test_figures_are_found_as_images_drawings_and_tables():
    found = pieces(PAPER)
    pages = [p["page"] for p in found if p["type"] == "picture"]
    assert 3 in pages and 4 in pages          # the architecture diagrams, stored as images
    assert {13, 14, 15} <= set(pages)         # the attention plots, drawn as vector lines
    assert 9 in pages                         # Table 3, kept as a picture too


def test_page_numbers_and_the_arxiv_stamp_are_dropped():
    text = " ".join(p["text"] for p in pieces(PAPER) if "text" in p)
    assert "arXiv:1706.03762" not in text


def test_fixed_chunks_cut_every_n_characters():
    assert fixed_chunks("abcdefghij", size=4) == ["abcd", "efgh", "ij"]


def test_semantic_chunks_start_at_headings_and_rows():
    parts = [{"page": 1, "type": "heading", "text": "A"}, {"page": 1, "type": "sentence", "text": "one"},
             {"page": 1, "type": "row", "text": "r1"}, {"page": 1, "type": "row", "text": "r2"}]
    vectors = [np.ones(3) / np.sqrt(3)] * 4
    assert [[p["text"] for p in chunk] for chunk in semantic_chunks(parts, vectors)] == [["A", "one"], ["r1"], ["r2"]]


def test_section_filter_matches_numbers_and_names():
    record = {"section": "5 Training / 5.2 Hardware and Schedule", "page": 7}
    assert Store.passes(record, section="5") and Store.passes(record, section="5.2") and Store.passes(record, section="training")
    assert not Store.passes(record, section="3") and not Store.passes(record, pages=(1, 6))


def test_references_are_not_indexed():
    assert not any(r["section"].startswith("References") for r in Store("text").records)


def test_keywords_drop_stop_words_and_keep_codes():
    assert words("What is warmup_steps in the big model?") == ["warmup", "steps", "big", "model"]


def test_keywords_tell_exact_numbers_apart():
    index = Keywords(["dff = 2048", "dmodel = 512", "dff = 4096 in the big model"])
    assert int(np.argmax(index.scores("2048"))) == 0


def test_fusion_rewards_chunks_both_searches_found():
    a, b, c = {"id": "a"}, {"id": "b"}, {"id": "c"}
    assert [chunk["id"] for chunk in fuse([a, b], [b, c])] == ["b", "a", "c"]


def test_neighbours_stay_on_the_same_page():
    store = Store("text")
    body = {r["text"].split("\n", 1)[1]: (r["doc"], r["page"]) for r in store.records}
    for chunk in store.records:
        for line in with_neighbours([chunk])[0]["text"].split("\n")[1:]:
            assert body[line] == (chunk["doc"], chunk["page"])


def test_proper_names_for_entity_resolution():
    assert key("Multi-Head Attention.") == "multi-head attention"
    assert is_proper("Transformer Big") and not is_proper("feed forward")


def test_answer_checker():
    mask = next(q for q in QUESTIONS if q["id"] == "mask")
    assert correct(mask, "Mask (opt.) is optional, then SoftMax.")
    assert not correct(mask, "SoftMax comes after the Mask step.")
    assert not correct(mask, "I could not find the mask or softmax.")
