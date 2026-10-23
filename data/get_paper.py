"""Download the paper this workshop searches.

"Attention Is All You Need", Vaswani et al., Google Brain and Google Research, 2017: fifteen pages of
text, tables, two architecture diagrams that are pictures, and attention plots drawn as vector graphics.
It is downloaded from arXiv rather than kept in the repository; the index of it is in index/.

Run: python data/get_paper.py
"""

import urllib.request
from pathlib import Path

URL = "https://arxiv.org/pdf/1706.03762v7"
PAPER = Path(__file__).parent / "docs" / "attention-is-all-you-need.pdf"

if PAPER.exists():
    print(f"Already here: {PAPER}")
else:
    PAPER.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(URL, PAPER)
    print(f"Downloaded {PAPER.stat().st_size // 1024} KB to {PAPER}")
