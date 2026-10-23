"""Download the paper from arXiv.

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
