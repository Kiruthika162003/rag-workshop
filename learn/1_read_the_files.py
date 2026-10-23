"""Step 1. What a program actually sees in your files.

A text-only pipeline reads the text layer of each PDF page. Charts, pasted tables, scans,
recordings and videos have no text layer, so for it they do not exist.

Run: python learn/1_read_the_files.py
"""

from collections import Counter

from core.documents import library, pieces, raw_text

print(f"\n{'file':36} {'kind':10} {'date':11} {'naive reader gets':>18}   careful reader finds")
for file in library():
    characters = sum(len(text.strip()) for _, text in raw_text(file["doc"]))
    found = dict(Counter(piece["type"] for piece in pieces(file["doc"])))
    print(f"{file['doc']:36} {file['kind']:10} {file['date']:11} {characters:>12} chars   {found}")

print("\nThe scanned winter flyer, as a naive reader sees it:")
print("   ", " ".join(raw_text("winter-hours-flyer.pdf")[0][1].split()))

print("\nA table row, read as one sentence that keeps its column names:")
print("   ", next(p["text"] for p in pieces("station-operations-memo-2026.pdf") if p["type"] == "row"))

print("\nTwo versions of one policy share a series, so we can tell which is newer:")
for file in library():
    if file["series"] == "Refund Policy":
        print(f"    {file['date']}  {file['title']}")
