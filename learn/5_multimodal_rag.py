"""Step 5. Multimodal RAG: search pictures, scans, audio and video by meaning, and let Gemini see them.

The question is embedded once and compared with both indexes: text chunks and media. The media that
match are sent to Gemini as they are, a chart as an image, a voicemail as audio, so it reads the
answer from the original, not from a summary.

Run: python learn/5_multimodal_rag.py
     python learn/5_multimodal_rag.py "Which bike lost power on the Green Island Bridge, and who reported it?"
     python learn/5_multimodal_rag.py "How do I free a bike that is stuck in a dock?"
"""

import sys
from pathlib import Path

from core.answer import answer
from core.documents import page_picture
from core.rag import find_media, find_text, rerank

question = " ".join(sys.argv[1:]) or "Which month had the most rides in 2025, and how many?"
print(f"\nQuestion: {question}")

passages = rerank(question, find_text(question, latest_only=True), keep=3)
media = find_media(question)
print("\nMedia that match, found by meaning:")
for m in media:
    print(f"    {m['score']:.3f}  {m['type']:8} {m['doc']}:  {' '.join(m['text'].split())[:80]}...")

result = answer(question, passages + media)
print("\nAnswer:\n   ", result["answer"])

print("\nSources:")
for n, source in enumerate(result["sources"], start=1):
    where = f"page {source['page']}" if source["type"] in ("text", "picture") else "play it: the answer gives the time"
    print(f"    [{n}] {source['type']:8} {source['title']}, {where}")

pictures = [m for m in media if m["type"] == "picture"]
if pictures:
    Path("out").mkdir(exist_ok=True)
    Path("out/picture.png").write_bytes(page_picture(pictures[0]["doc"], pictures[0]["page"], box=pictures[0]["box"]))
    print("\nThe page the picture is on, with the picture outlined: out/picture.png")
