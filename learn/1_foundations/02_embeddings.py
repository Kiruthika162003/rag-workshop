"""Embeddings: text turned into a list of numbers, so that similar meanings end up close together.

Closeness is the cosine similarity: 1.0 is the same meaning, near 0 is unrelated. The model here,
bge-small, runs on your own computer: no cloud call, no cost, about 20 milliseconds a sentence.
Watch the sentence about the decoder: it does not answer a question about the encoder, yet it
scores almost as high. Similar is not the same as "answers the question", and that is why
retrieval needs more than similarity: filters, keywords and a reranker, all in 3_text_rag.

The code: core/embeddings.py

Run: python learn/1_foundations/02_embeddings.py
     python learn/1_foundations/02_embeddings.py "What hardware did they train on?"
"""

from core.embeddings import TEXT_MODEL, embed_passages, embed_query
from core.show import question, step

asked = question("How many layers does the encoder have?")
vector = embed_query(asked)
step(f"{TEXT_MODEL} turns it into {len(vector)} numbers. The first eight:")
print("   ", [round(float(x), 3) for x in vector[:8]])

sentences = [
    "The encoder is composed of a stack of N = 6 identical layers.",
    "The decoder is also composed of a stack of N = 6 identical layers.",
    "We trained our models on one machine with 8 NVIDIA P100 GPUs.",
    "The Law will never be perfect, but its application should be just.",
]
step("How close is each sentence to the question?")
for score, sentence in sorted(zip(embed_passages(sentences) @ vector, sentences), reverse=True):
    print(f"    {score:.3f}  {sentence}")
