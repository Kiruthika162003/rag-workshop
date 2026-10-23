"""The questions every kind of RAG is tested on. Each one needs something a simple pipeline lacks.

An expected phrase may list spellings separated by |, like "2048|2,048".
"""

QUESTIONS = [
    {"id": "hardware", "needs": "the text", "ask": "On what hardware was the big model trained, and for how long?",
     "expect": ["P100", "3.5 days"]},
    {"id": "colour", "needs": "a figure", "ask": "In the Transformer architecture diagram, what colour are the Feed Forward blocks?",
     "expect": ["blue"], "from_picture": True},
    {"id": "chain", "needs": "three places joined",
     "ask": "How many parameters does the model that scored 28.4 BLEU on English-to-German have, and how long did it train?",
     "expect": ["213", "3.5 days"]},
    {"id": "vague", "needs": "a vague question", "ask": "what's the optional bit in the attention picture for?",
     "expect": ["mask", "leftward|auto-regressive|autoregressive|future|subsequent|illegal"]},
]


# Retrieval tests: a question in plain words, and words only the chunk that answers it contains.
FINDS = [
    ("how big is the hidden layer inside the feed-forward part?", "dff = 2048"),
    ("how long did the large model train?", "3.5 days"),
    ("what GPUs did they use?", "P100"),
    ("how many layers does the encoder have?", "N = 6 identical layers"),
    ("how were English-German sentences split into tokens?", "byte-pair encoding"),
    ("vocabulary size for English-French", "32000 word-piece"),
    ("learning rate warmup", "warmup_steps = 4000"),
    ("why divide the dot products by the square root of dk?", "extremely small gradients"),
    ("what beam size was used?", "beam size of 4"),
    ("dropout rate of the base model", "Pdrop = 0.1"),
    ("what does label smoothing do to perplexity?", "hurts perplexity"),
    ("BLEU on English to French", "41.8"),
    ("why sine and cosine for positions?", "extrapolate to sequence lengths"),
    ("how many attention heads?", "h = 8"),
    ("results on constituency parsing", "Wall Street Journal"),
    ("what optimizer did they train with?", "Adam optimizer"),
]


GAVE_UP = ("could not find", "cannot determine", "unable to", "do not specify", "does not specify", "not mentioned")


def from_memory(question: dict, sources: list[dict]) -> bool:
    """A famous paper is in the model's training data. An answer that only a picture holds, given
    with no picture among the sources, came from the model's memory, not from retrieval."""
    return question.get("from_picture", False) and not any(s["type"] == "picture" for s in sources)


def correct(question: dict, reply: str) -> bool:
    """Every expected phrase is in the answer, in order when the order is the point,
    and the answer does not say it gave up."""
    text = reply.replace("‑", "-").replace("**", "").lower()
    found = [min((i for i in (text.find(s.lower()) for s in phrase.split("|")) if i >= 0), default=-1)
             for phrase in question["expect"]]
    if -1 in found or any(phrase in text for phrase in GAVE_UP):
        return False
    return found == sorted(found) if question.get("in_order") else True
