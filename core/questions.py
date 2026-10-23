"""The questions every kind of RAG is tested on. Each one needs something a simple pipeline lacks.

An expected phrase may list spellings separated by |, like "2048|2,048".
"""

QUESTIONS = [
    {"id": "warmup", "needs": "an exact number", "ask": "How many warmup steps did the learning rate schedule use?",
     "expect": ["4000|4,000"]},
    {"id": "hardware", "needs": "two passages", "ask": "On what hardware was the big model trained, and for how long?",
     "expect": ["P100", "3.5 days"]},
    {"id": "parameters", "needs": "a table", "ask": "According to Table 3, how many parameters does the big model have?",
     "expect": ["213"]},
    {"id": "mask", "needs": "a figure", "ask": "In the scaled dot-product attention diagram, which step is optional, and what comes right after it?",
     "expect": ["mask", "softmax"], "in_order": True},
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


def correct(question: dict, reply: str) -> bool:
    """Every expected phrase is in the answer, in order when the order is the point,
    and the answer does not say it gave up."""
    text = reply.replace("‑", "-").replace("**", "").lower()
    found = [min((i for i in (text.find(s.lower()) for s in phrase.split("|")) if i >= 0), default=-1)
             for phrase in question["expect"]]
    if -1 in found or any(phrase in text for phrase in GAVE_UP):
        return False
    return found == sorted(found) if question.get("in_order") else True
