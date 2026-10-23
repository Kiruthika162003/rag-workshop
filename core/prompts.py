"""Every prompt the workshop sends to Gemini, in one place, with the shape of the reply it expects.

Open this file when someone asks "but what does the model actually see?"
"""

from google.genai import types


def schema(**fields) -> types.Schema:
    """A JSON object with these fields, all required. STRING, INTEGER, BOOLEAN or a list like [STRING]."""
    def kind(spec):
        return types.Schema(type="ARRAY", items=kind(spec[0])) if isinstance(spec, list) else types.Schema(type=spec)
    return types.Schema(type="OBJECT", properties={name: kind(spec) for name, spec in fields.items()}, required=list(fields))


# ---------------------------------------------------------------- answering

NAIVE = "Answer the question using the context.\n\nContext:\n{context}\n\nQuestion: {question}"

ANSWER = """Answer the question using only the numbered sources below. They come from one research paper.
- You may know this paper already. Do not use what you know: if a detail is not in a source below, it is not there.
- Put the source number after every fact you use, like [2]. Figures and tables are sources too.
- When a figure or a table shows the answer, read it from the picture.
- If the sources do not contain the answer, say you could not find it. Never guess.
- Write plain text: no LaTeX, no markdown. Write numbers like 213 million or 10^6.
- Be brief: two to four sentences."""


# ---------------------------------------------------------------- before searching: work on the question

FILTERS = """Turn a question about the paper "Attention Is All You Need" into a search.
Its top-level sections are: {sections}.
- query: the question, rewritten as the words a matching passage would contain.
- section: the number or name of the section the question is about, if it names or clearly means one, else empty.
- first_page, last_page: the pages, if the question names them, else 0 and 0.

Question: {question}"""
FILTERS_REPLY = schema(query="STRING", section="STRING", first_page="INTEGER", last_page="INTEGER")

REWRITE = """Write {n} different ways to search a machine learning paper for the answer to this question.
Use different words each time: plain words, the paper's technical terms, and the name of the thing it measures.

Question: {question}"""
REWRITE_REPLY = schema(searches=["STRING"])

HYPOTHETICAL = """Write the passage that would answer this question, as it would appear in a machine learning
research paper. Make up the details; only the wording matters, because it is used to search, not to answer.
Two sentences.

Question: {question}"""


# ---------------------------------------------------------------- after searching: judge what came back

RERANK = """For each numbered passage, score from 0 to 10 how well it answers the question.
10 = it contains the answer. 0 = it is about something else. Similar words are not enough.

Question: {question}

{passages}"""
RERANK_REPLY = types.Schema(type="ARRAY", items=schema(n="INTEGER", score="INTEGER"))

# ---------------------------------------------------------------- evaluating: a model as the judge
# Each judge gets its instruction, then the numbered sources (figures as pictures), then what it judges.

FAITHFUL = """You check an answer against the sources it was written from.
Split the answer into its separate factual claims; leave out citations like [1]. For each claim, say whether
the sources support it. A claim is supported only if a source states or shows it; sounding right is not enough."""
FAITHFUL_REPLY = types.Schema(type="ARRAY", items=schema(claim="STRING", supported="BOOLEAN"))

USEFUL = """You judge retrieved sources. For each numbered source, say whether it is useful for answering the
question: it contains part of the answer, or something the answer needs. Being on the same topic is not enough."""
USEFUL_REPLY = types.Schema(type="ARRAY", items=schema(n="INTEGER", useful="BOOLEAN"))

ATTRIBUTED = """You check whether retrieved sources contain a reference answer.
Split the reference answer into its separate statements. For each one, say whether the sources state or show it."""
ATTRIBUTED_REPLY = types.Schema(type="ARRAY", items=schema(statement="STRING", found="BOOLEAN"))

ASKED = """Here is an answer. Write the {n} questions it most directly answers, as a person would ask them.
If the answer only says it could not find the answer, set noncommittal to true."""
ASKED_REPLY = schema(questions=["STRING"], noncommittal="BOOLEAN")


# ---------------------------------------------------------------- building the index, once

DESCRIBE = ("This is a figure or a table from the paper \"Attention Is All You Need\". Describe it in plain text, "
            "no markdown, so it can be found by search. Say what kind of figure it is in one sentence. Then copy "
            "out every label, number and word in it exactly, and say how the parts connect, in order.")

READ_FACTS = """Write down every fact this passage states, as subject, relation, object.
- Use the paper's names for things, like "Transformer (big)", "multi-head attention", "encoder".
- Relations are short and lower case, like "has", "uses", "scores", "trained on", "is part of".
- Leave "since" empty.
- Only facts the passage states. Nothing you know from elsewhere.

Passage, from "{title}", section "{section}":
{text}"""
READ_FACTS_REPLY = types.Schema(type="ARRAY", items=schema(subject="STRING", relation="STRING", object="STRING", since="STRING"))


# ---------------------------------------------------------------- the agent

AGENT = """You answer questions about the paper "Attention Is All You Need" using only what your tools return.
Pick the tools the question needs. Figures, diagrams, attention plots and tables are found with search_figures;
use look_at when you need a detail from one. The text is searched with search_text, which can be limited to a
section. Questions that join facts from several parts of the paper use search_graph, then check the key fact
with search_text. Call several tools at once when they are independent.
Answer in two to four sentences of plain text, no LaTeX and no markdown. After each fact put the ref it
came from in square brackets, like [t14] or [p2]. If the tools do not have the answer, say so. Never guess,
and do not use what you already know about this paper."""
