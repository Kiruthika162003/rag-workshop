# RAG that reads text, tables and figures

A workshop from DevFest Troy, NY 2026. One research paper, "Attention Is All You Need"
(Vaswani et al., Google, 2017), and six kinds of retrieval-augmented generation that try to answer
questions about it: naive, text done right, multimodal, graph, corrective and agentic.

The paper is a good test because its answers live in different places. Some are in the text, some
only in a table whose cells come out of the PDF scrambled, and some only in a figure: the
architecture diagrams are pictures, and the attention plots are vector drawings with no picture
inside them at all.

Embeddings run on your own computer, so searching costs nothing. Gemini is called only to read:
to rerank, to answer, and once per figure and chunk when the index is built.

## The test questions

Four questions, each needing something different. `python evaluate.py` asks all six kinds of RAG
all four, about forty small Gemini calls.

| question | needs |
|---|---|
| How many warmup steps did the learning rate schedule use? | an exact number |
| On what hardware was the big model trained, and for how long? | two passages |
| According to Table 3, how many parameters does the big model have? | a table |
| In the scaled dot-product attention diagram, which step is optional, and what comes right after it? | a figure |

## Start

```
git clone https://github.com/Kiruthika162003/rag-workshop.git
```

```
cd rag-workshop
```

```
python -m venv .venv
```

```
.venv\Scripts\activate
```

```
pip install -r requirements.txt
```

```
copy .env.example .env
```

```
python data/get_paper.py
```

On macOS and Linux the activate line is `source .venv/bin/activate` and the copy is
`cp .env.example .env`. Put your Google Cloud project ID in `.env`, sign in with
`gcloud auth application-default login`, then:

```
python check.py
```

The first run downloads the two embedding models from Hugging Face, about 730 MB, once.
The index is already in the repository. To build it again: `python build_index.py`. Every Gemini
answer from the last build is kept in `index/gemini-cache.json`, so a rebuild only pays for what changed.

## Learn it step by step

One folder per kind of RAG, one file per idea. Every file runs on its own, prints what happens
in between, names the code it teaches at the top, and takes a question of your own:
`python learn/3_text_rag/05_keyword_search.py "P100"`.
The last file in each folder puts that folder's ideas together. Files marked *local* make no
cloud calls at all.

**1_foundations**

| file | the idea |
|---|---|
| `01_read_the_files.py` | what a program sees in a PDF: text, images, drawings, tables *(local)* |
| `02_embeddings.py` | text as numbers; similar is not the same as "answers it" *(local)* |
| `03_vector_search.py` | vector search by hand: one matrix multiplication *(local)* |

**2_naive_rag**

| file | the idea |
|---|---|
| `naive_rag.py` | the tutorial pipeline, the whole prompt it sends, and where it goes wrong |

**3_text_rag**

| file | the idea |
|---|---|
| `01_chunking.py` | fixed chunks against semantic chunks, and where the topic changes *(local)* |
| `02_chunk_headers.py` | the section path in every chunk *(local)* |
| `03_metadata_filters.py` | filter by section and page before ranking *(local)* |
| `04_self_query.py` | the model reads the filters out of the question |
| `05_keyword_search.py` | BM25, for symbols and numbers that embeddings blur *(local)* |
| `06_hybrid_search.py` | vector and keyword search merged with reciprocal rank fusion *(local)* |
| `07_query_rewriting.py` | multi-query and HyDE |
| `08_reranking.py` | read every candidate and score it, drop the weak ones |
| `09_diversity.py` | maximal marginal relevance: no near-copies in the prompt *(local)* |
| `10_context_expansion.py` | search small chunks, answer with their neighbours *(local)* |
| `11_grounded_answer.py` | the exact answer prompt and numbered sources |
| `12_open_the_page.py` | a link to the page, and the passage highlighted on it |
| `text_rag.py` | all of it in one pipeline |

**4_multimodal_rag**

| file | the idea |
|---|---|
| `01_figures_in_a_pdf.py` | images, vector drawings and tables, each turned into a picture *(local)* |
| `02_one_space.py` | CLIP: pictures and words in one embedding space *(local)* |
| `03_two_scales.py` | why picture scores and text scores are fused by rank *(local)* |
| `04_describe.py` | a text twin for every figure, written once *(local)* |
| `05_send_the_original.py` | answering from the description, and from the picture itself |
| `multimodal_rag.py` | all of it in one pipeline |

**5_graph_rag**

| file | the idea |
|---|---|
| `01_read_facts.py` | the model turns a chunk into (subject, relation, object) facts |
| `02_entities.py` | one spelling per thing, names inside names, and hubs *(local)* |
| `03_explore_the_graph.py` | every fact about one thing, from every section *(local)* |
| `04_where_to_start.py` | names in the question, names in the best chunks *(local)* |
| `05_walk.py` | hop by hop across the paper *(local)* |
| `graph_rag.py` | all of it in one pipeline |

**6_agentic_rag**

| file | the idea |
|---|---|
| `01_corrective_rag.py` | grade what came back, check the answer, search again |
| `02_tools.py` | the tools are functions; their docstrings are the prompt |
| `03_agent.py` | an agent that picks its own tools and filters |

**7_evaluation**

| file | the idea |
|---|---|
| `01_retrieval_metrics.py` | hit@1, hit@4 and MRR for every way of searching *(local; --rerank adds Gemini)* |
| `02_faithfulness.py` | is every claim in the answer backed by a source? |

## The app

```
python app/app.py
```

Open http://localhost:5050. Four tabs, one scenario each: the answer is in the text, only in a figure,
joined across three places, or the question is vague. Ask sends the question to every kind of RAG in
the tab at once; Explain opens what the method is, when to choose it, and its architecture. Click any
citation to see
the page it came from, with the passage highlighted or the figure outlined.

## What is in the repository

```
data/       get_paper.py downloads the paper; its metadata is in data/docs
core/       the building blocks, one idea per file
  prompts.py     every prompt sent to Gemini, in one place
  gemini.py      the only file that calls Google
  embeddings.py  text and picture embeddings, on this computer
  documents.py   read the PDF into headings, sentences, table rows and figures
  chunking.py    fixed and semantic chunking
  store.py       a vector store with metadata filters, small enough to read
  keywords.py    BM25 keyword search
  search.py      vector, keyword and hybrid search, diversity, neighbours, figures
  query.py       self-query filters, multi-query, HyDE
  rerank.py      the reranker
  answer.py      answers that cite their sources, and the faithfulness check
  graph.py       facts as links, entity resolution, hubs, and the walk
  rag.py         naive, text, multimodal, graph and corrective RAG, each a few lines
  agent.py       the agent, with four tools
  questions.py   the test questions and how an answer is marked
  show.py        how the lessons print things
build_index.py  read the paper once and build the indexes
learn/      seven folders, one idea per file
app/        the web app
evaluate.py the scorecard
tests/      fast checks with no model calls: python -m pytest
```

## Good to know

- Text embeddings: `BAAI/bge-small-en-v1.5`. Picture embeddings: `openai/clip-vit-base-patch32`.
  Both run on the CPU.
- Answers: `gemini-3.6-flash` on Vertex AI. To try another model, set `RAG_CHAT_MODEL` in `.env`.
- A busy model is retried a few times, then the app shows the error instead of waiting.
- The vector store is numpy and JSON on purpose, so you can read all of it. The same ideas run
  unchanged on Vertex AI Vector Search, AlloyDB or pgvector.
- The paper is downloaded from arXiv rather than kept here. Google grants permission to reproduce
  its tables and figures for journalistic or scholarly works with attribution: Vaswani et al.,
  "Attention Is All You Need", NeurIPS 2017, https://arxiv.org/abs/1706.03762.
