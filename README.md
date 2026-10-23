# RAG that sees charts, connects documents, and stops ignoring half your data

A workshop from DevFest Troy, NY 2026. Ten messy files from Collar City Bikes, a made-up bike
share in Troy, and five kinds of retrieval-augmented generation that try to answer questions
about them: naive, text done right, multimodal, graph and agentic.

The files are messy on purpose. The refund policy exists in two versions. The ridership numbers,
the station table and the truck route are pictures. The winter hours are on a scanned flyer. A
rider's complaint is a voicemail, and the dock reset steps are a video. One answer needs facts
from four different files.

New to all of this? Follow the [attendee guide](guide/RAG-Attendee-Guide.pdf). It starts from an
empty laptop.

## What each kind gets right

Nine test questions, each needing something a simple pipeline lacks. Run `python evaluate.py`
to reproduce it.

| question | needs | naive | text | multimodal | graph | agent |
|---|---|---|---|---|---|---|
| How long do I have to report a broken bike to get a refund? | the newest version | ✓ | ✓ | ✓ | · | ✓ |
| What is the charge for a lost bike? | the newest version | ✓ | ✓ | ✓ | ✓ | ✓ |
| Which month had the most rides in 2025, and how many? | a chart | · | · | ✓ | ✓ | ✓ |
| Which station has the most docks? | a table that is a picture | · | · | ✓ | · | ✓ |
| In what order does the rebalancing truck visit the stations? | a diagram | · | · | ✓ | ✓ | ✓ |
| What are the winter opening hours? | a scanned page | · | · | ✓ | · | ✓ |
| Which bike lost power on the Green Island Bridge, and who reported it? | audio | · | · | ✓ | ✓ | ✓ |
| How do I free a bike that is stuck in a dock? | video | · | · | ✓ | · | ✓ |
| The dock from incident IR-2026-031 has just been repaired. Who has to sign it off, and what is their phone number? | facts from four files | · | · | · | ✓ | ✓ |
| **right** | | **2 / 9** | **2 / 9** | **8 / 9** | **5 / 9** | **9 / 9** |

Graph RAG says it cannot find an answer that lives in a picture, a scan or a video; that is not what it is for. The agent is right on all nine and the slowest, at several model calls per question. Gemini words answers differently each run, so a count can move by one.

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

On macOS and Linux the activate line is `source .venv/bin/activate` and the copy is
`cp .env.example .env`. Put your Google Cloud project ID in `.env`, sign in with
`gcloud auth application-default login`, then:

```
python check.py
```

The index is already in the repository. To build it again from the files:

```
python build_index.py
```

## Learn it step by step

Each step is one idea and runs on its own. Every step prints what happens in between, not just the answer.

| step | run | the idea |
|---|---|---|
| 1 | `python learn/1_read_the_files.py` | what a program sees in a PDF, and what it misses |
| 2 | `python learn/2_chunks_and_embeddings.py` | fixed and semantic chunks; embeddings; the modality gap |
| 3 | `python learn/3_naive_rag.py` | the tutorial pipeline, and where it goes wrong |
| 4 | `python learn/4_text_rag.py` | metadata filters, newest version only, reranking, citations with page numbers |
| 5 | `python learn/5_multimodal_rag.py` | charts, scans, audio and video, found by meaning and read by Gemini |
| 6 | `python learn/6_graph_rag.py` | a knowledge graph that connects facts across files |
| 7 | `python learn/7_agentic_rag.py` | an agent that chooses its own retrieval and filters |

Each one takes a question of your own, for example
`python learn/5_multimodal_rag.py "How do I free a bike that is stuck in a dock?"`.

## The app

```
python app/app.py
```

Open http://localhost:5050. One question goes to all five kinds at once. Click any citation to
see the page it came from with the passage highlighted, or to play a recording from the moment
it was cited.

## What is in the repository

```
data/       make_documents.py and the ten files it makes, in data/docs
core/       the building blocks, one idea per file
  documents.py   read PDFs, tables, pictures, audio and video into pieces
  chunking.py    fixed and semantic chunking
  gemini.py      the only file that calls Google: embeddings and answers
  store.py       a vector store with metadata filters, small enough to read
  graph.py       facts as links, and the walk across them
  rag.py         naive, text, multimodal and graph RAG, built from small steps
  agent.py       the agent, with four tools
  answer.py      answers that cite their sources
  questions.py   the test questions and how an answer is marked
build_index.py  read every file once and build the indexes
learn/      seven steps, one idea each
app/        the web app
evaluate.py the scorecard
tests/      fast checks with no model calls: python -m pytest
```

## Good to know

- Everything runs on Gemini through Vertex AI: `gemini-3.8-flash` answers, `gemini-embedding-2`
  embeds text, pictures, audio and video into one space.
- Building the index costs a few cents. Each question costs a fraction of a cent; the agent
  makes several calls per question.
- The vector store is numpy and JSON on purpose, so you can read all of it. The same ideas run
  unchanged on Vertex AI Vector Search, AlloyDB or pgvector.
- `data/make_documents.py` rebuilds the files. The voicemail uses the speech voice built into
  Windows; the other files build anywhere.
