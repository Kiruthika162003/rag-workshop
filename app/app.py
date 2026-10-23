"""The web app: four tabs, one scenario each. A question goes to every kind of RAG in the tab at once,
side by side, with every source one click away and the architecture of each one in a side panel.

Run: python app/app.py   then open http://localhost:5050
"""

import time

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from stages import STAGES
from werkzeug.exceptions import HTTPException

from core import gemini, rag
from core.agent import ask_agent
from core.documents import DOCS, page_picture
from core.embeddings import embed_query, embed_query_for_pictures
from core.graph import graph, key
from core.questions import QUESTIONS, correct, from_memory
from core.store import store

app = Flask(__name__)

KINDS = {
    "text": ("Text RAG", "hybrid search, reranking, cited pages", rag.text),
    "multimodal": ("Multimodal RAG", "text plus figures, seen as pictures", rag.multimodal),
    "graph": ("Graph RAG", "facts linked across sections", rag.graph_rag),
    "agent": ("Agentic RAG", "an agent picks its own tools", lambda question: ask_agent(question, wait=50)),
}


def warm_up():
    """Load both embedding models, every index and the graph, and open the connection to Gemini,
    before the first question rather than during it."""
    gemini.ask("Reply with one word: ready")
    embed_query("warm up")
    embed_query_for_pictures("warm up")
    for name in ("naive", "text", "pictures", "descriptions"):
        store(name).keywords
    graph().hubs


@app.get("/")
def home():
    questions = {q["id"]: q for q in QUESTIONS}
    stages = [{**stage, "ask": questions[stage["question"]]["ask"],
               "cards": [{"id": k, "name": KINDS[k][0], "about": KINDS[k][1]} for k in stage["kinds"]]} for stage in STAGES]
    return render_template("index.html", stages=stages)


@app.post("/ask")
def ask():
    body = request.get_json()
    started = time.time()
    result = KINDS[body["kind"]][2](body["question"])
    test = next((q for q in QUESTIONS if q["ask"] == body["question"]), None)
    result["right"] = correct(test, result["answer"]) if test else None
    if result["right"] and from_memory(test, result["sources"]):
        result["right"], result["note"] = False, "Not in its sources: this came from the model's memory of the paper."
    result["seconds"] = round(time.time() - started, 1)
    if "graph" in result:
        # Draw the facts the answer walked along; an answer that found nothing draws nothing.
        said = "" if "could not find" in result["answer"].lower() else key(result["answer"])
        result["graph"]["path"] = [f for f in result["graph"]["facts"] if key(f["subject"]) in said and key(f["object"]) in said]
    return jsonify(result)


@app.errorhandler(Exception)
def failed(problem):
    """A model that is busy or timed out comes back to the page as a short message, not an HTML error page."""
    if isinstance(problem, HTTPException):
        return problem
    return jsonify({"error": f"{type(problem).__name__}: {problem}"[:300]}), 500


@app.get("/page/<ref>")
def page(ref: str):
    source = rag.record(ref)
    png = page_picture(source["doc"], source["page"], source.get("find", ()), source.get("box"))
    return Response(png, mimetype="image/png")


@app.get("/files/<path:name>")
def files(name: str):
    return send_from_directory(DOCS, name)


if __name__ == "__main__":
    warm_up()
    print("Ready: open http://localhost:5050")
    app.run(port=5050, threaded=True)
