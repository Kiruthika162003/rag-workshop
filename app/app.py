"""The web app: one question, five kinds of RAG side by side, and every source one click away.

Run: python app/app.py   then open http://localhost:5050
"""

import time

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from core import rag
from core.agent import ask_agent
from core.documents import DOCS, page_picture
from core.graph import key
from core.questions import QUESTIONS, correct

app = Flask(__name__)

KINDS = {
    "naive": ("Naive RAG", "fixed chunks, top three, no dates", rag.naive),
    "text": ("Text RAG", "semantic chunks, filters, reranking", rag.text),
    "multimodal": ("Multimodal RAG", "text plus pictures, audio and video", rag.multimodal),
    "graph": ("Graph RAG", "facts linked across files", rag.graph_rag),
    "agent": ("Agentic RAG", "an agent picks the tools", ask_agent),
}


@app.get("/")
def home():
    kinds = [{"id": k, "name": name, "about": about} for k, (name, about, _) in KINDS.items()]
    return render_template("index.html", kinds=kinds, questions=QUESTIONS)


@app.post("/ask")
def ask():
    body = request.get_json()
    started = time.time()
    result = KINDS[body["kind"]][2](body["question"])
    test = next((q for q in QUESTIONS if q["ask"] == body["question"]), None)
    result["right"] = correct(test, result["answer"]) if test else None
    result["seconds"] = round(time.time() - started, 1)
    if "graph" in result:
        said = key(result["answer"])
        result["graph"]["path"] = [f for f in result["graph"]["facts"] if key(f["subject"]) in said and key(f["object"]) in said]
    return jsonify(result)


@app.get("/page/<ref>")
def page(ref: str):
    source = rag.record(ref)
    png = page_picture(source["doc"], source["page"], source.get("find", ()), source.get("box"))
    return Response(png, mimetype="image/png")


@app.get("/files/<path:name>")
def files(name: str):
    return send_from_directory(DOCS, name)


if __name__ == "__main__":
    app.run(port=5050, threaded=True)
