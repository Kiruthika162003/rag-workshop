"""How the lessons print things, so each lesson is about its idea and not about print formatting."""

import sys


def question(default: str) -> str:
    """The question typed after the lesson's name, or the lesson's own."""
    asked = " ".join(arg for arg in sys.argv[1:] if not arg.startswith("--")) or default
    print(f"\nQuestion: {asked}")
    return asked


def step(title: str):
    print(f"\n{title}")


def body(chunk: dict) -> str:
    """A chunk's text without its header line, on one line."""
    return " ".join(chunk["text"].split("\n", 1)[-1].split())


def where(chunk: dict) -> str:
    """Page and innermost section, like "p7  5.2 Hardware and Schedule"."""
    return f"p{chunk['page']:<2} {chunk['section'].split(' / ')[-1][:28]:28}"


def chunks(found: list[dict], score: str = "score", width: int = 66):
    for c in found:
        print(f"    {c[score]:>6}  {where(c)}  {body(c)[:width]}")


def sources(result: dict):
    print("\nAnswer:\n   ", result["answer"].replace("\n", "\n    "))
    print("\nSources:")
    for n, s in enumerate(result["sources"], start=1):
        print(f"    [{n}] {s['type']:8} page {s['page']}, {s['section']}")


def prompt(parts):
    """The message sent to the model. A picture is shown as a placeholder."""
    for piece in parts if isinstance(parts, list) else [parts]:
        text = piece if isinstance(piece, str) else f"<{piece.inline_data.mime_type}, {len(piece.inline_data.data) // 1024} KB>"
        print("    | " + text.replace("\n", "\n    | "))
