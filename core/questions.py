"""The questions every kind of RAG is tested on. Each one needs something a simple pipeline lacks."""

QUESTIONS = [
    {"id": "refund", "needs": "the newest version", "ask": "How long do I have to report a broken bike to get a refund?",
     "expect": ["60 minutes"]},
    {"id": "lost-bike", "needs": "the newest version", "ask": "What is the charge for a lost bike?",
     "expect": ["950"]},
    {"id": "busiest-month", "needs": "a chart", "ask": "Which month had the most rides in 2025, and how many?",
     "expect": ["August", "14,280"]},
    {"id": "most-docks", "needs": "a table that is a picture", "ask": "Which station has the most docks?",
     "expect": ["RPI Student Union", "30"]},
    {"id": "truck", "needs": "a diagram", "ask": "In what order does the rebalancing truck visit the stations?",
     "expect": ["Monument Square", "RPI Student Union", "Lansingburgh", "Riverfront Park"], "in_order": True},
    {"id": "winter", "needs": "a scanned page", "ask": "What are the winter opening hours?",
     "expect": ["6:00", "9:00"]},
    {"id": "voicemail", "needs": "audio", "ask": "Which bike lost power on the Green Island Bridge, and who reported it?",
     "expect": ["9012", "Sam Carter"]},
    {"id": "stuck-dock", "needs": "video", "ask": "How do I free a bike that is stuck in a dock?",
     "expect": ["red", "5 seconds"]},
    {"id": "sign-off", "needs": "facts from four files",
     "ask": "The dock from incident IR-2026-031 has just been repaired. Who has to sign it off, and what is their phone number?",
     "expect": ["Daniel Kim", "518-555-0158"]},
]


GAVE_UP = ("could not find", "cannot determine", "unable to", "do not specify", "does not specify")


def correct(question: dict, reply: str) -> bool:
    """Every expected phrase is in the answer, in order when the order is the point,
    and the answer does not say it gave up: listing every phone number is not finding the right one."""
    text = reply.replace("‑", "-").replace("**", "")
    found = [text.find(phrase) for phrase in question["expect"]]
    if -1 in found or any(phrase in text.lower() for phrase in GAVE_UP):
        return False
    return found == sorted(found) if question.get("in_order") else True
