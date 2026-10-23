"""Send the original: answer from the description, then from the picture itself.

Run: python learn/4_multimodal_rag/05_send_the_original.py
     python learn/4_multimodal_rag/05_send_the_original.py "What does the multi-head attention diagram concatenate?"
"""

import time

from core.answer import prompt
from core.gemini import ask
from core.search import pictures
from core.show import prompt as show_prompt, question, step

asked = question("In the scaled dot-product attention diagram, which step is optional, and what comes right after it?")
found = pictures(asked)

for originals in (False, True):
    message = prompt(asked, found, originals=originals)
    step("From the picture:" if originals else "From the text twin:")
    show_prompt([piece if not isinstance(piece, str) else piece[:400] for piece in message[1:]])
    started = time.time()
    print(f"\n    -> {ask(message)}\n       ({time.time() - started:.1f} s)")
