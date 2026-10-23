"""Send the original: let the model look at the figure, not at a summary of it.

A description is only as good as what it chose to copy out; the picture always has everything.
Here both can answer, because the description happened to copy "Mask (opt.)"; the picture is the
safety net for the detail a description leaves out. The cost is a bigger request: tens of kilobytes
of picture instead of a few lines of text. This lesson answers both ways and times both.

The code: prompt() and answer() in core/answer.py

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
