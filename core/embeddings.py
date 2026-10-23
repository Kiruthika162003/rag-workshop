"""Embeddings that run on your own computer: no cloud calls, no cost per search.

    text      BAAI/bge-small-en-v1.5         384 numbers per passage; small, fast, and good
    pictures  openai/clip-vit-base-patch32   pictures and short texts in one 512-number space

Both download once from Hugging Face (about 130 MB and 600 MB) and run on the CPU.
"""

import io
from functools import cache, lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from transformers.utils import logging

logging.set_verbosity_error()
logging.disable_progress_bar()

TEXT_MODEL = "BAAI/bge-small-en-v1.5"
PICTURE_MODEL = "openai/clip-vit-base-patch32"
# bge was trained with this in front of search questions, and nothing in front of the passages.
QUERY = "Represent this sentence for searching relevant passages: "


def load(kind, name: str, **options):
    """From the copy on this computer if there is one; from Hugging Face the first time."""
    try:
        return kind.from_pretrained(name, local_files_only=True, **options)
    except OSError:
        return kind.from_pretrained(name, **options)


@cache
def text_model():
    return load(AutoTokenizer, TEXT_MODEL), load(AutoModel, TEXT_MODEL).eval()


@cache
def picture_model():
    # This model ships its weights as pytorch_model.bin; asking for that file stops a second 600 MB download.
    return load(CLIPProcessor, PICTURE_MODEL), load(CLIPModel, PICTURE_MODEL, use_safetensors=False).eval()


def unit(vectors: torch.Tensor) -> np.ndarray:
    """Length 1, so a dot product is the cosine similarity."""
    return torch.nn.functional.normalize(vectors, dim=-1).numpy()


@torch.no_grad()
def embed_passages(texts: list[str], batch: int = 32) -> np.ndarray:
    tokenizer, model = text_model()
    vectors = []
    for start in range(0, len(texts), batch):
        words = tokenizer(texts[start:start + batch], padding=True, truncation=True, max_length=512, return_tensors="pt")
        vectors.append(model(**words).last_hidden_state[:, 0])  # bge uses the vector of the first token
    return unit(torch.cat(vectors))


@lru_cache(maxsize=512)
def embed_query(question: str) -> np.ndarray:
    return embed_passages([QUERY + question])[0]


@torch.no_grad()
def embed_picture(png: bytes) -> np.ndarray:
    processor, model = picture_model()
    pixels = processor(images=Image.open(io.BytesIO(png)).convert("RGB"), return_tensors="pt")
    return unit(model.get_image_features(**pixels).pooler_output)[0]


@lru_cache(maxsize=512)
@torch.no_grad()
def embed_query_for_pictures(question: str) -> np.ndarray:
    processor, model = picture_model()
    words = processor(text=[question], padding=True, truncation=True, return_tensors="pt")
    return unit(model.get_text_features(**words).pooler_output)[0]
