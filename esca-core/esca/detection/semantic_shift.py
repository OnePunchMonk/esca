from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, List, Optional, Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray:  # (n, d)
        ...


@dataclass(frozen=True)
class BowEmbedder:
    """Lightweight fallback embedder.

    Uses feature hashing into a fixed-size bag-of-words vector. This is not as
    semantically rich as sentence-transformers, but keeps `esca-core` usable
    without heavyweight dependencies.
    """

    dim: int = 2048

    _token_re: re.Pattern[str] = re.compile(r"[a-zA-Z0-9']+")

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._token_re.findall(text.lower()):
                # stable hash -> bucket
                digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                vectors[row, idx] += 1.0

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return vectors / norms


def try_sentence_transformers_embedder(
    embedder_name: str,
    device: str,
) -> Optional[Embedder]:
    try:
        from sentence_transformers import SentenceTransformer

        class _STEmbedder:
            def __init__(self) -> None:
                self._model = SentenceTransformer(embedder_name, device=device)

            def encode(self, texts: List[str]) -> np.ndarray:
                # sentence-transformers returns np.ndarray when convert_to_numpy=True
                return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        return _STEmbedder()
    except Exception:
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_words(text: str, chunk_size_tokens: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i : i + chunk_size_tokens])
        for i in range(0, len(words), chunk_size_tokens)
        if words[i : i + chunk_size_tokens]
    ]


def semantic_shift(
    *,
    pre_text: str,
    post_text: str,
    embedder: Embedder,
    chunk_size_tokens: int = 100,
    context_chunks: int = 3,
) -> float:
    pre_chunks = chunk_words(pre_text, chunk_size_tokens)[-context_chunks:]
    post_chunks = chunk_words(post_text, chunk_size_tokens)[:context_chunks]

    if not pre_chunks or not post_chunks:
        return 0.0

    pre_emb = embedder.encode(pre_chunks).mean(axis=0)
    post_emb = embedder.encode(post_chunks).mean(axis=0)

    return float(1.0 - cosine_sim(pre_emb, post_emb))
