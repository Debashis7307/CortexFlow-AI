from __future__ import annotations

from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

BOT_PERSONAS: Dict[str, str] = {
    "Bot A (Tech Maximalist)": (
        "I believe AI and crypto will solve all human problems. I am highly optimistic "
        "about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
    ),
    "Bot B (Doomer / Skeptic)": (
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
    ),
    "Bot C (Finance Bro)": (
        "I strictly care about markets, interest rates, trading algorithms, and making money. "
        "I speak in finance jargon and view everything through the lens of ROI."
    ),
}


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def embed_text(text: str) -> np.ndarray:
    emb = _model.encode([text], normalize_embeddings=True)
    return np.asarray(emb[0], dtype=np.float32)


def create_vector_store() -> Tuple[faiss.IndexFlatIP, List[str], np.ndarray]:
    texts = list(BOT_PERSONAS.values())
    embeddings = _model.encode(texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = _l2_normalize(embeddings)

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)  # cosine similarity when vectors are normalized
    index.add(embeddings)

    return index, list(BOT_PERSONAS.keys()), embeddings