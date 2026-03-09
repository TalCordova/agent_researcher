"""
Embedding-based utilities using sentence-transformers.

Provides two services:
  1. Semantic deduplication — remove near-duplicate papers by cosine similarity.
  2. Relevance scoring — score each paper against the research topic query.

GPU is used automatically if CUDA is available (sentence-transformers / torch handle this).
The model is loaded once and cached at module level to avoid repeated downloads.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from schemas.paper import Paper

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load and cache the embedding model (downloaded once, ~80 MB)."""
    return SentenceTransformer(_MODEL_NAME)


def _encode(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def deduplicate_papers(
    papers: list[Paper],
    threshold: float = 0.92,
) -> list[Paper]:
    """
    Remove near-duplicate papers using cosine similarity of title+abstract embeddings.

    When two papers exceed `threshold` similarity, the one with the higher
    citation count is kept (or the first-seen paper if counts are equal/missing).

    Returns a deduplicated list, preserving original order for kept papers.
    """
    if not papers:
        return papers

    # First pass: exact dedup via DOI / normalised title
    seen_keys: set[str] = set()
    unique: list[Paper] = []
    for p in papers:
        key = p.dedup_key()
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(p)

    if len(unique) <= 1:
        return unique

    texts = [p.text_for_embedding() for p in unique]
    embeddings = _encode(texts)  # shape (N, D), already L2-normalised

    # Cosine similarity matrix = dot product of normalised vectors
    sim_matrix: np.ndarray = embeddings @ embeddings.T

    keep = [True] * len(unique)
    for i in range(len(unique)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(unique)):
            if not keep[j]:
                continue
            if sim_matrix[i, j] >= threshold:
                # Keep the one with more citations; drop the other
                ci = unique[i].citation_count or 0
                cj = unique[j].citation_count or 0
                if cj > ci:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    return [p for p, k in zip(unique, keep) if k]


def score_relevance(papers: list[Paper], topic: str) -> list[Paper]:
    """
    Score each paper's relevance to `topic` using cosine similarity.

    Sets `paper.relevance_score` (0.0–1.0) on each paper in-place and
    returns the list sorted by descending relevance score.
    """
    if not papers:
        return papers

    texts = [p.text_for_embedding() for p in papers]
    all_texts = [topic] + texts
    embeddings = _encode(all_texts)

    topic_vec = embeddings[0]          # shape (D,)
    paper_vecs = embeddings[1:]        # shape (N, D)

    scores: np.ndarray = paper_vecs @ topic_vec  # cosine sim (already normalised)

    updated: list[Paper] = []
    for paper, score in zip(papers, scores.tolist()):
        updated.append(paper.model_copy(update={"relevance_score": float(score)}))

    return sorted(updated, key=lambda p: p.relevance_score or 0.0, reverse=True)
