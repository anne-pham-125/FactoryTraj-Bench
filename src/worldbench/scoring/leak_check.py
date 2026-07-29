"""Train/test embedding-similarity leak check.

Callers supply embeddings themselves (from whichever encoder they used) -
this module doesn't know how to embed raw samples. A test sample whose
nearest train-embedding neighbor exceeds the similarity threshold is
flagged: it may mean the same clip/sample (or a near-duplicate) leaked
across the split.
"""
from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.95


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def check_leak(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    if len(train_embeddings) == 0 or len(test_embeddings) == 0:
        return {
            "threshold": threshold,
            "n_test": len(test_embeddings),
            "n_flagged": 0,
            "flagged_test_indices": [],
            "max_similarity": None,
        }

    sims = _cosine_similarity_matrix(np.asarray(test_embeddings), np.asarray(train_embeddings))
    max_sim_per_test = sims.max(axis=1)
    flagged = np.where(max_sim_per_test > threshold)[0]

    return {
        "threshold": threshold,
        "n_test": len(test_embeddings),
        "n_flagged": int(len(flagged)),
        "flagged_test_indices": flagged.tolist(),
        "max_similarity": float(max_sim_per_test.max()),
    }
