"""Do a film's reviews agree with each other?

Split each film's reviews into two random halves and average each half into a
vector. If reviews carry a consistent signal about the film, a film's two
halves should be closer to each other than to other films. This needs no
labels, so it can score an embedding model on exactly the text the map uses.
"""

from __future__ import annotations

import numpy as np


def _half_means(film_of_review: np.ndarray, vectors: np.ndarray, rng: np.random.Generator):
    n_films = int(film_of_review.max()) + 1
    # A random order, then alternate sides within each film.
    order = rng.permutation(len(film_of_review))
    order = order[np.argsort(film_of_review[order], kind="stable")]
    starts = np.searchsorted(film_of_review[order], np.arange(n_films))
    side = (np.arange(len(order)) - np.repeat(starts, np.diff([*starts, len(order)]))) % 2
    halves = []
    for s in (0, 1):
        total = np.zeros((n_films, vectors.shape[1]), dtype=np.float64)
        np.add.at(total, film_of_review[order[side == s]], vectors[order[side == s]])
        total -= total.mean(axis=0)
        halves.append(total / (np.linalg.norm(total, axis=1, keepdims=True) + 1e-9))
    return halves


def split_half_agreement(film_of_review: np.ndarray, vectors: np.ndarray, repeats: int = 5, seed: int = 0):
    """(agreement, top10) averaged over `repeats` random splits.

    agreement: for a film's first half, the share of other films whose second
    half is less similar than the film's own second half. 0.5 is chance, 1.0
    means every film's halves find each other first.
    top10: how often a film's own second half is among the ten most similar.

    Every film needs at least two reviews.
    """
    rng = np.random.default_rng(seed)
    agreement, top10 = [], []
    for _ in range(repeats):
        a, b = _half_means(film_of_review, vectors, rng)
        similarity = a @ b.T
        rank = (similarity > np.diag(similarity)[:, None]).sum(axis=1)
        agreement.append(1 - rank.mean() / (len(a) - 1))
        top10.append((rank < 10).mean())
    return float(np.mean(agreement)), float(np.mean(top10))
