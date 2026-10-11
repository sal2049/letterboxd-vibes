"""Remove what should not drive the grouping from the film vectors.

The map is meant to show how films are talked about, so plain metadata is
regressed out first: each embedding dimension is fitted from the chosen
dataset columns with ridge regression and only the residual is kept. Every
signal here is a column of the dataset; nothing is inferred from word lists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from letterboxd_vibes.films import genre_matrix


def confound_matrix(films: pd.DataFrame, signals: list[str]) -> np.ndarray:
    """One standardised column (or a few) per requested signal."""
    columns: list[np.ndarray] = []
    for signal in signals:
        if signal == "year":
            year = films["year"].astype(float)
            year = (year.fillna(year.median()) - year.median()).to_numpy()
            columns += [year, year**2]
        elif signal == "popularity":
            columns += [np.log1p(films["likes_total"].to_numpy()), films["n_reviews"].to_numpy(float)]
        elif signal == "rating":
            rating = films["rating"].astype(float)
            columns.append(rating.fillna(rating.mean()).to_numpy())
        elif signal == "genre":
            columns += list(genre_matrix(films["genres"].tolist())[0].T)
        else:
            raise ValueError(f"Unknown deconfound signal: {signal}")
    if not columns:
        return np.zeros((len(films), 0))
    matrix = np.column_stack(columns).astype(np.float64)
    return (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-9)


def residualize(vectors: np.ndarray, confounds: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Unit vectors with the part predictable from `confounds` removed."""
    if confounds.shape[1] == 0:
        return vectors
    fitted = Ridge(alpha=alpha).fit(confounds, vectors).predict(confounds)
    residual = vectors - fitted
    return (residual / (np.linalg.norm(residual, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
