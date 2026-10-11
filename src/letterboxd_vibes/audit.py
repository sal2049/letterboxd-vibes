"""Checks on the groups: how separate they are, and how much they track metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Association between two categorical variables: 0 unrelated, 1 identical."""
    table = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy()
    k = min(table.shape) - 1
    if k <= 0 or table.sum() == 0:
        return 0.0
    chi2 = chi2_contingency(table)[0]
    return float(np.sqrt(chi2 / (table.sum() * k)))


def _bins(values: pd.Series, q: int) -> np.ndarray:
    filled = values.astype(float).fillna(values.astype(float).median())
    return pd.qcut(filled, q=q, duplicates="drop", labels=False).to_numpy()


def separation(points: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Three standard measures of how well-separated the groups are.

    Silhouette: -1..1, higher is better, about 0 when groups touch.
    Davies-Bouldin: lower is better. Calinski-Harabasz: higher is better.
    Rows a method left unassigned (-1) are not counted.
    """
    kept = labels >= 0
    if kept.sum() < 3 or len(set(labels[kept])) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    x, y = points[kept], labels[kept]
    sample = min(5000, len(x))
    return {
        "silhouette": round(
            float(silhouette_score(x, y, metric="cosine", sample_size=sample, random_state=0)), 3
        ),
        "davies_bouldin": round(float(davies_bouldin_score(x, y)), 2),
        "calinski_harabasz": round(float(calinski_harabasz_score(x, y)), 1),
    }


def metadata_association(films: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """How strongly group membership tracks each thing it is not supposed to."""
    primary_genre = np.array([g[0] if g else "none" for g in films["genres"]])
    return {
        "release_decade": round(cramers_v(labels, _bins(films["year"], 10)), 3),
        "primary_genre": round(cramers_v(labels, primary_genre), 3),
        "rating_quintile": round(cramers_v(labels, _bins(films["rating"], 5)), 3),
        "popularity_quintile": round(cramers_v(labels, _bins(films["likes_total"], 5)), 3),
    }
