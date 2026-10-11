"""From stored review vectors to the film vectors the map is built from."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from letterboxd_vibes.clustering import unit_rows
from letterboxd_vibes.config import embeddings_dir
from letterboxd_vibes.deconfound import confound_matrix, residualize
from letterboxd_vibes.films import aggregate_films


def load_reviews(lang: str, model: str) -> pd.DataFrame:
    folder = embeddings_dir(lang, model)
    parts = sorted(folder.glob("part-*.parquet"))
    if not parts:
        raise SystemExit(f"No embeddings in {folder}. Run 03_embed_reviews.py first.")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta["model"] != model:
        raise SystemExit(f"{folder} holds vectors from {meta['model']}, not {model}.")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def film_points(reviews: pd.DataFrame, deconfound: list[str]):
    """The film table, a unit vector per film, a unit vector per review, and
    the film row every review belongs to.

    A film's vector is the plain mean of its review vectors. The average film
    is subtracted, so what is left is how a film differs from the others, and
    the chosen dataset columns are regressed out. Review vectors get the same
    centring so a review can be compared with a film or a group.
    """
    films, film_vectors, film_of_review = aggregate_films(reviews)
    centre = film_vectors.mean(axis=0)
    points = unit_rows(residualize(unit_rows(film_vectors - centre), confound_matrix(films, deconfound)))
    review_vectors = np.stack(reviews["embedding"].to_numpy()).astype(np.float32)
    return films, points, unit_rows(review_vectors - centre), film_of_review
