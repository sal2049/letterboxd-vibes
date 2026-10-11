"""From review vectors to one vector per film."""

from __future__ import annotations

import numpy as np
import pandas as pd

from letterboxd_vibes.text import film_slug


def aggregate_films(reviews: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Collapse reviews into films.

    Returns the film table, a unit vector per film (the plain mean of its
    review vectors), and for every review the row of the film it belongs to.
    """
    film_of_review, urls = pd.factorize(reviews["film_url"])
    vectors = np.stack(reviews["embedding"].to_numpy()).astype(np.float32)

    film_vectors = np.zeros((len(urls), vectors.shape[1]), dtype=np.float32)
    np.add.at(film_vectors, film_of_review, vectors)
    film_vectors /= np.linalg.norm(film_vectors, axis=1, keepdims=True) + 1e-9

    first = reviews.groupby("film_url", sort=False).first()
    grouped = reviews.groupby("film_url", sort=False)
    films = pd.DataFrame(
        {
            "film_url": urls,
            "film_id": [film_slug(u) for u in urls],
            "title": first["title"].to_numpy(),
            "year": first["year"].to_numpy(),
            "genres": [list(g) for g in first["genres"]],
            "director": [", ".join(list(d)[:2]) for d in first["directors"]],
            "rating": first["rating"].to_numpy(),
            "poster_url": first["poster_url"].to_numpy(),
            "n_reviews": grouped.size().to_numpy(),
            "likes_total": grouped["likes"].sum().to_numpy(),
        }
    )
    return films, film_vectors, film_of_review


def genre_matrix(genre_lists: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """0/1 film-by-genre matrix and its column names."""
    names = sorted({g for genres in genre_lists for g in genres})
    column = {g: i for i, g in enumerate(names)}
    matrix = np.zeros((len(genre_lists), len(names)), dtype=np.float32)
    for row, genres in enumerate(genre_lists):
        for g in genres:
            matrix[row, column[g]] = 1.0
    return matrix, names
