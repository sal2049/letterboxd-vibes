#!/usr/bin/env python3
"""Turn review vectors into the map: film vectors, groups, layout, evidence, names.

Writes web/public/data/movies_<lang>.json for the UI, outputs/audit_<lang>.json
with the measurements, and outputs/groups_<lang>.md for reading the groups by
hand.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from letterboxd_vibes.audit import metadata_association, separation
from letterboxd_vibes.clustering import (
    bootstrap_stability,
    central_members,
    cluster,
    layout_2d,
    neighbours,
)
from letterboxd_vibes.colours import position_hues, similarity_colours
from letterboxd_vibes.config import LANGS, OUTPUTS, load_config
from letterboxd_vibes.describe import clip_quote, distinctive_words
from letterboxd_vibes.export import apply_posters, load_tmdb_cache, write_group_sheet, write_web_json
from letterboxd_vibes.films import genre_matrix
from letterboxd_vibes.naming import name_groups
from letterboxd_vibes.reliability import split_half_agreement
from letterboxd_vibes.vectors import film_points, load_reviews

QUOTES_PER_FILM = 2
QUOTES_PER_GROUP = 6
MIN_QUOTE_CHARS = 80  # shorter reviews are usually one-liners that show little
REVIEWS_FOR_NAMING = 8
FILMS_FOR_NAMING = 12
FILMS_SHOWN = 9  # best-known members listed per group; plays no part in the grouping


def quote(reviews: pd.DataFrame, row: int) -> dict:
    review = reviews.iloc[row]
    return {"t": clip_quote(review["review_text"]), "u": review["username"], "l": int(review["likes"])}


def order_by_hue(labels: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Renumber groups by their direction from the middle of the map, so ids,
    colours and the legend all run in the same order."""
    ids = np.unique(labels[labels >= 0])
    centres = np.stack([np.median(coords[labels == g], axis=0) for g in ids])
    rank = {int(old): new for new, old in enumerate(ids[np.argsort(position_hues(centres))])}
    return np.array([rank.get(int(g), -1) for g in labels])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=LANGS, required=True)
    parser.add_argument("--no-names", action="store_true", help="skip the naming model, use words")
    args = parser.parse_args()
    cfg = load_config(args.lang)
    mcfg = cfg["map"]
    model = cfg["embedding"]["model"]
    naming_model = None if args.no_names else cfg["naming"]["model"]

    print("Loading review vectors")
    reviews = load_reviews(args.lang, model)

    print(f"Film vectors (regressed out: {', '.join(mcfg['deconfound'])})")
    films, points, review_points, film_of_review = film_points(reviews, mcfg["deconfound"])
    n = len(films)
    genres, genre_names = genre_matrix(films["genres"].tolist())

    print(f"Clustering ({mcfg['cluster_method']}, about {mcfg['n_clusters']} groups)")
    k = min(mcfg["n_clusters"], max(2, n // 25))
    labels = cluster(points, mcfg["cluster_method"], k)

    print("Laying out the map")
    coords = layout_2d(points, mcfg)
    labels = order_by_hue(labels, coords)
    group_ids = sorted(int(g) for g in set(labels) if g >= 0)
    centres = np.stack([np.median(coords[labels == g], axis=0) for g in group_ids])
    colours = similarity_colours(centres)
    nearest = neighbours(points, genres)

    # The evidence for each group. Everything below is read off the data.
    review_group = labels[film_of_review]
    words = distinctive_words({g: reviews.loc[review_group == g, "masked_text"].tolist() for g in group_ids})
    long_enough = (reviews["review_text"].str.len() >= MIN_QUOTE_CHARS).to_numpy()
    likes = films["likes_total"].to_numpy()

    # A film's quotes are its reviews closest to the film's own vector.
    own_film_match = np.einsum("ij,ij->i", review_points, points[film_of_review]) + long_enough
    quotes_by_film: list[list[dict]] = [[] for _ in range(n)]
    for row in np.argsort(-own_film_match):
        film = film_of_review[row]
        if len(quotes_by_film[film]) < QUOTES_PER_FILM:
            quotes_by_film[film].append(quote(reviews, int(row)))

    groups, evidence = [], []
    for g in group_ids:
        members = np.flatnonzero(labels == g)
        centre = points[members].mean(axis=0)
        centre /= np.linalg.norm(centre)
        genre_share = genres[members].mean(axis=0)
        exemplars = central_members(points, members)

        # The reviews that sound most like the group, at most one per film.
        candidates = np.flatnonzero((review_group == g) & long_enough)
        picked, used = [], set()
        for row in candidates[np.argsort(-(review_points[candidates] @ centre))]:
            film = int(film_of_review[row])
            if film not in used:
                used.add(film)
                picked.append(int(row))
            if len(picked) == REVIEWS_FOR_NAMING:
                break

        groups.append(
            {
                "id": g,
                "color": colours[g],
                "size": len(members),
                # mean similarity of the group's films to its centre: 1 = identical, 0 = unrelated
                "cohesion": round(float((points[members] @ centre).mean()), 3),
                "keywords": words[g],
                "genres": [
                    [genre_names[i], round(float(genre_share[i]), 3)]
                    for i in np.argsort(-genre_share)[:5]
                    if genre_share[i] > 0
                ],
                "cx": float(centres[g, 0]),
                "cy": float(centres[g, 1]),
                "exemplars": exemplars,
                "popular": [int(i) for i in members[np.argsort(-likes[members])[:FILMS_SHOWN]]],
                "quotes": [
                    {**quote(reviews, row), "f": int(film_of_review[row])}
                    for row in picked[:QUOTES_PER_GROUP]
                ],
            }
        )
        evidence.append(
            {
                "words": words[g],
                "films": [
                    f"{films['title'].iloc[i]} ({int(films['year'].iloc[i])})"
                    if pd.notna(films["year"].iloc[i])
                    else films["title"].iloc[i]
                    for i in exemplars[:FILMS_FOR_NAMING]
                ],
                # the model reads the same name-free text the embedding model read
                "reviews": [clip_quote(reviews["masked_text"].iloc[row], 220) for row in picked],
            }
        )

    print("Labelling groups" + (f" with {naming_model}" if naming_model else " from their words"))
    for group, name in zip(groups, name_groups(evidence, naming_model), strict=True):
        group.update(label=name["name"], named_by=name["by"])

    points_out = [
        {
            "id": film.film_id,
            "title": film.title,
            "year": int(film.year) if pd.notna(film.year) else None,
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": int(labels[i]),
            "poster": film.poster_url or "",
            "url": film.film_url,
            "tmdb": None,
            "genres": film.genres,
            "director": film.director,
            "n": int(film.n_reviews),
            "pop": int(film.likes_total),
            "quotes": quotes_by_film[i],
            "nn": nearest[i],
        }
        for i, film in enumerate(films.itertuples())
    ]

    enough = (films["n_reviews"] >= 2).to_numpy()[film_of_review]
    agreement, _ = split_half_agreement(
        pd.factorize(film_of_review[enough])[0],
        np.stack(reviews["embedding"].to_numpy())[enough],
    )
    audit = {
        "lang": args.lang,
        "films": n,
        "reviews": len(reviews),
        "embedding_model": model,
        "naming_model": naming_model,
        "split_half_agreement": round(agreement, 3),
        "cluster_method": mcfg["cluster_method"],
        "groups": len(group_ids),
        "unassigned_share": round(float((labels < 0).mean()), 3),
        **separation(points, labels),
        "stability": round(
            bootstrap_stability(lambda x, run: cluster(x, mcfg["cluster_method"], k, seed=run), points), 2
        ),
        "regressed_out": mcfg["deconfound"],
        "cramers_v": metadata_association(films, labels),
        "group_sizes": {f"{g['id']}. {g['label']}": g["size"] for g in groups},
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"audit_{args.lang}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    payload = {
        "meta": {
            "lang": args.lang,
            "embedding_model": model,
            "cluster_method": mcfg["cluster_method"],
            "naming_model": naming_model,
            "n_movies": n,
            "n_reviews": len(reviews),
            "n_clusters": len(group_ids),
            "poster_source": "none",
        },
        "clusters": groups,
        "points": points_out,
    }
    apply_posters(payload, load_tmdb_cache())
    print(f"Wrote {write_web_json(payload, args.lang)}")
    print(f"Wrote {write_group_sheet(payload, audit, args.lang)}")
    print(json.dumps({k: audit[k] for k in ("groups", "silhouette", "stability", "cramers_v")}))


if __name__ == "__main__":
    main()
