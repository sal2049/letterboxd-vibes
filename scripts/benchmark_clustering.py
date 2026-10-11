#!/usr/bin/env python3
"""Compare clustering algorithms on the film vectors.

Every algorithm in letterboxd_vibes.clustering.ALGORITHMS is run for each
requested number of groups and measured the same way:

  silhouette, Davies-Bouldin, Calinski-Harabasz   how well-separated the groups are
  stability      agreement between runs on different 80% samples of the films
  unassigned     share of films the method leaves out
  largest        share of films in the biggest group
  genre V        how strongly groups track genre (lower is better here)

Selection rule, fixed before looking at any groups: discard methods that leave
more than 10% of films unassigned or put more than 20% in one group, then take
the most stable; silhouette breaks ties. Writes outputs/benchmark_clustering_<lang>.json and .md.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from letterboxd_vibes.audit import metadata_association, separation
from letterboxd_vibes.clustering import ALGORITHMS, bootstrap_stability, cluster
from letterboxd_vibes.config import LANGS, OUTPUTS, load_config
from letterboxd_vibes.vectors import film_points, load_reviews

MAX_UNASSIGNED = 0.10
MAX_LARGEST = 0.20

POINTS: np.ndarray
FILMS: pd.DataFrame


def measure(task: tuple[int, str]) -> dict:
    k, method = task
    started = time.time()
    labels = cluster(POINTS, method, k)
    kept = labels >= 0
    sizes = np.bincount(labels[kept])
    row = {
        "method": method,
        "groups_asked": k,
        "groups_found": len(sizes),
        **separation(POINTS, labels),
        "stability": round(
            bootstrap_stability(lambda x, run: cluster(x, method, k, seed=run), POINTS, reps=3), 2
        ),
        "unassigned": round(float(1 - kept.mean()), 2),
        "largest": round(float(sizes.max() / len(labels)), 2),
        "genre_v": metadata_association(FILMS, labels)["primary_genre"],
        "seconds": round(time.time() - started),
    }
    row["eligible"] = row["unassigned"] <= MAX_UNASSIGNED and row["largest"] <= MAX_LARGEST
    return row


def load(lang: str) -> None:
    """Film vectors, loaded once per worker process."""
    global POINTS, FILMS
    cfg = load_config(lang)
    reviews = load_reviews(lang, cfg["embedding"]["model"])
    FILMS, POINTS, _, _ = film_points(reviews, cfg["map"]["deconfound"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", choices=LANGS, default="en")
    parser.add_argument("--groups", type=int, nargs="*", default=[20, 40, 60])
    parser.add_argument("--methods", nargs="*", default=list(ALGORITHMS))
    parser.add_argument("--jobs", type=int, default=4, help="runs measured at the same time")
    args = parser.parse_args()
    model = load_config(args.lang)["embedding"]["model"]

    load(args.lang)
    films = FILMS
    tasks = [(k, method) for k in args.groups for method in args.methods]
    rows = []
    with ProcessPoolExecutor(args.jobs, initializer=load, initargs=(args.lang,)) as pool:
        for row in pool.map(measure, tasks):
            rows.append(row)
            print(json.dumps(row), flush=True)

    best = {}
    for k in args.groups:
        eligible = [r for r in rows if r["groups_asked"] == k and r["eligible"]]
        if eligible:
            best[k] = max(eligible, key=lambda r: (r["stability"], r["silhouette"] or -1))["method"]

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    result = {"embedding_model": model, "films": len(films), "selected_by_groups": best, "runs": rows}
    (OUTPUTS / f"benchmark_clustering_{args.lang}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    header = [
        "Groups",
        "Method",
        "Silhouette",
        "Davies-Bouldin",
        "Calinski-Harabasz",
        "Stability",
        "Unassigned",
        "Largest",
        "Genre V",
        "",
    ]
    table = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        note = (
            "selected"
            if best.get(r["groups_asked"]) == r["method"]
            else ("" if r["eligible"] else "not eligible")
        )
        cells = [
            r["groups_asked"] if r["groups_found"] == r["groups_asked"] else f"{r['groups_asked']} (found {r['groups_found']})",
            r["method"], r["silhouette"], r["davies_bouldin"], r["calinski_harabasz"], r["stability"],
            f"{r['unassigned']:.0%}", f"{r['largest']:.0%}", r["genre_v"], note,
        ]  # fmt: skip
        table.append("| " + " | ".join(str(c) for c in cells) + " |")
    (OUTPUTS / f"benchmark_clustering_{args.lang}.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
