#!/usr/bin/env python3
"""Do longer reviews say more about a film than short ones?

Measures split-half agreement on reviews of at least N words, for several N.
Every row uses the same number of films and the same number of reviews per
film, so the only thing that changes is how long the reviews are. This is the
evidence behind `masking.min_words`.

To see the rows below the current cut-off, set `masking.min_words` low, run
the mask and embed steps, then run this. Writes outputs/review_length_<lang>.md.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from letterboxd_vibes.config import LANGS, OUTPUTS, load_config
from letterboxd_vibes.masking import word_count
from letterboxd_vibes.reliability import split_half_agreement
from letterboxd_vibes.vectors import load_reviews

PER_FILM = 4
FILMS = 600


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", choices=LANGS, default="en")
    parser.add_argument("--words", type=int, nargs="*", default=[4, 8, 12, 15, 20, 30, 50])
    args = parser.parse_args()
    cfg = load_config(args.lang)

    reviews = load_reviews(args.lang, cfg["embedding"]["model"])
    vectors = np.stack(reviews["embedding"].to_numpy())
    words = reviews["masked_text"].map(word_count).to_numpy()
    rng = np.random.default_rng(0)

    lines = [
        "# Review length and agreement",
        "",
        f"Split-half agreement on {FILMS} random films with exactly {PER_FILM} reviews each (0.5 = chance).",
        "",
        "| Reviews of at least | Films with enough such reviews | Agreement | Own half in top 10 |",
        "|---|---|---|---|",
    ]
    for minimum in args.words:
        long_enough = reviews[words >= minimum]
        sizes = long_enough.groupby("film_url").size()
        eligible = sizes[sizes >= PER_FILM].index
        if len(eligible) < FILMS:
            continue
        chosen = rng.choice(eligible, FILMS, replace=False)
        sample = long_enough[long_enough["film_url"].isin(chosen)].groupby("film_url").head(PER_FILM)
        agreement, top10 = split_half_agreement(
            pd.factorize(sample["film_url"])[0], vectors[sample.index], repeats=5
        )
        lines.append(f"| {minimum} words | {len(eligible):,} | {agreement:.3f} | {top10:.1%} |")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"review_length_{args.lang}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
