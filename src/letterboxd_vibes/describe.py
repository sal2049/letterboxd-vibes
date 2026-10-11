"""The evidence shown for a group: the words its reviews over-use, and tidy quotes."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def distinctive_words(
    texts_by_group: dict[int, list[str]],
    top_n: int = 8,
    exclude: set[str] | None = None,
    max_share: float = 0.15,
) -> dict[int, list[str]]:
    """Words a group's reviews use more than everyone else's.

    A word is counted once per review, so one review repeating it fifty times
    changes nothing. Score: p_in * log(p_in / p_out), the group's share of
    reviews using the word against the share elsewhere. Common English words
    (scikit-learn's list) and any word in more than `max_share` of all reviews
    are ignored; `exclude` removes extra tokens such as director names, which say
    who made a film, not how it felt.
    """
    ids = sorted(texts_by_group)
    texts = [t for g in ids for t in texts_by_group[g]]
    owner = np.array([g for g in ids for _ in texts_by_group[g]])
    if len(ids) < 2 or not texts:
        return {g: [] for g in ids}

    vectorizer = CountVectorizer(
        binary=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5 if len(texts) > 2000 else 2,
        max_df=max_share,
        max_features=30000,
        token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
    )
    try:
        counts = vectorizer.fit_transform(texts)
    except ValueError:  # nothing left after stop words
        return {g: [] for g in ids}
    terms = vectorizer.get_feature_names_out()
    total = np.asarray(counts.sum(axis=0)).ravel()
    blocked = exclude or set()

    words: dict[int, list[str]] = {}
    for g in ids:
        rows = owner == g
        inside = np.asarray(counts[rows].sum(axis=0)).ravel()
        p_in = (inside + 0.5) / (rows.sum() + 1)
        p_out = (total - inside + 0.5) / ((~rows).sum() + 1)
        score = p_in * np.log(p_in / p_out)
        score[inside < max(3, 0.004 * rows.sum())] = -np.inf

        picked: list[str] = []
        for idx in np.argsort(-score):
            if score[idx] <= 0:
                break
            term = terms[idx]
            if any(w in blocked for w in term.split()):
                continue
            if any(term in p or p in term for p in picked):  # "heart" after "my heart"
                continue
            picked.append(term)
            if len(picked) == top_n:
                break
        words[g] = picked
    return words


def clip_quote(text: str, limit: int = 260) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(".,;:!?- ") + "…"
