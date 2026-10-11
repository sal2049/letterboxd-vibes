"""Take the names out of reviews before they are embedded.

An embedding model given a raw review mostly notices who and what it mentions:
the actor, the franchise, the country. Films then group by those. Removing
that vocabulary leaves how people talk about the film, which is the part this
project is after.

No word list is written by hand. Three rules decide what goes, each read off
the data:

rare         the word appears in reviews of too few films to be a shared way
             of talking (character names, typos, in-jokes)
capitalised  across all reviews the word is written with a capital more often
             than not when it is not starting a sentence (people, places,
             nationalities, brands, franchises)
credits      the word is in the title or the billed cast and directors of the
             film being reviewed; removed from that film's reviews only, and
             only if it is not an everyday word
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
SENTENCE_BREAK = re.compile(r"(?<=[.!?\n])\s+")
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})
# English always capitalises the pronoun, so its capital says nothing.
PRONOUN_I = re.compile(r"i('\w+)?$")

MIN_NAME_LETTERS = 3  # "Li", "Jo" and initials would take common short words with them


@dataclass(frozen=True)
class Vocabulary:
    """Which words survive, and why the others did not."""

    keep: frozenset[str]
    rare: frozenset[str]
    capitalised: frozenset[str]

    def summary(self, examples: dict[str, list[str]]) -> dict:
        return {
            "kept": len(self.keep),
            "removed_rare": len(self.rare),
            "removed_capitalised": len(self.capitalised),
            "examples": examples,
        }


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.translate(APOSTROPHES))


def word_counts(film_ids: Iterable, texts: Iterable[str]) -> dict[str, tuple[int, int, int]]:
    """word -> (films whose reviews use it, lower-case uses, capitalised uses).

    A capitalised use is counted only away from the start of a sentence, and
    words in all capitals are ignored, so shouting and ordinary sentence case
    do not make a word look like a name.
    """
    films: dict[str, set] = defaultdict(set)
    lower: Counter[str] = Counter()
    capital: Counter[str] = Counter()
    for film, text in zip(film_ids, texts, strict=True):
        for sentence in SENTENCE_BREAK.split(text.translate(APOSTROPHES)):
            for position, token in enumerate(TOKEN.findall(sentence)):
                word = token.lower()
                films[word].add(film)
                if token.isupper() and len(token) > 1:
                    continue
                if token.islower():
                    lower[word] += 1
                elif token[0].isupper() and position > 0:
                    capital[word] += 1
    return {w: (len(f), lower[w], capital[w]) for w, f in films.items()}


def build_vocabulary(
    counts: dict[str, tuple[int, int, int]], min_films: int, capital_share: float
) -> Vocabulary:
    rare, capitalised, keep = set(), set(), set()
    for word, (n_films, lower, capital) in counts.items():
        if n_films < min_films:
            rare.add(word)
        elif capital / max(1, capital + lower) > capital_share and not PRONOUN_I.match(word):
            capitalised.add(word)
        else:
            keep.add(word)
    return Vocabulary(frozenset(keep), frozenset(rare), frozenset(capitalised))


def everyday_words(counts: dict[str, tuple[int, int, int]], n_films: int, share: float) -> frozenset[str]:
    """Words used for so many films that a title containing one says nothing ("the", "love", "man")."""
    return frozenset(w for w, (films, _, _) in counts.items() if films > share * n_films)


def credit_words(title: str, people: Iterable[str], everyday: frozenset[str] = frozenset()) -> frozenset[str]:
    """Words of a film's own title and credited names, minus everyday words."""
    words = {t.lower() for t in tokens(title)}
    for name in people:
        words.update(t.lower() for t in tokens(name))
    return frozenset(w for w in words if len(w) >= MIN_NAME_LETTERS and w not in everyday)


def mask(text: str, vocabulary: Vocabulary, credits: frozenset[str] = frozenset()) -> str:
    """The review with every removed word deleted and the rest left as written."""
    text = text.translate(APOSTROPHES)

    def keep_or_drop(match: re.Match) -> str:
        word = match.group().lower()
        return match.group() if word in vocabulary.keep and word not in credits else ""

    out = TOKEN.sub(keep_or_drop, text)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)  # punctuation left hanging by a deleted word
    return re.sub(r"\s+", " ", out).strip(" ,;:-")


def word_count(text: str) -> int:
    return len(TOKEN.findall(text))
