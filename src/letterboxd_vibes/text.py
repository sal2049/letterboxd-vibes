from __future__ import annotations

import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WS_RE = re.compile(r"\s+")

# About 5% of the reviews in the dump are Letterboxd's spoiler gate, captured
# in place of the hidden review. It says nothing about the film.
SPOILER_GATE_RE = re.compile(
    r"this review may contain spoilers\.?\s*i can handle the truth\.?", re.IGNORECASE
)


def clean_review(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = SPOILER_GATE_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def is_valid_review(text: str, min_chars: int = 40) -> bool:
    """Long enough, and mostly letters rather than emoji, digits or punctuation."""
    if len(text) < min_chars:
        return False
    letters = sum(1 for c in text if c.isalpha())
    return letters >= 0.5 * len(text)


def film_slug(film_url: str) -> str:
    return film_url.rstrip("/").split("/")[-1]


def parse_likes(likes: str | float | None) -> float:
    """'11,662' -> 11662.0"""
    if likes is None:
        return 0.0
    try:
        return float(str(likes).replace(",", "").strip())
    except ValueError:
        return 0.0


def parse_rating(rating: str | None) -> float | None:
    """'4.62 out of 5' -> 4.62"""
    try:
        return float(str(rating).split()[0])
    except (ValueError, IndexError):
        return None


def parse_year(year: str | int | None) -> int | None:
    try:
        return int(year) if year else None
    except (TypeError, ValueError):
        return None
