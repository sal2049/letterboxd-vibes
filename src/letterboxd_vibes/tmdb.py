"""Match a Letterboxd film to its TMDB entry by title and year."""

from __future__ import annotations

import os
import re
import time
import unicodedata

import httpx

API = "https://api.themoviedb.org/3"


def make_client() -> httpx.Client:
    """Accepts either TMDB credential: the "API Read Access Token" (sent as a
    Bearer header) or the short v3 "API Key" (sent as the api_key parameter)."""
    token = os.environ.get("TMDB_ACCESS_TOKEN") or ""
    key = os.environ.get("TMDB_API_KEY") or ""
    if key.startswith("eyJ"):  # an access token pasted into the key slot
        token, key = key, ""
    if token:
        return httpx.Client(base_url=API, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if key:
        return httpx.Client(base_url=API, params={"api_key": key}, timeout=20)
    raise SystemExit(
        "No TMDB credential found. Add TMDB_ACCESS_TOKEN (or TMDB_API_KEY) to .env; "
        "create one at https://www.themoviedb.org/settings/api"
    )


def _norm(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_title.lower()).strip()


def _get(client: httpx.Client, path: str, params: dict) -> dict:
    for attempt in range(5):
        response = client.get(path, params=params)
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", 1 + attempt)))
            continue
        if response.status_code == 401:
            raise SystemExit("TMDB rejected the credential (401). Check the value in .env.")
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"TMDB kept rate-limiting {path}")


def pick(results: list[dict], title: str, year: int | None, kind: str) -> dict | None:
    """The best search result, or None if nothing is a safe match.

    Accepted: the same title within a year of the release date, or a title
    containing (or contained in) ours in exactly the same year. Among those,
    prefer an exact title, then the closer year, then a result with a poster,
    then the more popular one.
    """
    name_keys = ("title", "original_title") if kind == "movie" else ("name", "original_name")
    date_key = "release_date" if kind == "movie" else "first_air_date"
    want = _norm(title)
    best: tuple[tuple, dict] | None = None
    for result in results:
        names = {_norm(result.get(k) or "") for k in name_keys} - {""}
        exact = want in names
        loose = bool(want) and any(want in n or n in want for n in names)
        released = (result.get(date_key) or "")[:4]
        gap = abs(int(released) - year) if (year and released.isdigit()) else 9
        if not ((exact and (gap <= 1 or not year)) or (loose and gap == 0)):
            continue
        score = (exact, -gap, bool(result.get("poster_path")), result.get("popularity") or 0)
        if best is None or score > best[0]:
            best = (score, result)
    return best[1] if best else None


def lookup(client: httpx.Client, title: str, year: int | None) -> dict:
    """{"id", "type", "poster_path"} for the film, or {"id": None} if not found."""
    base = {"query": title, "include_adult": "false"}
    searches = [
        ("movie", {**base, **({"year": year} if year else {})}),
        # Letterboxd lists miniseries and specials that TMDB files under TV.
        ("tv", {**base, **({"first_air_date_year": year} if year else {})}),
        ("movie", base),
    ]
    for kind, params in searches:
        hit = pick(_get(client, f"/search/{kind}", params).get("results") or [], title, year, kind)
        if hit:
            return {"id": hit["id"], "type": kind, "poster_path": hit.get("poster_path")}
    return {"id": None}
