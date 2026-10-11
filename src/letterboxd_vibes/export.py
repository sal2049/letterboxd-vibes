"""Write the two things people read: the web JSON and the cluster evidence sheet."""

from __future__ import annotations

import json
from pathlib import Path

from letterboxd_vibes.config import DATA, OUTPUTS, WEB_DATA

TMDB_CACHE = DATA / "tmdb_cache.json"


def load_tmdb_cache() -> dict[str, dict]:
    if not TMDB_CACHE.exists():
        return {}
    return json.loads(TMDB_CACHE.read_text(encoding="utf-8"))


def apply_posters(payload: dict, cache: dict[str, dict]) -> None:
    """Prefer the TMDB poster for every film that was matched; note where posters come from."""
    sources = set()
    for point in payload["points"]:
        hit = cache.get(point["id"]) or {}
        if hit.get("id"):
            point["tmdb"] = {"id": hit["id"], "type": hit.get("type", "movie")}
        if hit.get("poster_path"):
            point["poster"] = hit["poster_path"]
        if point["poster"]:
            sources.add("tmdb" if point["poster"].startswith("/") else "letterboxd")
    payload["meta"]["poster_source"] = "mixed" if len(sources) > 1 else next(iter(sources), "none")


def write_web_json(payload: dict, lang: str) -> Path:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    path = WEB_DATA / f"movies_{lang}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def write_group_sheet(payload: dict, audit: dict, lang: str) -> Path:
    """One Markdown page per track for reading the groups by hand."""
    points = payload["points"]
    named = (
        f"Theme labels were written by {audit['naming_model']} from the evidence listed"
        if audit["naming_model"]
        else "Labels are each group's three most distinctive words"
    )
    lines = [
        f"# Groups ({lang})",
        "",
        f"{audit['films']} films in {audit['groups']} groups, found by {audit['cluster_method']} on film vectors",
        f"from {audit['embedding_model']}, which read the reviews with names removed. {named}.",
        "A label is a reading of the evidence under it; check it against the films and reviews.",
        "",
        (
            f"Split-half agreement of the review vectors: {audit['split_half_agreement']} (0.5 = chance). "
            f"Separation: silhouette {audit['silhouette']}, Davies-Bouldin {audit['davies_bouldin']}, "
            f"Calinski-Harabasz {audit['calinski_harabasz']}. Stability across resamples: {audit['stability']}."
        ),
        "",
        "Association between group and metadata (Cramér's V, 0 = unrelated, 1 = identical): "
        + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in audit["cramers_v"].items())
        + ".",
        "",
    ]
    for group in payload["clusters"]:
        films = "; ".join(f"{points[i]['title']} ({points[i]['year']})" for i in group["exemplars"])
        lines += [
            f"## {group['id']}. {group['label']}",
            "",
            f"{group['size']} films, cohesion {group['cohesion']}.",
            "",
            "**Words its reviews over-use:** " + ", ".join(group["keywords"]),
            "",
            "**Genre mix:** " + ", ".join(f"{g} {s:.0%}" for g, s in group["genres"]),
            "",
            f"**Films nearest the centre:** {films}",
            "",
            "**Reviews closest to the group's centre:**",
            "",
            *[f'- "{q["t"]}" (on *{points[q["f"]]["title"]}*)' for q in group["quotes"]],
            "",
        ]
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / f"groups_{lang}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
