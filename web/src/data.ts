import { hexToRgb, UNASSIGNED } from "./colors";
import type { FilmPoint, Lang, MoodDataset } from "./types";

export async function loadDataset(lang: Lang): Promise<MoodDataset> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/movies_${lang}.json`);
  if (!res.ok) {
    throw new Error(`No data for this track yet — run the pipeline for "${lang}".`);
  }
  const data = (await res.json()) as MoodDataset;
  if (!data.clusters.every((c) => typeof c.cohesion === "number")) {
    throw new Error(
      `movies_${lang}.json was written by an older pipeline. Re-run scripts/04_build_map.py --lang ${lang}.`,
    );
  }
  const colourOf = new Map(data.clusters.map((c) => [c.id, c.color]));
  for (const p of data.points) {
    p.hex = colourOf.get(p.cluster) ?? UNASSIGNED;
    p.rgb = hexToRgb(p.hex);
  }
  return data;
}

type PosterSize = "w92" | "w154" | "w342" | "w500";

/** TMDB paths are stored bare ("/abc.jpg") so the size can be chosen per use. */
export function posterUrl(poster: string, size: PosterSize): string {
  if (!poster) return "";
  if (poster.startsWith("/")) return `https://image.tmdb.org/t/p/${size}${poster}`;
  return poster;
}

export function tmdbUrl(film: FilmPoint): string | null {
  return film.tmdb ? `https://www.themoviedb.org/${film.tmdb.type}/${film.tmdb.id}` : null;
}

const METHOD_NAMES: Record<string, string> = {
  kmeans: "k-means",
  "umap-kmeans": "UMAP + k-means",
  "umap-hdbscan": "UMAP + HDBSCAN",
  spectral: "spectral clustering",
  ward: "Ward linkage",
  "average-linkage": "average linkage",
  "gaussian-mixture": "a Gaussian mixture",
  birch: "BIRCH",
  leiden: "Leiden community detection",
};

/** Readable name for a clustering method id from the pipeline. */
export function methodName(id: string): string {
  return METHOD_NAMES[id] ?? id;
}

export function yearExtent(points: FilmPoint[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of points) {
    if (p.year == null) continue;
    if (p.year < lo) lo = p.year;
    if (p.year > hi) hi = p.year;
  }
  return Number.isFinite(lo) ? [lo, hi] : [1900, 2030];
}
