export type Lang = "en";

export type RGB = [number, number, number];

export interface Quote {
  /** review text */
  t: string;
  /** reviewer */
  u: string;
  /** likes */
  l: number;
}

export interface FilmPoint {
  id: string;
  title: string;
  year: number | null;
  x: number;
  y: number;
  /** group id; -1 = left unassigned by the clustering method */
  cluster: number;
  /** TMDB poster path ("/abc.jpg"), an absolute URL, or "" */
  poster: string;
  /** Letterboxd film page */
  url: string;
  tmdb: { id: number; type: "movie" | "tv" } | null;
  genres: string[];
  director: string;
  /** reviews that went into this film's vector */
  n: number;
  /** total likes on those reviews; only decides which posters win space on the map */
  pop: number;
  quotes: Quote[];
  /** indexes (into points) of the films whose reviews read most like this one's */
  nn: number[];
  /** the film's group colour, filled in when the data is loaded */
  hex: string;
  rgb: RGB;
}

export interface Cluster {
  id: number;
  /** theme label: a short phrase on the mood and what the stories are about */
  label: string;
  /** "model": written by the naming model from the evidence below; "words": the group's top words */
  named_by: "model" | "words";
  /** set by the pipeline from the group's position, so nearby groups look alike */
  color: string;
  size: number;
  /** mean similarity of the group's films to its centre, 0..1 */
  cohesion: number;
  /** words this group's reviews use more than the rest */
  keywords: string[];
  /** [genre, share of the group's films] */
  genres: [string, number][];
  cx: number;
  cy: number;
  /** indexes (into points) of the films nearest the group's centre */
  exemplars: number[];
  /** indexes (into points) of the group's best-known films; shown only, never used to group */
  popular: number[];
  /** reviews closest to the group's centre; f indexes into points */
  quotes: (Quote & { f: number })[];
}

export interface MoodDataset {
  meta: {
    lang: Lang;
    embedding_model: string;
    cluster_method: string;
    /** null when groups are named by their words */
    naming_model: string | null;
    n_movies: number;
    n_reviews: number;
    n_clusters: number;
    poster_source: "tmdb" | "letterboxd" | "mixed" | "none";
  };
  clusters: Cluster[];
  points: FilmPoint[];
}

export type Focus =
  | { kind: "fit"; nonce: number }
  | { kind: "cluster"; id: number; nonce: number }
  | { kind: "film"; index: number; nonce: number };
