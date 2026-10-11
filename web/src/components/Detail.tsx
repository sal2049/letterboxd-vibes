import { methodName, tmdbUrl } from "../data";
import type { Cluster, FilmPoint, MoodDataset } from "../types";
import { Poster } from "./Poster";

function sharesGenre(a: FilmPoint, b: FilmPoint): boolean {
  return a.genres.some((g) => b.genres.includes(g));
}

interface FilmProps {
  data: MoodDataset;
  film: FilmPoint;
  onSelect: (index: number) => void;
  onClose: () => void;
  onPin: (id: number) => void;
}

export function FilmDetail({ data, film, onSelect, onClose, onPin }: FilmProps) {
  const cluster = data.clusters.find((c) => c.id === film.cluster);
  const tmdb = tmdbUrl(film);

  return (
    <article className="detail">
      <button type="button" className="detail-close" onClick={onClose} aria-label="Close film">
        ×
      </button>

      <div className="film-head">
        <Poster film={film} size="w342" className="film-poster" eager />
        <div>
          <h2>{film.title}</h2>
          <p className="muted">
            {film.year ?? "—"}
            {film.director ? ` · ${film.director}` : ""}
          </p>
          {cluster ? (
            <button
              type="button"
              className="chip chip-mood"
              style={{ ["--c" as string]: film.hex }}
              onClick={() => onPin(cluster.id)}
            >
              <i />
              {cluster.label}
            </button>
          ) : (
            <span className="chip chip-mood" style={{ ["--c" as string]: film.hex }}>
              <i />
              Unassigned
            </span>
          )}
        </div>
      </div>

      {film.quotes.length > 0 && (
        <section>
          <h3>In viewers' words</h3>
          {film.quotes.map((q, n) => (
            <blockquote key={n} dir="auto">
              <p>{q.t}</p>
              <footer>{q.u || "anonymous"}</footer>
            </blockquote>
          ))}
        </section>
      )}

      {film.nn.length > 0 && (
        <section>
          <h3>Talked about the same way</h3>
          <ul className="poster-grid">
            {film.nn.map((i) => {
              const other = data.points[i];
              if (!other) return null;
              return (
                <li key={other.id}>
                  <button
                    type="button"
                    onClick={() => onSelect(i)}
                    title={`${other.title} (${other.year ?? "—"})`}
                  >
                    <Poster film={other} size="w154" />
                    {!sharesGenre(film, other) && <span className="tag">different genre</span>}
                  </button>
                </li>
              );
            })}
          </ul>
          <p className="note">
            Films whose reviews read most like this one's once names are removed. Not a ranking, and
            not a recommendation.
          </p>
        </section>
      )}

      <section className="film-meta">
        <p className="muted">
          {film.genres.length > 0 && <>{film.genres.join(" · ")} · </>}
          read from {film.n} review{film.n === 1 ? "" : "s"}
        </p>
        <p className="links">
          <a href={film.url} target="_blank" rel="noreferrer">
            Letterboxd ↗
          </a>
          {tmdb && (
            <a href={tmdb} target="_blank" rel="noreferrer">
              TMDB ↗
            </a>
          )}
        </p>
      </section>
    </article>
  );
}

function FilmRow({
  title,
  data,
  films,
  onSelect,
}: {
  title: string;
  data: MoodDataset;
  films: number[];
  onSelect: (index: number) => void;
}) {
  if (films.length === 0) return null;
  return (
    <section>
      <h3>{title}</h3>
      <ul className="poster-grid">
        {films.slice(0, 9).map((i) => {
          const f = data.points[i];
          if (!f) return null;
          return (
            <li key={f.id}>
              <button type="button" onClick={() => onSelect(i)} title={`${f.title} (${f.year ?? "—"})`}>
                <Poster film={f} size="w154" />
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

interface ClusterProps {
  data: MoodDataset;
  cluster: Cluster;
  onSelect: (index: number) => void;
  onClose: () => void;
}

export function ClusterDetail({ data, cluster, onSelect, onClose }: ClusterProps) {
  const byModel = cluster.named_by === "model";
  const namer = data.meta.naming_model?.split("/").pop();

  return (
    <article className="detail">
      <button type="button" className="detail-close" onClick={onClose} aria-label="Show all groups">
        ×
      </button>
      <p className="eyebrow">
        <i style={{ background: cluster.color }} /> Group {cluster.id + 1} ·{" "}
        {cluster.size.toLocaleString()} films
      </p>
      <h2 className="cluster-title">{cluster.label}</h2>

      <p className="note name-note">
        {byModel
          ? `This label was written by a language model (${namer}) that was shown the words, films and reviews below. It is a reading of that evidence, so check it against them.`
          : "Labelled by the words its reviews over-use most."}
      </p>

      {cluster.quotes.length > 0 && (
        <section>
          <h3>Reviews closest to its centre</h3>
          {cluster.quotes.slice(0, 4).map((q, n) => (
            <blockquote key={n} dir="auto" style={{ borderInlineStartColor: cluster.color }}>
              <p>{q.t}</p>
              <footer>
                on{" "}
                <button type="button" className="link" onClick={() => onSelect(q.f)}>
                  {data.points[q.f]?.title}
                </button>
              </footer>
            </blockquote>
          ))}
        </section>
      )}

      {cluster.keywords.length > 0 && (
        <section>
          <h3>Words its reviews over-use</h3>
          <p className="chips">
            {cluster.keywords.map((k) => (
              <span key={k} className="chip" dir="auto">
                {k}
              </span>
            ))}
          </p>
        </section>
      )}

      <FilmRow title="Films nearest its centre" data={data} films={cluster.exemplars} onSelect={onSelect} />
      <FilmRow title="Best known inside" data={data} films={cluster.popular} onSelect={onSelect} />

      {cluster.genres.length > 0 && (
        <section>
          <h3>Genres inside</h3>
          <ul className="genre-mix">
            {cluster.genres.map(([g, s]) => (
              <li key={g}>
                <span>{g}</span>
                <span className="moodbar">
                  <b style={{ width: `${s * 100}%` }} />
                </span>
                <span className="muted">{Math.round(s * 100)}%</span>
              </li>
            ))}
          </ul>
          <p className="note">
            Genre is regressed out before the films are grouped. This is only what ended up inside.
          </p>
        </section>
      )}
    </article>
  );
}

export function Overview({ data, onSurprise }: { data: MoodDataset; onSurprise: () => void }) {
  return (
    <article className="detail overview">
      <p className="eyebrow">How it was made</p>
      <h2 className="cluster-title">Which films do people talk about the same way?</h2>
      <ol>
        <li>
          <span>
            Names are stripped from every review: actors, titles, countries, franchises. What is
            left is how people talk about the film.
          </span>
        </li>
        <li>
          <span>
            An embedding model (<code>{data.meta.embedding_model.split("/").pop()}</code>) turns
            each stripped review into numbers. A film is the average of its reviews, with year,
            popularity, rating and genre taken out.
          </span>
        </li>
        <li>
          <span>
            A clustering model ({methodName(data.meta.cluster_method)}) finds{" "}
            {data.meta.n_clusters} groups of films that sit close together. Nearby groups get
            similar colours.
          </span>
        </li>
        <li>
          <span>
            {data.meta.naming_model
              ? "A language model reads each group's words, films and reviews and writes a theme label. Open a group to check the label against that evidence."
              : "Each group is labelled by the words its reviews over-use. Open one to see the films and reviews behind it."}
          </span>
        </li>
      </ol>
      <button type="button" className="button" onClick={onSurprise}>
        Take me somewhere
      </button>
    </article>
  );
}
