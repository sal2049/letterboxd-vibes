import { UNASSIGNED } from "../colors";
import { methodName } from "../data";
import type { MoodDataset } from "../types";
import { YearRange } from "./YearRange";

interface Props {
  data: MoodDataset | null;
  pinned: number | null;
  onPin: (id: number | null) => void;
  onHover: (id: number | null) => void;
  yearExtent: [number, number];
  yearRange: [number, number];
  onYearRange: (range: [number, number]) => void;
}

export function Sidebar({
  data,
  pinned,
  onPin,
  onHover,
  yearExtent,
  yearRange,
  onYearRange,
}: Props) {
  const groups = data?.clusters ?? [];
  const unassigned = data ? data.points.filter((p) => p.cluster < 0).length : 0;

  return (
    <aside className="sidebar">
      <header className="brand">
        <h1>
          Vibe <em>Map</em>
        </h1>
        <p>Films placed by how people talk about them in reviews.</p>
      </header>

      {data && (
        <dl className="stats">
          <div>
            <dt>Films</dt>
            <dd>{data.meta.n_movies.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Reviews</dt>
            <dd>{data.meta.n_reviews.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Groups</dt>
            <dd>{data.meta.n_clusters}</dd>
          </div>
        </dl>
      )}

      <section className="legend" aria-label="Groups">
        <div className="section-head">
          <h2>Groups</h2>
          {pinned != null && (
            <button type="button" className="link" onClick={() => onPin(null)}>
              Show all
            </button>
          )}
        </div>
        <p className="legend-note">
          Listed in map order, so neighbours in the list are neighbours on the map and share a
          colour range. Labels are a model's reading of each group, with the group's own words under them.
        </p>
        <ul onMouseLeave={() => onHover(null)}>
          {groups.map((c) => (
            <li key={c.id}>
              <button
                type="button"
                className={`legend-row ${pinned === c.id ? "is-on" : ""} ${
                  pinned != null && pinned !== c.id ? "is-off" : ""
                }`}
                aria-pressed={pinned === c.id}
                onMouseEnter={() => onHover(c.id)}
                onFocus={() => onHover(c.id)}
                onBlur={() => onHover(null)}
                onClick={() => onPin(pinned === c.id ? null : c.id)}
              >
                <i style={{ background: c.color }} />
                <span className="legend-name">{c.label}</span>
                <span className="legend-count">{c.size.toLocaleString()}</span>
                <span className="legend-sentence">{c.keywords.slice(0, 4).join(" · ")}</span>
              </button>
            </li>
          ))}
        </ul>
        {unassigned > 0 && (
          <p className="legend-loose">
            <i style={{ background: UNASSIGNED }} />
            <span>
              {unassigned.toLocaleString()} films were left unassigned by the clustering method.
            </span>
          </p>
        )}
      </section>

      <section className="filter">
        <div className="section-head">
          <h2>Released</h2>
          <span className="muted">
            {yearRange[0]} – {yearRange[1]}
          </span>
        </div>
        <YearRange extent={yearExtent} value={yearRange} onChange={onYearRange} />
      </section>

      <footer className="credits">
        {data && (
          <p>
            Reviews from Letterboxd via a{" "}
            <a
              href="https://huggingface.co/datasets/PratikDhonde/letterboxd-all-movie-data"
              target="_blank"
              rel="noreferrer"
            >
              Hugging Face dataset
            </a>
            . Embedded with <code>{data.meta.embedding_model}</code>, grouped with{" "}
            {methodName(data.meta.cluster_method)}.
          </p>
        )}
        {data && data.meta.poster_source !== "none" && data.meta.poster_source !== "letterboxd" && (
          <p>This product uses the TMDB API but is not endorsed or certified by TMDB.</p>
        )}
      </footer>
    </aside>
  );
}
