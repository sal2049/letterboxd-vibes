import { useEffect, useRef } from "react";
import type { MoodDataset } from "../types";
import { Poster } from "./Poster";

interface Props {
  data: MoodDataset;
  selected: number | null;
  /** group pinned in the legend; its shelf is scrolled into view */
  pinned: number | null;
  onSelect: (index: number) => void;
  onShowOnMap: (clusterId: number) => void;
}

/** The same groups as the map, as shelves: one row of the films nearest each group's centre. */
export function Gallery({ data, selected, pinned, onSelect, onShowOnMap }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (pinned == null) return;
    rootRef.current
      ?.querySelector(`[data-cluster="${pinned}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [pinned]);

  return (
    <div className="gallery" ref={rootRef}>
      {data.clusters.map((c) => {
        const quote = c.quotes[0];
        return (
          <section
            key={c.id}
            className="shelf"
            data-cluster={c.id}
            style={{ ["--c" as string]: c.color }}
          >
            <header>
              <div>
                <h2>
                  <i />
                  {c.label}
                </h2>
                <p className="muted">
                  {c.size.toLocaleString()} films
                  {c.keywords.length > 0 && <> · {c.keywords.slice(0, 4).join(", ")}</>}
                </p>
              </div>
              <button
                type="button"
                className="button button-quiet"
                onClick={() => onShowOnMap(c.id)}
              >
                Show on map
              </button>
            </header>

            <ul className="shelf-row">
              {c.exemplars.map((i) => {
                const f = data.points[i];
                if (!f) return null;
                return (
                  <li key={f.id}>
                    <button
                      type="button"
                      className={i === selected ? "is-selected" : ""}
                      onClick={() => onSelect(i)}
                    >
                      <Poster film={f} size="w342" />
                      <strong>{f.title}</strong>
                      <span className="muted">{f.year ?? "—"}</span>
                    </button>
                  </li>
                );
              })}
            </ul>

            {quote && (
              <blockquote dir="auto">
                <p>{quote.t}</p>
                <footer>
                  {quote.u || "anonymous"} on {data.points[quote.f]?.title}
                </footer>
              </blockquote>
            )}
          </section>
        );
      })}
    </div>
  );
}
