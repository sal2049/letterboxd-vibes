import { useCallback, useEffect, useMemo, useState } from "react";
import { loadDataset, yearExtent } from "./data";
import type { Focus, MoodDataset } from "./types";
import { ClusterDetail, FilmDetail, Overview } from "./components/Detail";
import { Gallery } from "./components/Gallery";
import { MoodMap } from "./components/MoodMap";
import { Search } from "./components/Search";
import { Sidebar } from "./components/Sidebar";

type ViewMode = "map" | "moods";

export default function App() {
  const [mode, setMode] = useState<ViewMode>("map");
  const [data, setData] = useState<MoodDataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [years, setYears] = useState<[number, number]>([1900, 2030]);
  const [focus, setFocus] = useState<Focus>({ kind: "fit", nonce: 0 });

  useEffect(() => {
    let stale = false;
    setData(null);
    setError(null);
    setSelected(null);
    setPinned(null);
    setHovered(null);
    loadDataset("en")
      .then((d) => {
        if (stale) return;
        setData(d);
        setYears(yearExtent(d.points));
      })
      .catch((e: Error) => !stale && setError(e.message));
    return () => {
      stale = true;
    };
  }, []);

  const extent = useMemo(() => (data ? yearExtent(data.points) : years), [data, years]);

  /** Pin a group (or clear with null) and move the camera to it. */
  const pin = useCallback((id: number | null) => {
    setSelected(null);
    setPinned(id);
    setFocus(
      id == null ? { kind: "fit", nonce: Date.now() } : { kind: "cluster", id, nonce: Date.now() },
    );
  }, []);

  /** Select a film and bring it into view (search, neighbours, shelves). */
  const goToFilm = useCallback(
    (index: number) => {
      setSelected(index);
      setFocus({ kind: "film", index, nonce: Date.now() });
      // A pinned group would hide a film from another one.
      setPinned((p) => (p != null && data?.points[index]?.cluster !== p ? null : p));
    },
    [data],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (selected != null) setSelected(null);
      else if (pinned != null) pin(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, pinned, pin]);

  const film = data && selected != null ? data.points[selected] : null;
  const cluster = data && pinned != null ? data.clusters.find((c) => c.id === pinned) : null;

  return (
    <div className="app">
      <Sidebar
        data={data}
        pinned={pinned}
        onPin={pin}
        onHover={setHovered}
        yearExtent={extent}
        yearRange={years}
        onYearRange={setYears}
      />

      <header className="toolbar">
        <div className="segmented" role="tablist" aria-label="View">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "map"}
            className={mode === "map" ? "is-on" : ""}
            onClick={() => setMode("map")}
          >
            Map
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "moods"}
            className={mode === "moods" ? "is-on" : ""}
            onClick={() => setMode("moods")}
          >
            Shelves
          </button>
        </div>

        <Search points={data?.points ?? []} onPick={goToFilm} />
      </header>

      <main className="stage">
        {error && <p className="empty">{error}</p>}
        {!error && !data && <p className="empty">Loading films…</p>}
        {data && mode === "map" && (
          <MoodMap
            data={data}
            // A hover previews; with nothing hovered the pin applies.
            active={hovered ?? pinned}
            yearRange={years}
            selected={selected}
            focus={focus}
            onSelect={setSelected}
            onFocusCluster={pin}
          />
        )}
        {data && mode === "moods" && (
          <Gallery
            data={data}
            selected={selected}
            pinned={pinned}
            onSelect={setSelected}
            onShowOnMap={(id) => {
              setMode("map");
              pin(id);
            }}
          />
        )}
      </main>

      <aside className="panel">
        {data && film && (
          <FilmDetail
            key={film.id}
            data={data}
            film={film}
            onSelect={goToFilm}
            onClose={() => setSelected(null)}
            onPin={pin}
          />
        )}
        {data && !film && cluster && (
          <ClusterDetail
            key={cluster.id}
            data={data}
            cluster={cluster}
            onSelect={goToFilm}
            onClose={() => pin(null)}
          />
        )}
        {data && !film && !cluster && (
          <Overview
            data={data}
            onSurprise={() => {
              const pool = data.points.filter((p) => p.poster && p.cluster >= 0);
              const pick = pool[Math.floor(Math.random() * pool.length)];
              if (pick) goToFilm(data.points.indexOf(pick));
            }}
          />
        )}
      </aside>
    </div>
  );
}
