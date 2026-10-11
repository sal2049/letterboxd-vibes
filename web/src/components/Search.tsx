import { useEffect, useMemo, useRef, useState } from "react";
import type { FilmPoint } from "../types";
import { Poster } from "./Poster";

interface Props {
  points: FilmPoint[];
  onPick: (index: number) => void;
}

export function Search({ points, onPick }: Props) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);

  const results = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (needle.length < 2) return [];
    const out: { i: number; rank: number }[] = [];
    for (let i = 0; i < points.length; i++) {
      const at = points[i].title.toLowerCase().indexOf(needle);
      if (at >= 0) out.push({ i, rank: (at === 0 ? 0 : 1e9) - points[i].pop });
    }
    return out.sort((a, b) => a.rank - b.rank).slice(0, 7);
  }, [q, points]);

  useEffect(() => setCursor(0), [q]);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const pick = (i: number) => {
    onPick(i);
    setQ("");
    setOpen(false);
  };

  return (
    <div className="search" ref={boxRef}>
      <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
        <circle cx="7" cy="7" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M10.5 10.5 14 14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      <input
        type="search"
        value={q}
        placeholder="Find a film"
        aria-label="Find a film"
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setCursor((c) => Math.min(c + 1, results.length - 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setCursor((c) => Math.max(c - 1, 0));
          } else if (e.key === "Enter" && results[cursor]) {
            pick(results[cursor].i);
          } else if (e.key === "Escape") {
            setOpen(false);
          }
        }}
      />
      {open && q.trim().length >= 2 && (
        <ul className="search-results">
          {results.length === 0 && <li className="search-empty">No film with that title</li>}
          {results.map(({ i }, n) => (
            <li key={points[i].id}>
              <button
                type="button"
                className={n === cursor ? "is-cursor" : ""}
                onMouseEnter={() => setCursor(n)}
                onClick={() => pick(i)}
              >
                <Poster film={points[i]} size="w92" eager />
                <span>
                  <strong>{points[i].title}</strong>
                  <span className="muted">
                    {points[i].year ?? "—"}
                    {points[i].director ? ` · ${points[i].director}` : ""}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
