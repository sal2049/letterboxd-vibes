import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { LinearInterpolator, OrthographicView } from "@deck.gl/core";
import type { PickingInfo } from "@deck.gl/core";
import { ScatterplotLayer } from "@deck.gl/layers";
import { SURFACE } from "../colors";
import type { Cluster, FilmPoint, Focus, MoodDataset } from "../types";
import { Poster } from "./Poster";

interface ViewState {
  target: [number, number, number];
  zoom: number;
  minZoom?: number;
  maxZoom?: number;
  transitionDuration?: number;
  transitionInterpolator?: LinearInterpolator;
}

interface Props {
  data: MoodDataset;
  /** the group pinned or hovered in the legend; everything else recedes */
  active: number | null;
  yearRange: [number, number];
  selected: number | null;
  focus: Focus;
  onSelect: (index: number | null) => void;
  onFocusCluster: (id: number) => void;
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

const VIEW = new OrthographicView({ id: "ortho" });
const FLY = new LinearInterpolator(["target", "zoom"]);
/** how far past "fit" you zoom before dots turn into posters */
const POSTER_ZOOM = 1.15;
const MAX_POSTERS = 48;
const PAD = 56;
const LABEL_MAX_WIDTH = 236;

function hits(a: Box, b: Box, gap = 0): boolean {
  return (
    a.x < b.x + b.w + gap && a.x + a.w + gap > b.x && a.y < b.y + b.h + gap && a.y + a.h + gap > b.y
  );
}

function bounds(points: FilmPoint[]) {
  let x0 = Infinity,
    x1 = -Infinity,
    y0 = Infinity,
    y1 = -Infinity;
  for (const p of points) {
    if (p.x < x0) x0 = p.x;
    if (p.x > x1) x1 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.y > y1) y1 = p.y;
  }
  return { x0, x1, y0, y1 };
}

function fitView(points: FilmPoint[], w: number, h: number, pad = PAD) {
  const b = bounds(points);
  const dx = Math.max(b.x1 - b.x0, 1e-6);
  const dy = Math.max(b.y1 - b.y0, 1e-6);
  const zoom = Math.log2(Math.min((w - pad * 2) / dx, (h - pad * 2) / dy));
  return { target: [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, 0] as [number, number, number], zoom };
}

export function MoodMap({
  data,
  active,
  yearRange,
  selected,
  focus,
  onSelect,
  onFocusCluster,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<ViewState | null>(null);
  const [hover, setHover] = useState<{ index: number; x: number; y: number } | null>(null);
  const points = data.points;

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: Math.round(width), h: Math.round(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const home = useMemo(
    () => (size.w && size.h && points.length ? fitView(points, size.w, size.h) : null),
    [points, size.w, size.h],
  );

  // Camera moves requested from outside (legend, search, reset) and the first fit.
  const lastFocus = useRef<Focus | null>(null);
  const lastData = useRef<MoodDataset | null>(null);
  useEffect(() => {
    if (!home) return;
    const limits = { minZoom: home.zoom - 0.6, maxZoom: home.zoom + 6 };
    const fresh = lastData.current !== data || view === null;
    lastData.current = data;
    if (fresh) {
      lastFocus.current = focus;
      setView({ ...home, ...limits });
      return;
    }
    if (lastFocus.current === focus) {
      // Container was resized: keep the reader's position, refresh the limits.
      setView((v) => (v ? { ...v, ...limits, transitionDuration: 0 } : v));
      return;
    }
    lastFocus.current = focus;
    let next = home;
    if (focus.kind === "cluster") {
      const members = points.filter((p) => p.cluster === focus.id);
      if (members.length) {
        const f = fitView(members, size.w, size.h, 90);
        next = { target: f.target, zoom: Math.min(f.zoom, home.zoom + 2.2) };
      }
    } else if (focus.kind === "film") {
      const p = points[focus.index];
      if (p) next = { target: [p.x, p.y, 0], zoom: Math.max(view?.zoom ?? 0, home.zoom + 2.4) };
    }
    setView({ ...next, ...limits, transitionDuration: 650, transitionInterpolator: FLY });
    // `view` is read, not tracked: this effect only answers focus/data/size changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus, home, data]);

  const zoomBy = (delta: number) =>
    setView((v) =>
      v
        ? {
            ...v,
            zoom: Math.min(v.maxZoom ?? 99, Math.max(v.minZoom ?? -99, v.zoom + delta)),
            transitionDuration: 250,
            transitionInterpolator: FLY,
          }
        : v,
    );

  const [yMin, yMax] = yearRange;

  const layers = useMemo(() => {
    const visible = (p: FilmPoint) =>
      (active == null || p.cluster === active) &&
      (p.year == null || (p.year >= yMin && p.year <= yMax));

    const base = new ScatterplotLayer<FilmPoint>({
      id: "films",
      data: points,
      getPosition: (p) => [p.x, p.y],
      getRadius: (p) => (p.cluster < 0 ? 0.022 : 0.03),
      radiusUnits: "common",
      radiusMinPixels: 2.2,
      radiusMaxPixels: 9,
      getFillColor: (p) => {
        const [r, g, b] = p.rgb;
        if (!visible(p)) return [r, g, b, 22];
        return [r, g, b, p.cluster < 0 ? 120 : 230];
      },
      stroked: true,
      getLineColor: [...SURFACE, 255],
      lineWidthUnits: "pixels",
      getLineWidth: 0.75,
      pickable: true,
      updateTriggers: { getFillColor: [active, yMin, yMax] },
    });

    const ringed: { p: FilmPoint; kind: "near" | "hover" | "selected" }[] = [];
    if (selected != null && points[selected]) {
      for (const i of points[selected].nn)
        if (points[i]) ringed.push({ p: points[i], kind: "near" });
      ringed.push({ p: points[selected], kind: "selected" });
    }
    if (hover && points[hover.index] && hover.index !== selected) {
      ringed.push({ p: points[hover.index], kind: "hover" });
    }

    const rings = new ScatterplotLayer<(typeof ringed)[number]>({
      id: "rings",
      data: ringed,
      getPosition: (d) => [d.p.x, d.p.y],
      getRadius: (d) => (d.kind === "near" ? 6 : 9),
      radiusUnits: "pixels",
      filled: true,
      getFillColor: (d) => [...d.p.rgb, 255] as [number, number, number, number],
      stroked: true,
      getLineColor: (d) => (d.kind === "near" ? [255, 255, 255, 150] : [255, 255, 255, 255]),
      lineWidthUnits: "pixels",
      getLineWidth: (d) => (d.kind === "near" ? 1.25 : 2),
    });

    return [base, rings];
  }, [points, active, yMin, yMax, selected, hover]);

  // Everything drawn in HTML above the canvas is placed here, in one pass, so
  // nothing can land on top of anything else: group labels first, then posters
  // (most-liked first). Controls live in the bar under the canvas, never over
  // the data.
  const overlay = useMemo(() => {
    if (!view || !home || !size.w) return { labels: [], posters: [], posterMode: false };
    const scale = 2 ** view.zoom;
    const [tx, ty] = view.target;
    const sx = (x: number) => (x - tx) * scale + size.w / 2;
    const sy = (y: number) => (y - ty) * scale + size.h / 2;

    const taken: Box[] = [];

    // There are too many groups to label at once. From far out only the
    // largest are named; more appear with every step of zoom, and the one
    // pinned in the legend always goes first.
    const order = data.clusters
      .filter((c) => c.id >= 0)
      .sort((a, b) => Number(b.id === active) - Number(a.id === active) || b.size - a.size);
    const maxLabels = Math.round(10 + Math.max(0, view.zoom - home.zoom) * 16);

    const labels: { c: Cluster; x: number; y: number }[] = [];
    for (const c of order) {
      if (labels.length >= maxLabels) break;
      // Theme labels are phrases; long ones wrap onto a second or third line.
      const textWidth = c.label.length * 7.1;
      const w = Math.min(LABEL_MAX_WIDTH, textWidth + 38);
      const lines = Math.min(3, Math.ceil(textWidth / (LABEL_MAX_WIDTH - 38)));
      const h = 30 + (lines - 1) * 16;
      const cx = sx(c.cx);
      const cy = sy(c.cy);
      if (cx < 0 || cx > size.w || cy < 0 || cy > size.h) continue;
      // Try the centroid, then just above and below it.
      for (const dy of [0, -h - 8, h + 8]) {
        const box = {
          x: Math.min(Math.max(cx - w / 2, 8), size.w - w - 8),
          y: Math.min(Math.max(cy - h / 2 + dy, 8), size.h - h - 8),
          w,
          h,
        };
        if (taken.some((t) => hits(t, box, 6))) continue;
        taken.push(box);
        labels.push({ c, x: box.x, y: box.y });
        break;
      }
    }

    const posterMode = view.zoom - home.zoom >= POSTER_ZOOM;
    const posters: { i: number; x: number; y: number; w: number; h: number }[] = [];
    if (posterMode) {
      const grow = Math.min(1, (view.zoom - home.zoom - POSTER_ZOOM) / 2);
      const w = Math.round(46 + grow * 26);
      const h = Math.round(w * 1.5);
      const candidates: number[] = [];
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (!p.poster) continue;
        if (active != null && p.cluster !== active) continue;
        if (p.year != null && (p.year < yMin || p.year > yMax)) continue;
        const x = sx(p.x);
        const y = sy(p.y);
        if (x < w / 2 + 6 || x > size.w - w / 2 - 6 || y < h / 2 + 6 || y > size.h - h / 2 - 6)
          continue;
        candidates.push(i);
      }
      // The film being read about always gets its poster; then most-liked first.
      const rank = (i: number) => (i === selected ? Infinity : points[i].pop);
      candidates.sort((a, b) => rank(b) - rank(a));
      for (const i of candidates) {
        if (posters.length >= MAX_POSTERS) break;
        const p = points[i];
        const box = { x: sx(p.x) - w / 2, y: sy(p.y) - h / 2, w, h };
        if (taken.some((t) => hits(t, box, 8))) continue;
        taken.push(box);
        posters.push({ i, ...box });
      }
    }
    return { labels, posters, posterMode };
  }, [view, home, size.w, size.h, data.clusters, points, active, yMin, yMax, selected]);

  const onHover = useCallback((info: PickingInfo<FilmPoint>) => {
    if (info.layer?.id === "films" && info.index >= 0) {
      setHover({ index: info.index, x: info.x, y: info.y });
    } else {
      setHover(null);
    }
  }, []);

  const onClick = useCallback(
    (info: PickingInfo<FilmPoint>) => {
      onSelect(info.layer?.id === "films" && info.index >= 0 ? info.index : null);
    },
    [onSelect],
  );

  const tip = hover ? points[hover.index] : null;
  const tipCluster = tip ? data.clusters.find((c) => c.id === tip.cluster) : null;
  const tipLeft = hover && hover.x > size.w - 300;
  const tipUp = hover && hover.y > size.h - 150;

  return (
    <div className="map">
      <div className="map-canvas" ref={wrapRef}>
        {view && (
          <DeckGL
            views={VIEW}
            viewState={view}
            onViewStateChange={({ viewState }) => setView(viewState as ViewState)}
            controller={{
              dragRotate: false,
              scrollZoom: { speed: 0.012, smooth: true },
              inertia: 250,
            }}
            layers={layers}
            pickingRadius={12}
            onHover={onHover}
            onClick={onClick}
            getCursor={({ isDragging, isHovering }) =>
              isDragging ? "grabbing" : isHovering ? "pointer" : "grab"
            }
          />
        )}

        <div className="map-overlay">
          {overlay.posters.map(({ i, x, y, w, h }) => (
            <button
              key={points[i].id}
              type="button"
              className={`map-poster ${i === selected ? "is-selected" : ""}`}
              style={{ left: x, top: y, width: w, height: h }}
              title={points[i].title}
              onClick={() => onSelect(i)}
            >
              <Poster film={points[i]} size="w154" eager />
            </button>
          ))}

          {overlay.labels.map(({ c, x, y }) => (
            <button
              key={c.id}
              type="button"
              className={`map-label ${active == null || active === c.id ? "" : "is-dim"}`}
              style={{ left: x, top: y, maxWidth: LABEL_MAX_WIDTH, ["--c" as string]: c.color }}
              onClick={() => onFocusCluster(c.id)}
            >
              <i />
              {c.label}
            </button>
          ))}

          {tip && hover && (
            <div
              className="map-tip"
              style={{
                left: hover.x,
                top: hover.y,
                transform: `translate(${tipLeft ? "calc(-100% - 16px)" : "16px"}, ${
                  tipUp ? "calc(-100% - 12px)" : "12px"
                })`,
              }}
            >
              <Poster film={tip} size="w92" eager />
              <div>
                <strong>{tip.title}</strong>
                <span className="muted">
                  {tip.year ?? "—"}
                  {tip.director ? ` · ${tip.director}` : ""}
                </span>
                <span className="tip-mood">
                  <i style={{ background: tip.hex }} />
                  {tipCluster ? tipCluster.label : "Unassigned"}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="map-bar">
        <p className="map-hint">
          {overlay.posterMode
            ? "Showing the most-liked films in view · click one for its reviews"
            : "Scroll to zoom · drag to pan · zoom in to see posters"}
        </p>
        <div className="map-zoom">
          <button type="button" onClick={() => zoomBy(0.8)} aria-label="Zoom in">
            +
          </button>
          <button type="button" onClick={() => zoomBy(-0.8)} aria-label="Zoom out">
            −
          </button>
          <button
            type="button"
            aria-label="Fit the whole map"
            onClick={() =>
              home &&
              setView((v) => ({
                ...v,
                ...home,
                transitionDuration: 500,
                transitionInterpolator: FLY,
              }))
            }
          >
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
              <path
                d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
