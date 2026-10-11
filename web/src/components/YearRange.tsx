interface Props {
  extent: [number, number];
  value: [number, number];
  onChange: (range: [number, number]) => void;
}

/** Two-thumb range on one track. */
export function YearRange({ extent, value, onChange }: Props) {
  const [lo, hi] = extent;
  const [a, b] = value;
  const span = Math.max(1, hi - lo);
  const left = ((a - lo) / span) * 100;
  const right = ((b - lo) / span) * 100;

  return (
    <div className="range">
      <div className="range-track">
        <b style={{ left: `${left}%`, width: `${right - left}%` }} />
      </div>
      <input
        type="range"
        aria-label="Earliest year"
        min={lo}
        max={hi}
        value={a}
        onChange={(e) => onChange([Math.min(Number(e.target.value), b), b])}
      />
      <input
        type="range"
        aria-label="Latest year"
        min={lo}
        max={hi}
        value={b}
        onChange={(e) => onChange([a, Math.max(Number(e.target.value), a)])}
      />
    </div>
  );
}
