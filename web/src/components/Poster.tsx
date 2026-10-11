import { useEffect, useState } from "react";
import { posterUrl } from "../data";
import type { FilmPoint } from "../types";

interface Props {
  film: FilmPoint;
  size: "w92" | "w154" | "w342" | "w500";
  className?: string;
  eager?: boolean;
}

/** 2:3 poster with a titled placeholder when there is no image (or it fails). */
export function Poster({ film, size, className, eager }: Props) {
  const src = posterUrl(film.poster, size);
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);

  return (
    <div className={`poster ${className ?? ""}`} style={{ ["--tint" as string]: film.hex }}>
      {src && !failed ? (
        <img
          src={src}
          alt=""
          loading={eager ? "eager" : "lazy"}
          decoding="async"
          referrerPolicy="no-referrer"
          draggable={false}
          onError={() => setFailed(true)}
        />
      ) : (
        <span className="poster-fallback">{film.title}</span>
      )}
    </div>
  );
}
