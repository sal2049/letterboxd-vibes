import type { RGB } from "./types";

// Group colours come with the data: the pipeline derives each one from where
// the group sits on the map, so neighbouring groups shade into each other.
// Only the neutrals live here.
export const UNASSIGNED = "#59606d";
export const SURFACE: RGB = [15, 17, 21];

export function hexToRgb(hex: string): RGB {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
