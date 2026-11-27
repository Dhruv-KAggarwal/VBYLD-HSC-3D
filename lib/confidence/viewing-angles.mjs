/**
 * Viewpoint geometry statistics.
 *
 * Ten frames from nearly the same spot constrain depth no better than one, so
 * what matters is the angular spread of the viewing rays. It is measured from
 * the length R of the mean unit direction: acos(R) is zero when every ray is
 * parallel and grows with the spread.
 */

/** Angular spread (degrees) at which viewpoint diversity is considered full. */
export const FULL_SPREAD_DEGREES = 20;
/** Rays shallower than this to the surface (80 degrees off normal) do not count. */
export const MIN_INCIDENCE_COS = Math.cos((80 * Math.PI) / 180);

export function spreadDegrees(sumX, sumY, sumZ, count) {
  if (count < 2) return 0;
  const length = Math.hypot(sumX, sumY, sumZ) / count;
  return (Math.acos(Math.min(1, Math.max(-1, length))) * 180) / Math.PI;
}

/** 0..1; highly oblique-only coverage is discounted. */
export function diversityScore(spread, bestCos, count) {
  if (count < 2) return 0;
  return Math.min(1, spread / FULL_SPREAD_DEGREES) * (0.4 + 0.6 * Math.min(1, Math.max(0, bestCos)));
}
