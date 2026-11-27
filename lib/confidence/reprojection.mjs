/**
 * Reprojection consistency of the sparse landmarks near each face.
 *
 * COLMAP's per-point error is the distance in pixels between where a feature was
 * observed and where the solved geometry says it should be. It measures
 * self-consistency of the solution, not metric accuracy.
 */

import { PointGrid, estimateSpacing } from '../spatial.mjs';

const MIN_POINTS = 3;
const MAX_POINTS = 64;
export const GOOD_ERROR_PX = 0.3;
export const BAD_ERROR_PX = 2;

export const reprojectionScore = (error) => Math.min(1, Math.max(0, 1 - (error - GOOD_ERROR_PX) / (BAD_ERROR_PX - GOOD_ERROR_PX)));

/**
 * @returns {{median:Float32Array, observations:Float32Array, radius:number}|null}
 *   median is -1 where too few landmarks lie nearby
 */
export function reprojectionPerFace(sparse, samples) {
  if (!sparse?.count) return null;
  const coarse = new PointGrid(sparse.positions);
  const spacing = estimateSpacing(coarse);
  const grid = new PointGrid(sparse.positions, spacing * 3);
  const median = new Float32Array(samples.faces).fill(-1);
  const observations = new Float32Array(samples.faces);
  const errors = new Float64Array(MAX_POINTS);
  for (let f = 0; f < samples.faces; f += 1) {
    const radius = Math.max(3 * spacing, 2 * Math.sqrt(samples.area[f]));
    let count = 0;
    let tracks = 0;
    grid.forEachWithin(samples.centroid[f * 3], samples.centroid[f * 3 + 1], samples.centroid[f * 3 + 2], radius, (i) => {
      if (sparse.errors[i] < 0) return true;
      errors[count] = sparse.errors[i];
      tracks += sparse.trackLengths[i];
      count += 1;
      return count < MAX_POINTS;
    });
    if (count < MIN_POINTS) continue;
    const sorted = errors.subarray(0, count).sort();
    median[f] = sorted[count >> 1];
    observations[f] = tracks / count;
  }
  return { median, observations, radius: 3 * spacing };
}
