/**
 * Measure how far a simplified surface is from the original, and reject levels
 * that are invalid or distorted.
 *
 * geometricError, as written to the manifest, is the 95th percentile surface
 * distance in metres (scene units where the scale is unverified), taken as the
 * larger of the two directions:
 *   original -> level  detects detail that was lost,
 *   level -> original  detects surface that was invented (bridging).
 */

import { MeshBVH } from '../bvh.mjs';
import { faceGeometry } from '../glbmesh.mjs';

/** Largest accepted geometricError as a fraction of the scene diagonal. */
export const ERROR_LIMITS = Object.freeze({ 1: 0.002, 2: 0.004, 3: 0.01, 4: 0.025 });

/** Area-weighted, deterministic surface samples. */
export function surfaceSamples(positions, indices, count) {
  const faces = indices.length / 3;
  const cumulative = new Float64Array(faces);
  let total = 0;
  for (let f = 0; f < faces; f += 1) {
    total += faceGeometry(positions, indices, f).area;
    cumulative[f] = total;
  }
  const out = new Float32Array(count * 3);
  if (!(total > 0)) return out;
  let face = 0;
  let a = 0.5;
  let b = 0.5;
  for (let i = 0; i < count; i += 1) {
    const target = ((i + 0.5) / count) * total;
    while (face < faces - 1 && cumulative[face] < target) face += 1;
    // R2 low-discrepancy sequence for the position within the face.
    a = (a + 0.7548776662466927) % 1;
    b = (b + 0.5698402909980532) % 1;
    const r = Math.sqrt(a);
    const w = [1 - r, r * (1 - b), r * b];
    for (let k = 0; k < 3; k += 1) {
      out[i * 3 + k] = w[0] * positions[indices[face * 3] * 3 + k]
        + w[1] * positions[indices[face * 3 + 1] * 3 + k]
        + w[2] * positions[indices[face * 3 + 2] * 3 + k];
    }
  }
  return out;
}

function distances(samples, bvh) {
  const out = new Float64Array(samples.length / 3);
  let hint = -1;
  for (let i = 0; i < out.length; i += 1) {
    const hit = bvh.closestPoint(samples[i * 3], samples[i * 3 + 1], samples[i * 3 + 2], Infinity, hint);
    out[i] = hit ? hit.distance : 0;
    hint = hit ? hit.face : -1;
  }
  out.sort();
  let sum = 0;
  let square = 0;
  for (const d of out) { sum += d; square += d * d; }
  return {
    mean: sum / out.length,
    rms: Math.sqrt(square / out.length),
    p95: out[Math.floor(out.length * 0.95)],
    max: out[out.length - 1],
  };
}

export function createErrorMeter(original, metresPerUnit, count = 12000) {
  const originalBvh = new MeshBVH(original.positions, original.indices);
  const originalSamples = surfaceSamples(original.positions, original.indices, count);
  return {
    originalBvh,
    measure(positions, indices) {
      const forward = distances(originalSamples, new MeshBVH(positions, indices));
      const backward = distances(surfaceSamples(positions, indices, count), originalBvh);
      const scale = (stats) => Object.fromEntries(
        Object.entries(stats).map(([k, v]) => [k, Math.round(v * metresPerUnit * 100000) / 100000]),
      );
      return {
        geometricError: Math.round(Math.max(forward.p95, backward.p95) * metresPerUnit * 100000) / 100000,
        p95Units: Math.max(forward.p95, backward.p95),
        originalToLevel: scale(forward),
        levelToOriginal: scale(backward),
        samplesEachWay: count,
      };
    },
  };
}

/** Structural validity of a level before it is written. */
export function checkGeometry(positions, indices) {
  if (!indices.length || indices.length % 3) return 'The level has no triangles.';
  const vertices = positions.length / 3;
  for (let i = 0; i < indices.length; i += 1) {
    const v = indices[i];
    if (v >= vertices) return 'The level references a missing vertex.';
    if (!Number.isFinite(positions[v * 3]) || !Number.isFinite(positions[v * 3 + 1]) || !Number.isFinite(positions[v * 3 + 2])) {
      return 'The level contains invalid coordinates.';
    }
  }
  return null;
}
