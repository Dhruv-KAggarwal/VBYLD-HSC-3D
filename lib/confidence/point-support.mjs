/**
 * Dense-point support: does measured geometry actually lie on this face?
 *
 * Both terms are relative to the cloud's own sampling (distance in multiples of
 * the median point spacing, density against the scene's median), so a sparser
 * reconstruction is not penalised wholesale.
 */

import { median } from '../spatial.mjs';

const MAX_COUNT = 200;
/** The neighbourhood searched around a face, in cloud spacings. */
const MIN_RADIUS = 3;
const MAX_RADIUS = 10;

/** @returns {{support:Float32Array, density:Float32Array}} support is 0..1 */
export function supportPerFace(cloud, mesh, samples, faces = null) {
  const list = faces || Array.from({ length: samples.faces }, (_, i) => i);
  const support = new Float32Array(samples.faces).fill(-1);
  const closeness = new Float32Array(samples.faces);
  const counts = new Float32Array(samples.faces);
  const s = cloud.spacing;
  const p = mesh.positions;
  for (const f of list) {
    const a = mesh.indices[f * 3] * 3;
    const b = mesh.indices[f * 3 + 1] * 3;
    const c = mesh.indices[f * 3 + 2] * 3;
    // Centroid and the three points halfway towards the corners, so a large
    // face is not judged by its middle alone.
    const probes = [[samples.centroid[f * 3], samples.centroid[f * 3 + 1], samples.centroid[f * 3 + 2]]];
    for (const v of [a, b, c]) {
      probes.push([0, 1, 2].map((k) => (samples.centroid[f * 3 + k] + p[v + k]) / 2));
    }
    let score = 0;
    for (const [x, y, z] of probes) {
      const hit = cloud.grid.nearest(x, y, z, 4 * s);
      if (hit) score += Math.min(1, Math.max(0, 1 - (hit.distance - s) / (3 * s)));
    }
    closeness[f] = score / probes.length;
    // Normalised by neighbourhood area so large faces compare with small ones.
    // The radius is capped: a huge bridging face must not turn one query into
    // millions of cell visits.
    const radius = Math.min(MAX_RADIUS * s, Math.max(MIN_RADIUS * s, Math.sqrt(samples.area[f])));
    let count = 0;
    cloud.grid.forEachWithin(...probes[0], radius, () => {
      count += 1;
      return count < MAX_COUNT;
    });
    counts[f] = count / (radius * radius);
  }
  const populated = [];
  const step = Math.max(1, Math.floor(list.length / 20000));
  for (let i = 0; i < list.length; i += step) if (counts[list[i]] > 0) populated.push(counts[list[i]]);
  const typical = median(populated) || 1;
  const density = new Float32Array(samples.faces);
  for (const f of list) {
    density[f] = Math.min(1, counts[f] / typical);
    support[f] = 0.65 * closeness[f] + 0.35 * density[f];
  }
  return { support, density };
}
