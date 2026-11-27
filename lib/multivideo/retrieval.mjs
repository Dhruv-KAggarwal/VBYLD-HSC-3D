/**
 * Image retrieval for cross-video candidate selection.
 *
 * A VLAD global descriptor (Jegou et al., CVPR 2010) is aggregated from the
 * local features COLMAP has already extracted, against a small codebook learned
 * from this job's own images. It needs no model weights, no download and no
 * GPU, and works for SIFT and learned descriptors alike. Similarity only
 * PROPOSES pairs: every proposed pair is still matched and geometrically
 * verified before it counts as overlap.
 */

import { readDescriptors, readImages } from './database.mjs';

const CLUSTERS = 48;

function nearestCentre(data, offset, centres, dim, k) {
  let best = 0;
  let bestDistance = Infinity;
  for (let c = 0; c < k; c += 1) {
    let distance = 0;
    const base = c * dim;
    for (let d = 0; d < dim; d += 1) {
      const diff = data[offset + d] - centres[base + d];
      distance += diff * diff;
      if (distance >= bestDistance) break;
    }
    if (distance < bestDistance) { bestDistance = distance; best = c; }
  }
  return best;
}

function kmeans(samples, dim, k, iterations = 8) {
  const n = samples.length / dim;
  const centres = new Float32Array(k * dim);
  // Deterministic spread initialisation, so a re-run proposes the same pairs.
  for (let c = 0; c < k; c += 1) {
    const source = Math.floor(((c + 0.5) * n) / k);
    centres.set(samples.subarray(source * dim, (source + 1) * dim), c * dim);
  }
  const sums = new Float64Array(k * dim);
  const counts = new Uint32Array(k);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    sums.fill(0);
    counts.fill(0);
    for (let i = 0; i < n; i += 1) {
      const c = nearestCentre(samples, i * dim, centres, dim, k);
      counts[c] += 1;
      for (let d = 0; d < dim; d += 1) sums[c * dim + d] += samples[i * dim + d];
    }
    for (let c = 0; c < k; c += 1) {
      if (!counts[c]) continue;
      for (let d = 0; d < dim; d += 1) centres[c * dim + d] = sums[c * dim + d] / counts[c];
    }
  }
  return centres;
}

function vlad(local, centres, dim, k) {
  const out = new Float32Array(k * dim);
  for (let i = 0; i < local.count; i += 1) {
    const c = nearestCentre(local.data, i * dim, centres, dim, k);
    for (let d = 0; d < dim; d += 1) out[c * dim + d] += local.data[i * dim + d] - centres[c * dim + d];
  }
  // Intra-cluster normalisation suppresses bursts (a facade of identical
  // windows), then a global L2 normalisation makes dot product a similarity.
  let total = 0;
  for (let c = 0; c < k; c += 1) {
    let sum = 0;
    for (let d = 0; d < dim; d += 1) sum += out[c * dim + d] ** 2;
    if (sum <= 0) continue;
    const inv = 1 / Math.sqrt(sum);
    for (let d = 0; d < dim; d += 1) { out[c * dim + d] *= inv; total += out[c * dim + d] ** 2; }
  }
  if (total > 0) {
    const inv = 1 / Math.sqrt(total);
    for (let i = 0; i < out.length; i += 1) out[i] *= inv;
  }
  return out;
}

/**
 * @returns {{names:string[], similarity:(a:number,b:number)=>number}|null}
 *   indices refer to `names`.
 */
export function buildRetrievalIndex(db, onProgress = null) {
  const images = [...readImages(db).values()].sort((a, b) => a.name.localeCompare(b.name));
  if (images.length < 2) return null;
  const perImage = Math.max(60, Math.floor(20000 / images.length));
  const sampleChunks = [];
  let dim = 0;
  let sampled = 0;
  for (const image of images) {
    const local = readDescriptors(db, image.id, perImage);
    if (!local) continue;
    dim ||= local.dim;
    if (local.dim !== dim) continue;
    sampleChunks.push(local.data);
    sampled += local.count;
  }
  if (!dim || sampled < CLUSTERS * 4) return null;
  const samples = new Float32Array(sampled * dim);
  let offset = 0;
  for (const chunk of sampleChunks) { samples.set(chunk, offset); offset += chunk.length; }
  const centres = kmeans(samples, dim, CLUSTERS);
  onProgress?.(0.4);
  // One image's descriptors are in memory at a time.
  const global = images.map((image, i) => {
    const local = readDescriptors(db, image.id, 900);
    if (i % 20 === 0) onProgress?.(0.4 + (0.6 * i) / images.length);
    return local && local.dim === dim ? vlad(local, centres, dim, CLUSTERS) : null;
  });
  const cache = new Map();
  const similarity = (a, b) => {
    const key = a < b ? a * images.length + b : b * images.length + a;
    let value = cache.get(key);
    if (value === undefined) {
      value = 0;
      const x = global[a];
      const y = global[b];
      if (x && y) for (let i = 0; i < x.length; i += 1) value += x[i] * y[i];
      cache.set(key, value);
    }
    return value;
  };
  return { names: images.map((image) => image.name), similarity, method: 'vlad', dimension: CLUSTERS * dim };
}
