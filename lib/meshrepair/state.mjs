/**
 * The working mesh during repair, with per-face provenance and source ids that
 * every edit has to carry along.
 */

import { faceCount, vertexCount } from '../meshdata.mjs';
import { faceGeometry } from '../glbmesh.mjs';

export function createState(mesh) {
  const faces = faceCount(mesh);
  const source = new Int32Array(faces);
  for (let f = 0; f < faces; f += 1) source[f] = f;
  return { mesh, source, codes: new Uint8Array(faces) };
}

/** Raise a face's provenance; it never moves back toward "observed". */
export function mark(state, face, code) {
  if (code > state.codes[face]) state.codes[face] = code;
}

/** Keep only the faces for which `keep[f]` is truthy. Returns removed source ids. */
export function filterFaces(state, keep) {
  const { mesh, source, codes } = state;
  const faces = faceCount(mesh);
  let kept = 0;
  for (let f = 0; f < faces; f += 1) if (keep[f]) kept += 1;
  if (kept === faces) return [];
  const indices = new Uint32Array(kept * 3);
  const facePrim = new Uint16Array(kept);
  const nextSource = new Int32Array(kept);
  const nextCodes = new Uint8Array(kept);
  const removed = [];
  let w = 0;
  for (let f = 0; f < faces; f += 1) {
    if (!keep[f]) {
      if (source[f] >= 0) removed.push(source[f]);
      continue;
    }
    indices[w * 3] = mesh.indices[f * 3];
    indices[w * 3 + 1] = mesh.indices[f * 3 + 1];
    indices[w * 3 + 2] = mesh.indices[f * 3 + 2];
    facePrim[w] = mesh.facePrim[f];
    nextSource[w] = source[f];
    nextCodes[w] = codes[f];
    w += 1;
  }
  mesh.indices = indices;
  mesh.facePrim = facePrim;
  state.source = nextSource;
  state.codes = nextCodes;
  return removed;
}

/** Drop vertices no face references. Returns how many were dropped. */
export function compactVertices(state) {
  const { mesh } = state;
  const count = vertexCount(mesh);
  const remap = new Int32Array(count).fill(-1);
  for (let i = 0; i < mesh.indices.length; i += 1) remap[mesh.indices[i]] = 0;
  let next = 0;
  for (let v = 0; v < count; v += 1) if (remap[v] === 0) { remap[v] = next; next += 1; }
  if (next === count) return 0;
  const move = (array, stride, Type) => {
    if (!array) return null;
    const out = new Type(next * stride);
    for (let v = 0; v < count; v += 1) {
      if (remap[v] < 0) continue;
      for (let k = 0; k < stride; k += 1) out[remap[v] * stride + k] = array[v * stride + k];
    }
    return out;
  };
  mesh.positions = move(mesh.positions, 3, Float32Array);
  mesh.uvs = move(mesh.uvs, 2, Float32Array);
  mesh.colors = move(mesh.colors, 4, Uint8Array);
  for (let i = 0; i < mesh.indices.length; i += 1) mesh.indices[i] = remap[mesh.indices[i]];
  return count - next;
}

export function faceAreas(mesh) {
  const areas = new Float32Array(faceCount(mesh));
  for (let f = 0; f < areas.length; f += 1) areas[f] = faceGeometry(mesh.positions, mesh.indices, f).area;
  return areas;
}

/** Median edge length over a sample: the mesh's own length scale. */
export function medianEdge(mesh, samples = 6000) {
  const faces = faceCount(mesh);
  const step = Math.max(1, Math.floor(faces / samples));
  const lengths = [];
  const p = mesh.positions;
  for (let f = 0; f < faces; f += step) {
    for (let k = 0; k < 3; k += 1) {
      const a = mesh.indices[f * 3 + k] * 3;
      const b = mesh.indices[f * 3 + ((k + 1) % 3)] * 3;
      const length = Math.hypot(p[a] - p[b], p[a + 1] - p[b + 1], p[a + 2] - p[b + 2]);
      if (Number.isFinite(length) && length > 0) lengths.push(length);
    }
  }
  lengths.sort((a, b) => a - b);
  return lengths[Math.floor(lengths.length / 2)] || 1;
}

/** Diagonal of the box around all referenced, finite vertices. */
export function boundingDiagonal(mesh) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.indices.length; i += 1) {
    const v = mesh.indices[i] * 3;
    for (let k = 0; k < 3; k += 1) {
      const x = mesh.positions[v + k];
      if (x < lo[k]) lo[k] = x;
      if (x > hi[k]) hi[k] = x;
    }
  }
  return Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
}

/** Whether dense points lie near a position. */
export function supported(cloud, x, y, z, radius) {
  let found = false;
  // Capped: a huge face must not turn one query into millions of cell visits.
  cloud.grid.forEachWithin(x, y, z, Math.min(radius, 12 * cloud.spacing), () => { found = true; return false; });
  return found;
}
