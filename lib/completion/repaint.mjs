/**
 * Small unphotographed triangles that no estimated wall replaces.
 *
 * Deleting them would open holes in the surface, so they keep their shape and
 * take the colour of the photographed surface around them, spread inwards from
 * the edge of each patch.
 */

import { weldPositions } from '../meshdata.mjs';
import { faceColor } from './observed.mjs';

const SWATCH = 5;

/** One colour per corner of every listed face: Float32Array(faces * 9). */
export function spreadColours(mesh, textures, keep, faces, fallback = [128, 124, 112]) {
  const { map, count } = weldPositions(mesh.positions);
  const colour = new Float32Array(count * 3);
  const known = new Uint8Array(count);
  const wanted = new Uint8Array(count);
  for (const f of faces) for (let k = 0; k < 3; k += 1) wanted[map[mesh.indices[f * 3 + k]]] = 1;

  const sum = new Float64Array(count * 3);
  const n = new Uint32Array(count);
  const sample = [0, 0, 0];
  for (let f = 0; f < mesh.indices.length / 3; f += 1) {
    if (!keep[f]) continue;
    for (let k = 0; k < 3; k += 1) {
      const v = map[mesh.indices[f * 3 + k]];
      // Sampled a little inside the face, away from the edge of its atlas patch.
      if (!wanted[v] || !faceColor(mesh, textures, f, ...[0, 1, 2].map((c) => (c === k ? 0.7 : 0.15)), sample)) continue;
      for (let c = 0; c < 3; c += 1) sum[v * 3 + c] += sample[c];
      n[v] += 1;
    }
  }
  for (let v = 0; v < count; v += 1) {
    if (!n[v]) continue;
    for (let c = 0; c < 3; c += 1) colour[v * 3 + c] = sum[v * 3 + c] / n[v];
    known[v] = 1;
  }

  for (let pass = 0; pass < 60; pass += 1) {
    const reached = new Map();
    for (const f of faces) {
      const ids = [0, 1, 2].map((k) => map[mesh.indices[f * 3 + k]]);
      for (const v of ids) {
        if (known[v]) continue;
        for (const u of ids) {
          if (!known[u]) continue;
          const entry = reached.get(v) || [0, 0, 0, 0];
          for (let c = 0; c < 3; c += 1) entry[c] += colour[u * 3 + c];
          entry[3] += 1;
          reached.set(v, entry);
        }
      }
    }
    if (!reached.size) break;
    for (const [v, entry] of reached) {
      for (let c = 0; c < 3; c += 1) colour[v * 3 + c] = entry[c] / entry[3];
      known[v] = 1;
    }
  }

  const out = new Float32Array(faces.length * 9);
  faces.forEach((f, i) => {
    for (let k = 0; k < 3; k += 1) {
      const v = map[mesh.indices[f * 3 + k]];
      for (let c = 0; c < 3; c += 1) out[i * 9 + k * 3 + c] = known[v] ? colour[v * 3 + c] : fallback[c];
    }
  });
  return out;
}

/**
 * A strip of tiny texture swatches, one per face, each shading between its three
 * corner colours.
 *
 * @returns {{width:number, height:number, data:Uint8Array, uv:(face:number, corner:number)=>number[]}}
 */
export function swatchImage(colours, width) {
  const faces = colours.length / 9;
  const across = Math.floor(width / SWATCH);
  const height = Math.max(1, Math.ceil(faces / across)) * SWATCH;
  const data = new Uint8Array(width * height * 3);
  // Corner texels: (0.5, 0.5), (SWATCH - 0.5, 0.5), (0.5, SWATCH - 0.5).
  const span = SWATCH - 1;
  for (let f = 0; f < faces; f += 1) {
    const x0 = (f % across) * SWATCH;
    const y0 = ((f / across) | 0) * SWATCH;
    for (let y = 0; y < SWATCH; y += 1) {
      for (let x = 0; x < SWATCH; x += 1) {
        const b = x / span;
        const c = y / span;
        for (let k = 0; k < 3; k += 1) {
          const value = colours[f * 9 + k] * (1 - b - c) + colours[f * 9 + 3 + k] * b + colours[f * 9 + 6 + k] * c;
          data[((y0 + y) * width + x0 + x) * 3 + k] = Math.max(0, Math.min(255, value));
        }
      }
    }
  }
  const uv = (face, corner) => [
    (face % across) * SWATCH + (corner === 1 ? SWATCH - 0.5 : 0.5),
    ((face / across) | 0) * SWATCH + (corner === 2 ? SWATCH - 0.5 : 0.5),
  ];
  return { width, height, data, uv };
}
