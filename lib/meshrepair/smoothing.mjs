/**
 * Edge-preserving noise smoothing.
 *
 * Taubin's lambda/mu scheme (no shrinkage), with two protections for
 * architecture: neighbours across a normal discontinuity are down-weighted
 * (bilateral), and vertices on sharp creases or open boundaries are held. Every
 * vertex is also clamped to a fraction of the length of ITS OWN edges, so a
 * finely meshed facade cannot be moved by an amount sized for coarse terrain.
 *
 * It operates on welded vertices, so the copies of a position on either side of
 * a texture seam move together and no crack can open.
 */

import { buildTopology, faceCount } from '../meshdata.mjs';
import { faceGeometry } from '../glbmesh.mjs';
import { mark, medianEdge } from './state.mjs';
import { PROVENANCE } from './provenance.mjs';

export const SMOOTHING = Object.freeze({
  off: null,
  light: { iterations: 3, featureAngle: 30, maxDisplacement: 0.12 },
  moderate: { iterations: 6, featureAngle: 35, maxDisplacement: 0.22 },
  strong: { iterations: 10, featureAngle: 45, maxDisplacement: 0.35 },
});

export function smoothingParameters({ strength = 'off', iterations = null, edgePreservation = 0.7 } = {}) {
  const base = strength === 'custom' ? { ...SMOOTHING.moderate } : SMOOTHING[strength];
  if (!base) return null;
  const count = strength === 'custom' ? Math.min(60, Math.max(1, Math.round(Number(iterations) || 5))) : base.iterations;
  return { ...base, iterations: count, edgePreservation: Math.min(1, Math.max(0, Number(edgePreservation))) };
}

export function smoothMesh(state, params) {
  const { mesh } = state;
  const topology = buildTopology(mesh);
  const { welded, tri, edgeCount, edgeA, edgeB, edgeUse, edgeStart, edgeHalves } = topology;
  const n = welded.count;
  const position = new Float64Array(n * 3);
  for (let v = 0; v < n; v += 1) {
    const r = welded.representative[v] * 3;
    position.set([mesh.positions[r], mesh.positions[r + 1], mesh.positions[r + 2]], v * 3);
  }
  const original = position.slice();
  const faces = faceCount(mesh);
  const faceNormal = new Float32Array(faces * 3);
  const vertexNormal = new Float64Array(n * 3);
  for (let f = 0; f < faces; f += 1) {
    const g = faceGeometry(mesh.positions, mesh.indices, f);
    faceNormal.set(g.normal, f * 3);
    for (let k = 0; k < 3; k += 1) {
      for (let a = 0; a < 3; a += 1) vertexNormal[tri[f * 3 + k] * 3 + a] += g.normal[a] * g.area;
    }
  }
  for (let v = 0; v < n; v += 1) {
    const l = Math.hypot(vertexNormal[v * 3], vertexNormal[v * 3 + 1], vertexNormal[v * 3 + 2]) || 1;
    for (let a = 0; a < 3; a += 1) vertexNormal[v * 3 + a] /= l;
  }

  // 1 moves freely, 0 is pinned.
  const mobility = new Float32Array(n).fill(1);
  const cosFeature = Math.cos((params.featureAngle * Math.PI) / 180);
  const sharpCount = new Uint8Array(n);
  for (let e = 0; e < edgeCount; e += 1) {
    if (edgeUse[e] !== 2) {
      mobility[edgeA[e]] = 0;
      mobility[edgeB[e]] = 0;
      continue;
    }
    const f = ((edgeHalves[edgeStart[e]] / 3) | 0) * 3;
    const g = ((edgeHalves[edgeStart[e] + 1] / 3) | 0) * 3;
    const cos = faceNormal[f] * faceNormal[g] + faceNormal[f + 1] * faceNormal[g + 1] + faceNormal[f + 2] * faceNormal[g + 2];
    if (cos < cosFeature) {
      sharpCount[edgeA[e]] = Math.min(255, sharpCount[edgeA[e]] + 1);
      sharpCount[edgeB[e]] = Math.min(255, sharpCount[edgeB[e]] + 1);
    }
  }
  // One sharp edge is usually noise; two or more form a crease worth keeping.
  for (let v = 0; v < n; v += 1) if (sharpCount[v] >= 2) mobility[v] *= 1 - params.edgePreservation;

  // Neighbour weights fall off with the angle between vertex normals; stronger
  // preservation means a narrower falloff.
  const sigma = 0.9 - 0.75 * params.edgePreservation;
  const weight = new Float32Array(edgeCount);
  for (let e = 0; e < edgeCount; e += 1) {
    const a = edgeA[e] * 3;
    const b = edgeB[e] * 3;
    const cos = vertexNormal[a] * vertexNormal[b] + vertexNormal[a + 1] * vertexNormal[b + 1] + vertexNormal[a + 2] * vertexNormal[b + 2];
    const angle = Math.acos(Math.max(-1, Math.min(1, cos)));
    weight[e] = Math.exp(-((angle / sigma) ** 2) / 2) + 1e-4;
  }

  const sum = new Float64Array(n * 3);
  const total = new Float64Array(n);
  // The shortest edge at a vertex, capped at the mesh's typical edge.
  const typical = medianEdge(mesh);
  const limit = new Float64Array(n).fill(typical);
  for (let e = 0; e < edgeCount; e += 1) {
    const a = edgeA[e];
    const b = edgeB[e];
    const length = Math.hypot(original[a * 3] - original[b * 3], original[a * 3 + 1] - original[b * 3 + 1], original[a * 3 + 2] - original[b * 3 + 2]);
    if (length < limit[a]) limit[a] = length;
    if (length < limit[b]) limit[b] = length;
  }
  for (let v = 0; v < n; v += 1) limit[v] *= params.maxDisplacement;
  const step = (factor) => {
    sum.fill(0);
    total.fill(0);
    for (let e = 0; e < edgeCount; e += 1) {
      const a = edgeA[e];
      const b = edgeB[e];
      const w = weight[e];
      total[a] += w;
      total[b] += w;
      for (let k = 0; k < 3; k += 1) {
        sum[a * 3 + k] += w * position[b * 3 + k];
        sum[b * 3 + k] += w * position[a * 3 + k];
      }
    }
    for (let v = 0; v < n; v += 1) {
      if (!mobility[v] || !total[v]) continue;
      for (let k = 0; k < 3; k += 1) {
        position[v * 3 + k] += factor * mobility[v] * (sum[v * 3 + k] / total[v] - position[v * 3 + k]);
      }
    }
  };
  for (let i = 0; i < params.iterations; i += 1) {
    step(0.5);
    step(-0.53);
  }

  const moved = new Uint8Array(n);
  let modified = 0;
  let largest = 0;
  for (let v = 0; v < n; v += 1) {
    const d = [0, 1, 2].map((k) => position[v * 3 + k] - original[v * 3 + k]);
    let length = Math.hypot(...d);
    if (length > limit[v]) {
      for (let k = 0; k < 3; k += 1) position[v * 3 + k] = original[v * 3 + k] + (d[k] * limit[v]) / length;
      length = limit[v];
    }
    if (length > limit[v] * 0.05) { moved[v] = 1; modified += 1; largest = Math.max(largest, length); }
  }
  for (let v = 0; v < mesh.positions.length / 3; v += 1) {
    const w = welded.map[v];
    if (!moved[w]) continue;
    for (let k = 0; k < 3; k += 1) mesh.positions[v * 3 + k] = position[w * 3 + k];
  }
  for (let f = 0; f < faces; f += 1) {
    if (moved[tri[f * 3]] || moved[tri[f * 3 + 1]] || moved[tri[f * 3 + 2]]) mark(state, f, PROVENANCE.SMOOTHED);
  }
  return { ...params, displacementFactor: params.maxDisplacement, modifiedVertices: modified, maxDisplacement: largest };
}
