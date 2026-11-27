/**
 * Conservative geometry cleanup. None of these steps moves a vertex by more
 * than the welding tolerance, and no vertices are merged across a texture seam,
 * a colour difference or a material boundary.
 */

import { buildTopology, faceCount, vertexCount, weldPositions } from '../meshdata.mjs';
import { faceGeometry } from '../glbmesh.mjs';
import { compactVertices, filterFaces, mark, medianEdge } from './state.mjs';
import { PROVENANCE } from './provenance.mjs';

export function removeInvalidFaces(state) {
  const { mesh } = state;
  const vertices = vertexCount(mesh);
  const p = mesh.positions;
  const faces = faceCount(mesh);
  const keep = new Uint8Array(faces);
  for (let f = 0; f < faces; f += 1) {
    let ok = 1;
    for (let k = 0; k < 3; k += 1) {
      const v = mesh.indices[f * 3 + k];
      if (v >= vertices || !Number.isFinite(p[v * 3]) || !Number.isFinite(p[v * 3 + 1]) || !Number.isFinite(p[v * 3 + 2])) ok = 0;
    }
    keep[f] = ok;
  }
  const removed = filterFaces(state, keep);
  compactVertices(state);
  return removed;
}

/**
 * Merge render vertices that coincide within `tolerance` and agree on UV,
 * colour and primitive.
 */
export function weldDuplicateVertices(state, tolerance = medianEdge(state.mesh) * 1e-3) {
  const { mesh } = state;
  const vertices = vertexCount(mesh);
  const primOf = new Int32Array(vertices).fill(-1);
  for (let f = 0; f < faceCount(mesh); f += 1) {
    for (let k = 0; k < 3; k += 1) primOf[mesh.indices[f * 3 + k]] = mesh.facePrim[f];
  }
  const welded = weldPositions(mesh.positions, tolerance);
  const heads = new Int32Array(welded.count).fill(-1);
  const chain = new Int32Array(vertices).fill(-1);
  const remap = new Uint32Array(vertices);
  const sameAttributes = (a, b) => {
    if (primOf[a] !== primOf[b]) return false;
    if (mesh.uvs && (Math.abs(mesh.uvs[a * 2] - mesh.uvs[b * 2]) > 1e-6 || Math.abs(mesh.uvs[a * 2 + 1] - mesh.uvs[b * 2 + 1]) > 1e-6)) return false;
    if (mesh.colors) {
      for (let k = 0; k < 4; k += 1) if (mesh.colors[a * 4 + k] !== mesh.colors[b * 4 + k]) return false;
    }
    return true;
  };
  let merged = 0;
  const moved = new Uint8Array(vertices);
  for (let v = 0; v < vertices; v += 1) {
    const group = welded.map[v];
    let other = heads[group];
    while (other >= 0 && !sameAttributes(other, v)) other = chain[other];
    if (other < 0) {
      remap[v] = v;
      chain[v] = heads[group];
      heads[group] = v;
    } else {
      remap[v] = other;
      merged += 1;
      if (mesh.positions[v * 3] !== mesh.positions[other * 3]
        || mesh.positions[v * 3 + 1] !== mesh.positions[other * 3 + 1]
        || mesh.positions[v * 3 + 2] !== mesh.positions[other * 3 + 2]) moved[other] = 1;
    }
  }
  // Seam copies of one position must stay bit-identical after welding, or a
  // hairline crack opens between charts.
  const anchor = welded.representative;
  let snapped = 0;
  for (let v = 0; v < vertices; v += 1) {
    const a = anchor[welded.map[v]];
    let differs = false;
    for (let k = 0; k < 3; k += 1) if (mesh.positions[v * 3 + k] !== mesh.positions[a * 3 + k]) differs = true;
    if (!differs) continue;
    for (let k = 0; k < 3; k += 1) mesh.positions[v * 3 + k] = mesh.positions[a * 3 + k];
    moved[v] = 1;
    snapped += 1;
  }
  for (let f = 0; f < faceCount(mesh); f += 1) {
    for (let k = 0; k < 3; k += 1) {
      const v = mesh.indices[f * 3 + k];
      if (moved[v] || moved[remap[v]]) mark(state, f, PROVENANCE.CLEANED);
      mesh.indices[f * 3 + k] = remap[v];
    }
  }
  compactVertices(state);
  return { merged, snapped, tolerance };
}

export function removeDegenerateFaces(state) {
  const { mesh } = state;
  const topology = buildTopology(mesh);
  const floor = (medianEdge(mesh) * 1e-4) ** 2;
  const faces = faceCount(mesh);
  const keep = new Uint8Array(faces);
  for (let f = 0; f < faces; f += 1) {
    keep[f] = !topology.degenerate[f] && faceGeometry(mesh.positions, mesh.indices, f).area > floor ? 1 : 0;
  }
  return filterFaces(state, keep);
}

export function removeDuplicateFaces(state) {
  const { mesh } = state;
  const welded = weldPositions(mesh.positions);
  const faces = faceCount(mesh);
  const keep = new Uint8Array(faces);
  const seen = new Set();
  for (let f = 0; f < faces; f += 1) {
    const key = [0, 1, 2].map((k) => welded.map[mesh.indices[f * 3 + k]]).sort((a, b) => a - b).join(',');
    keep[f] = seen.has(key) ? 0 : 1;
    seen.add(key);
  }
  return filterFaces(state, keep);
}

/**
 * Resolve non-manifold edges only where it is safe: by removing "fins", faces
 * that hang off such an edge with a free edge of their own. Junctions where
 * every face is fully connected are left alone and reported.
 */
export function repairNonManifoldEdges(state, maxPasses = 3) {
  const removed = [];
  let remaining = 0;
  for (let pass = 0; pass < maxPasses; pass += 1) {
    const { mesh } = state;
    const topology = buildTopology(mesh);
    const faces = faceCount(mesh);
    const keep = new Uint8Array(faces).fill(1);
    let changed = 0;
    remaining = 0;
    for (let e = 0; e < topology.edgeCount; e += 1) {
      if (topology.edgeUse[e] <= 2) continue;
      const incident = [];
      for (let i = topology.edgeStart[e]; i < topology.edgeStart[e + 1]; i += 1) {
        incident.push((topology.edgeHalves[i] / 3) | 0);
      }
      const live = incident.filter((f) => keep[f]);
      let excess = live.length - 2;
      if (excess <= 0) continue;
      const fins = live
        .filter((f) => [0, 1, 2].some((k) => {
          const edge = topology.halfEdge[f * 3 + k];
          return edge >= 0 && topology.edgeUse[edge] === 1;
        }))
        .map((f) => ({ f, area: faceGeometry(mesh.positions, mesh.indices, f).area }))
        .sort((a, b) => a.area - b.area);
      for (const fin of fins) {
        if (excess <= 0) break;
        keep[fin.f] = 0;
        excess -= 1;
        changed += 1;
      }
      if (excess > 0) remaining += 1;
    }
    if (!changed) break;
    removed.push(...filterFaces(state, keep));
  }
  return { removed, remaining };
}

/**
 * Make neighbouring faces agree on winding. Within each patch the majority
 * orientation wins, so the fewest faces are flipped.
 */
export function orientFaces(state) {
  const { mesh } = state;
  const topology = buildTopology(mesh);
  const faces = faceCount(mesh);
  const { tri, halfEdge, edgeUse, edgeStart, edgeHalves } = topology;
  const visited = new Uint8Array(faces);
  const flip = new Uint8Array(faces);
  const queue = new Int32Array(faces);
  let flipped = 0;
  for (let seed = 0; seed < faces; seed += 1) {
    if (visited[seed]) continue;
    let head = 0;
    let tail = 0;
    queue[tail] = seed;
    tail += 1;
    visited[seed] = 1;
    while (head < tail) {
      const f = queue[head];
      head += 1;
      for (let k = 0; k < 3; k += 1) {
        const h = f * 3 + k;
        const e = halfEdge[h];
        if (e < 0 || edgeUse[e] !== 2) continue;
        const first = edgeHalves[edgeStart[e]];
        const twin = first === h ? edgeHalves[edgeStart[e] + 1] : first;
        const g = (twin / 3) | 0;
        if (visited[g]) continue;
        visited[g] = 1;
        flip[g] = flip[f] ^ (tri[h] === tri[twin] ? 1 : 0);
        queue[tail] = g;
        tail += 1;
      }
    }
    let count = 0;
    for (let i = 0; i < tail; i += 1) count += flip[queue[i]];
    const invert = count * 2 > tail ? 1 : 0;
    for (let i = 0; i < tail; i += 1) {
      const f = queue[i];
      if ((flip[f] ^ invert) === 0) continue;
      const t = mesh.indices[f * 3 + 1];
      mesh.indices[f * 3 + 1] = mesh.indices[f * 3 + 2];
      mesh.indices[f * 3 + 2] = t;
      mark(state, f, PROVENANCE.CLEANED);
      flipped += 1;
    }
  }
  return flipped;
}
