/**
 * One indexed mesh across every primitive, plus its topology.
 *
 * A textured photogrammetry mesh duplicates a vertex once per UV chart it
 * touches, and spreads faces over one primitive per texture atlas. Neither
 * split is a break in the surface. So topology (edges, boundaries, components)
 * is computed on position-welded vertex ids, while geometry edits keep the
 * original render vertices and therefore their texture coordinates.
 */

import { readFile, writeFile } from 'node:fs/promises';
import { applyMatrix, matrixScale, readMeshGlb, recomputeNormals, writeMeshGlb } from './glbmesh.mjs';

function toRgba(colors, count) {
  const out = new Uint8Array(count * 4).fill(255);
  if (!colors) return out;
  const { values, componentType, stride } = colors;
  const factor = componentType === 5126 ? 255 : componentType === 5123 ? 1 / 257 : 1;
  for (let i = 0; i < count; i += 1) {
    for (let k = 0; k < Math.min(4, stride); k += 1) out[i * 4 + k] = Math.round(values[i * stride + k] * factor);
  }
  return out;
}

/**
 * @returns {{positions:Float32Array, uvs:Float32Array|null, colors:Uint8Array|null,
 *   indices:Uint32Array, facePrim:Uint16Array, prims:object[], images:object[],
 *   rootMatrix:number[]|null, unlit:boolean, passthrough:object[]}}
 */
export function unifyMesh(read) {
  const triangles = read.primitives.filter((p) => p.mode === 4);
  if (!triangles.length) throw new Error('The model contains no triangle geometry.');
  let vertices = 0;
  let indexCount = 0;
  for (const p of triangles) {
    vertices += p.positions.length / 3;
    indexCount += p.indices.length;
  }
  const hasUv = triangles.some((p) => p.uvs);
  const hasColor = triangles.some((p) => p.colors);
  const positions = new Float32Array(vertices * 3);
  const uvs = hasUv ? new Float32Array(vertices * 2) : null;
  const colors = hasColor ? new Uint8Array(vertices * 4) : null;
  const indices = new Uint32Array(indexCount);
  const facePrim = new Uint16Array(indexCount / 3);
  let vertexBase = 0;
  let indexBase = 0;
  const prims = triangles.map((p, slot) => {
    const count = p.positions.length / 3;
    positions.set(p.positions, vertexBase * 3);
    if (uvs && p.uvs) uvs.set(p.uvs, vertexBase * 2);
    if (colors) colors.set(toRgba(p.colors, count), vertexBase * 4);
    for (let i = 0; i < p.indices.length; i += 1) indices[indexBase + i] = p.indices[i] + vertexBase;
    facePrim.fill(slot, indexBase / 3, (indexBase + p.indices.length) / 3);
    vertexBase += count;
    indexBase += p.indices.length;
    return { material: p.material, imageIndex: p.imageIndex, hasUv: Boolean(p.uvs), hasColor: Boolean(p.colors) };
  });
  return {
    positions,
    uvs,
    colors,
    indices,
    facePrim,
    prims,
    images: read.images,
    rootMatrix: read.rootMatrix,
    unlit: read.unlit,
    passthrough: read.primitives.filter((p) => p.mode !== 4),
  };
}

export async function loadMesh(path) {
  return unifyMesh(readMeshGlb(await readFile(path)));
}

export const faceCount = (mesh) => mesh.indices.length / 3;
export const vertexCount = (mesh) => mesh.positions.length / 3;
/** Metres per local unit, carried by the alignment matrix (1 when already metric). */
export const meshScale = (mesh) => matrixScale(mesh.rootMatrix);

/** Positions in the viewer's world frame (Y up, metres). */
export function worldPositions(mesh) {
  const out = Float32Array.from(mesh.positions);
  return mesh.rootMatrix ? applyMatrix(out, mesh.rootMatrix, 1) : out;
}

/**
 * Split back into one primitive per original primitive, compacting vertices.
 *
 * @returns {{primitives: object[], faceOrder: Int32Array}} faceOrder maps each
 *   output face index to the face of `mesh` it came from, which is how per-face
 *   sidecar data stays aligned with the written file.
 */
export function splitMesh(mesh, { normals = null } = {}) {
  const faces = faceCount(mesh);
  const vertexNormals = normals || recomputeNormals(mesh.positions, mesh.indices);
  const faceOrder = new Int32Array(faces);
  const primitives = [];
  let written = 0;
  const remap = new Int32Array(vertexCount(mesh));
  mesh.prims.forEach((prim, slot) => {
    const members = [];
    for (let f = 0; f < faces; f += 1) if (mesh.facePrim[f] === slot) members.push(f);
    if (!members.length) return;
    remap.fill(-1);
    const sources = [];
    const indices = new Uint32Array(members.length * 3);
    members.forEach((f, i) => {
      for (let k = 0; k < 3; k += 1) {
        const v = mesh.indices[f * 3 + k];
        if (remap[v] < 0) { remap[v] = sources.length; sources.push(v); }
        indices[i * 3 + k] = remap[v];
      }
      faceOrder[written] = f;
      written += 1;
    });
    const pick = (source, stride, Type) => {
      const out = new Type(sources.length * stride);
      sources.forEach((v, i) => {
        for (let k = 0; k < stride; k += 1) out[i * stride + k] = source[v * stride + k];
      });
      return out;
    };
    primitives.push({
      mode: 4,
      positions: pick(mesh.positions, 3, Float32Array),
      normals: pick(vertexNormals, 3, Float32Array),
      uvs: prim.hasUv && mesh.uvs ? pick(mesh.uvs, 2, Float32Array) : null,
      colors: prim.hasColor && mesh.colors
        ? { values: pick(mesh.colors, 4, Uint8Array), componentType: 5121, stride: 4 }
        : null,
      indices,
      material: prim.material,
      imageIndex: prim.imageIndex,
    });
  });
  return { primitives, faceOrder };
}

export function encodeMesh(mesh, { extras = null, generator = 'Skyforge' } = {}) {
  const { primitives, faceOrder } = splitMesh(mesh);
  const glb = writeMeshGlb({
    primitives: [...primitives, ...(mesh.passthrough || [])],
    images: mesh.images,
    rootMatrix: mesh.rootMatrix,
    unlit: mesh.unlit,
    generator,
    extras,
  });
  return { glb, faceOrder };
}

export async function saveMesh(path, mesh, options) {
  const { glb, faceOrder } = encodeMesh(mesh, options);
  await writeFile(path, glb);
  return { bytes: glb.length, faceOrder };
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

/**
 * Map every vertex to a topological id shared by all vertices at its position.
 * `tolerance` 0 welds bit-identical positions only.
 */
export function weldPositions(positions, tolerance = 0) {
  const count = positions.length / 3;
  const map = new Uint32Array(count);
  let size = 16;
  while (size < count * 2) size *= 2;
  const mask = size - 1;
  const table = new Int32Array(size).fill(-1);
  const representative = [];
  const bits = new Int32Array(positions.buffer, positions.byteOffset, positions.length);
  const keyOf = tolerance > 0
    ? (i, k) => Math.round(positions[i * 3 + k] / tolerance)
    : (i, k) => bits[i * 3 + k];
  for (let i = 0; i < count; i += 1) {
    const x = keyOf(i, 0);
    const y = keyOf(i, 1);
    const z = keyOf(i, 2);
    let slot = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0 & mask;
    for (;;) {
      const entry = table[slot];
      if (entry < 0) {
        table[slot] = representative.length;
        map[i] = representative.length;
        representative.push(i);
        break;
      }
      const other = representative[entry];
      if (keyOf(other, 0) === x && keyOf(other, 1) === y && keyOf(other, 2) === z) {
        map[i] = entry;
        break;
      }
      slot = (slot + 1) & mask;
    }
  }
  return { map, count: representative.length, representative: Uint32Array.from(representative) };
}

export class UnionFind {
  constructor(size) {
    this.parent = new Int32Array(size);
    for (let i = 0; i < size; i += 1) this.parent[i] = i;
  }

  find(start) {
    const p = this.parent;
    let x = start;
    while (p[x] !== x) { p[x] = p[p[x]]; x = p[x]; }
    return x;
  }

  union(a, b) {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent[ra] = rb;
  }
}

/**
 * Edge and adjacency tables over welded vertices.
 *
 * Half-edge `h = face * 3 + corner` runs from `tri[h]` to the next corner.
 * Faces that collapse to fewer than three distinct welded vertices contribute
 * no edges and are flagged in `degenerate`.
 */
export function buildTopology(mesh, welded = weldPositions(mesh.positions)) {
  const faces = faceCount(mesh);
  const tri = new Uint32Array(faces * 3);
  for (let i = 0; i < tri.length; i += 1) tri[i] = welded.map[mesh.indices[i]];
  const degenerate = new Uint8Array(faces);
  const halfEdge = new Int32Array(faces * 3).fill(-1);

  let size = 16;
  while (size < faces * 4) size *= 2;
  const mask = size - 1;
  const table = new Int32Array(size).fill(-1);
  const edgeA = new Uint32Array(faces * 3);
  const edgeB = new Uint32Array(faces * 3);
  const edgeUse = new Uint32Array(faces * 3);
  let edges = 0;
  for (let f = 0; f < faces; f += 1) {
    const a = tri[f * 3];
    const b = tri[f * 3 + 1];
    const c = tri[f * 3 + 2];
    if (a === b || b === c || a === c) { degenerate[f] = 1; continue; }
    for (let k = 0; k < 3; k += 1) {
      const u = tri[f * 3 + k];
      const v = tri[f * 3 + ((k + 1) % 3)];
      const lo = u < v ? u : v;
      const hi = u < v ? v : u;
      let slot = (Math.imul(lo, 73856093) ^ Math.imul(hi, 19349663)) >>> 0 & mask;
      for (;;) {
        const entry = table[slot];
        if (entry < 0) {
          table[slot] = edges;
          edgeA[edges] = lo;
          edgeB[edges] = hi;
          halfEdge[f * 3 + k] = edges;
          edgeUse[edges] = 1;
          edges += 1;
          break;
        }
        if (edgeA[entry] === lo && edgeB[entry] === hi) {
          halfEdge[f * 3 + k] = entry;
          edgeUse[entry] += 1;
          break;
        }
        slot = (slot + 1) & mask;
      }
    }
  }
  // Compressed list of half-edges per edge.
  const edgeStart = new Uint32Array(edges + 1);
  for (let e = 0; e < edges; e += 1) edgeStart[e + 1] = edgeStart[e] + edgeUse[e];
  const cursor = edgeStart.slice(0, edges);
  const edgeHalves = new Uint32Array(edgeStart[edges]);
  for (let h = 0; h < halfEdge.length; h += 1) {
    const e = halfEdge[h];
    if (e < 0) continue;
    edgeHalves[cursor[e]] = h;
    cursor[e] += 1;
  }
  return {
    welded,
    tri,
    degenerate,
    halfEdge,
    edgeCount: edges,
    edgeA: edgeA.subarray(0, edges),
    edgeB: edgeB.subarray(0, edges),
    edgeUse: edgeUse.subarray(0, edges),
    edgeStart,
    edgeHalves,
  };
}

/** Faces incident to each welded vertex, compressed-sparse-row. */
export function vertexFaces(topology) {
  const { tri, welded } = topology;
  const start = new Uint32Array(welded.count + 1);
  for (let i = 0; i < tri.length; i += 1) start[tri[i] + 1] += 1;
  for (let v = 1; v <= welded.count; v += 1) start[v] += start[v - 1];
  const cursor = start.slice(0, welded.count);
  const list = new Uint32Array(tri.length);
  for (let i = 0; i < tri.length; i += 1) {
    list[cursor[tri[i]]] = (i / 3) | 0;
    cursor[tri[i]] += 1;
  }
  return { start, list };
}

/** Connected components by shared welded vertices. */
export function faceComponents(topology) {
  const { tri, welded } = topology;
  const sets = new UnionFind(welded.count);
  for (let f = 0; f < tri.length; f += 3) {
    sets.union(tri[f], tri[f + 1]);
    sets.union(tri[f + 1], tri[f + 2]);
  }
  const id = new Map();
  const component = new Int32Array(tri.length / 3);
  for (let f = 0; f < component.length; f += 1) {
    const root = sets.find(tri[f * 3]);
    let value = id.get(root);
    if (value === undefined) { value = id.size; id.set(root, value); }
    component[f] = value;
  }
  return { component, count: id.size };
}

/**
 * Closed boundary loops.
 *
 * From a boundary half-edge the successor is found by rotating through the
 * faces around its end vertex, which keeps two holes that merely touch at a
 * vertex from being traced as one.
 *
 * @returns {{loops: Array<{vertices:number[], halves:number[]}>, open: number}}
 *   `vertices` are welded ids in the order the surrounding faces traverse them.
 */
export function boundaryLoops(topology) {
  const { tri, halfEdge, edgeUse, edgeStart, edgeHalves } = topology;
  const isBoundary = (h) => halfEdge[h] >= 0 && edgeUse[halfEdge[h]] === 1;
  const next = (h) => h - (h % 3) + ((h + 1) % 3);
  const successor = (h) => {
    let candidate = next(h);
    for (let guard = 0; guard < 256; guard += 1) {
      if (isBoundary(candidate)) return candidate;
      const e = halfEdge[candidate];
      if (e < 0 || edgeUse[e] !== 2) return -1;
      const first = edgeHalves[edgeStart[e]];
      const twin = first === candidate ? edgeHalves[edgeStart[e] + 1] : first;
      // A consistently oriented neighbour traverses the shared edge backwards.
      if (tri[twin] === tri[candidate]) return -1;
      candidate = next(twin);
    }
    return -1;
  };
  const seen = new Uint8Array(halfEdge.length);
  const loops = [];
  let open = 0;
  for (let h = 0; h < halfEdge.length; h += 1) {
    if (seen[h] || !isBoundary(h)) continue;
    const halves = [];
    let current = h;
    while (current >= 0 && !seen[current]) {
      seen[current] = 1;
      halves.push(current);
      current = successor(current);
    }
    if (current !== h) { open += 1; continue; }
    loops.push({ halves, vertices: halves.map((x) => tri[x]) });
  }
  return { loops, open };
}
