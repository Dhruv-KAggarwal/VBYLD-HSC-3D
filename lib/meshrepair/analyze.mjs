/**
 * Mesh quality analysis. Every number here is counted or measured on the mesh
 * passed in; anything estimated from a sample says so and reports the sample.
 */

import { MeshBVH } from '../bvh.mjs';
import { boundaryLoops, buildTopology, faceComponents, faceCount, vertexCount, vertexFaces, weldPositions } from '../meshdata.mjs';
import { faceGeometry } from '../glbmesh.mjs';
import { boundingDiagonal, createState, medianEdge, supported } from './state.mjs';
import { removeInvalidFaces } from './cleanup.mjs';

const THIN_QUALITY = 0.05;

/** 4*sqrt(3)*area / sum of squared edges: 1 for equilateral, 0 for a sliver. */
function quality(p, a, b, c, area) {
  const sq = (i, j) => (p[i] - p[j]) ** 2 + (p[i + 1] - p[j + 1]) ** 2 + (p[i + 2] - p[j + 2]) ** 2;
  const sum = sq(a, b) + sq(b, c) + sq(c, a);
  return sum > 0 ? (6.928203 * area) / sum : 0;
}

function segmentHitsTriangle(p, s, e, a, b, c) {
  const dx = p[e] - p[s];
  const dy = p[e + 1] - p[s + 1];
  const dz = p[e + 2] - p[s + 2];
  const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
  const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
  const h = [dy * e2[2] - dz * e2[1], dz * e2[0] - dx * e2[2], dx * e2[1] - dy * e2[0]];
  const det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
  if (Math.abs(det) < 1e-18) return false;
  const t0 = [p[s] - p[a], p[s + 1] - p[a + 1], p[s + 2] - p[a + 2]];
  const u = (t0[0] * h[0] + t0[1] * h[1] + t0[2] * h[2]) / det;
  if (u < 1e-6 || u > 1 - 1e-6) return false;
  const q = [t0[1] * e1[2] - t0[2] * e1[1], t0[2] * e1[0] - t0[0] * e1[2], t0[0] * e1[1] - t0[1] * e1[0]];
  const v = (dx * q[0] + dy * q[1] + dz * q[2]) / det;
  if (v < 1e-6 || u + v > 1 - 1e-6) return false;
  const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
  return t > 1e-6 && t < 1 - 1e-6;
}

function sampleSelfIntersections(mesh, topology, samples) {
  const faces = faceCount(mesh);
  const bvh = new MeshBVH(mesh.positions, mesh.indices);
  const p = mesh.positions;
  const { tri } = topology;
  const step = Math.max(1, Math.floor(faces / samples));
  let tested = 0;
  let intersecting = 0;
  for (let f = 0; f < faces; f += step) {
    if (topology.degenerate[f]) continue;
    tested += 1;
    const own = [mesh.indices[f * 3] * 3, mesh.indices[f * 3 + 1] * 3, mesh.indices[f * 3 + 2] * 3];
    const lo = [0, 1, 2].map((k) => Math.min(p[own[0] + k], p[own[1] + k], p[own[2] + k]));
    const hi = [0, 1, 2].map((k) => Math.max(p[own[0] + k], p[own[1] + k], p[own[2] + k]));
    let hit = false;
    bvh.forEachInBox(lo, hi, (g) => {
      if (hit || g === f) return;
      for (let i = 0; i < 3; i += 1) {
        for (let j = 0; j < 3; j += 1) if (tri[f * 3 + i] === tri[g * 3 + j]) return;
      }
      const other = [mesh.indices[g * 3] * 3, mesh.indices[g * 3 + 1] * 3, mesh.indices[g * 3 + 2] * 3];
      for (let k = 0; k < 3 && !hit; k += 1) {
        hit = segmentHitsTriangle(p, own[k], own[(k + 1) % 3], ...other)
          || segmentHitsTriangle(p, other[k], other[(k + 1) % 3], ...own);
      }
    });
    if (hit) intersecting += 1;
  }
  return {
    method: 'edge-triangle test on a regular sample of faces',
    sampledFaces: tested,
    intersectingFaces: intersecting,
    estimatedFraction: tested ? Math.round((intersecting / tested) * 10000) / 10000 : 0,
  };
}

export function countNonManifoldVertices(topology) {
  const { tri, welded } = topology;
  const { start, list } = vertexFaces(topology);
  let count = 0;
  for (let v = 0; v < welded.count; v += 1) {
    const n = start[v + 1] - start[v];
    if (n < 2) continue;
    // Faces around v belong to one fan if they chain through shared edges at v.
    const group = Array.from({ length: n }, (_, i) => i);
    const find = (x) => { while (group[x] !== x) { group[x] = group[group[x]]; x = group[x]; } return x; };
    const neighbour = new Map();
    for (let i = 0; i < n; i += 1) {
      const f = list[start[v] + i];
      if (topology.degenerate[f]) { group[i] = -1; continue; }
      for (let k = 0; k < 3; k += 1) {
        const u = tri[f * 3 + k];
        if (u === v) continue;
        if (neighbour.has(u)) group[find(i)] = find(neighbour.get(u));
        else neighbour.set(u, i);
      }
    }
    const roots = new Set();
    for (let i = 0; i < n; i += 1) if (group[i] !== -1) roots.add(find(i));
    if (roots.size > 1) count += 1;
  }
  return count;
}

/**
 * @param {object} context  from enhance/context.mjs (for scale and the cloud)
 * @param {object} mesh     the mesh to analyse
 */
export async function analyzeMesh(context, input, { selfIntersectionSamples = 3000, onProgress = null } = {}) {
  const started = Date.now();
  let mesh = input;
  const vertices = vertexCount(mesh);
  const p = mesh.positions;

  let invalidIndices = 0;
  let invalidCoordinates = 0;
  for (let v = 0; v < vertices; v += 1) {
    if (!Number.isFinite(p[v * 3]) || !Number.isFinite(p[v * 3 + 1]) || !Number.isFinite(p[v * 3 + 2])) invalidCoordinates += 1;
  }
  const used = new Uint8Array(vertices);
  for (let i = 0; i < mesh.indices.length; i += 1) {
    if (mesh.indices[i] >= vertices) invalidIndices += 1;
    else used[mesh.indices[i]] = 1;
  }
  if (invalidIndices || invalidCoordinates) {
    // Measure everything else on a copy without the unreadable faces.
    const copy = createState({ ...mesh, positions: mesh.positions, indices: mesh.indices.slice(), facePrim: mesh.facePrim.slice(), uvs: mesh.uvs?.slice() || null, colors: mesh.colors?.slice() || null });
    removeInvalidFaces(copy);
    mesh = copy.mesh;
  }
  const faces = faceCount(mesh);
  let isolatedVertices = 0;
  for (let v = 0; v < vertices; v += 1) if (!used[v]) isolatedVertices += 1;

  // Isolated vertices were counted on the input; the rest uses the clean mesh.
  const q = mesh.positions;
  const welded = weldPositions(q);
  const topology = buildTopology(mesh, welded);
  onProgress?.(0.2);

  // Same position, UV and colour: a true duplicate rather than a seam split.
  const seen = new Set();
  let duplicateVertices = 0;
  for (let v = 0; v < vertexCount(mesh); v += 1) {
    const uv = mesh.uvs ? `${mesh.uvs[v * 2]},${mesh.uvs[v * 2 + 1]}` : '';
    const color = mesh.colors ? `${mesh.colors[v * 4]},${mesh.colors[v * 4 + 1]},${mesh.colors[v * 4 + 2]}` : '';
    const key = `${welded.map[v]}|${uv}|${color}`;
    if (seen.has(key)) duplicateVertices += 1;
    else seen.add(key);
  }
  seen.clear();

  const edgeLength = medianEdge(mesh);
  const areaFloor = (edgeLength * 1e-4) ** 2;
  const normals = new Float32Array(faces * 3);
  const faceKeys = new Set();
  let degenerateFaces = 0;
  let duplicateFaces = 0;
  let thinTriangles = 0;
  let area = 0;
  let againstStored = 0;
  let comparedStored = 0;
  for (let f = 0; f < faces; f += 1) {
    const g = faceGeometry(q, mesh.indices, f);
    normals.set(g.normal, f * 3);
    if (Number.isFinite(g.area)) area += g.area;
    if (topology.degenerate[f] || !(g.area > areaFloor)) { degenerateFaces += 1; continue; }
    const key = [topology.tri[f * 3], topology.tri[f * 3 + 1], topology.tri[f * 3 + 2]].sort((a, b) => a - b).join(',');
    if (faceKeys.has(key)) duplicateFaces += 1;
    else faceKeys.add(key);
    const a = mesh.indices[f * 3] * 3;
    if (quality(q, a, mesh.indices[f * 3 + 1] * 3, mesh.indices[f * 3 + 2] * 3, g.area) < THIN_QUALITY) thinTriangles += 1;
    if (context.storedNormals) {
      const n = context.storedNormals;
      comparedStored += 1;
      if (n[a] * g.normal[0] + n[a + 1] * g.normal[1] + n[a + 2] * g.normal[2] < 0) againstStored += 1;
    }
  }
  faceKeys.clear();
  onProgress?.(0.4);

  let boundaryEdges = 0;
  let nonManifoldEdges = 0;
  let inconsistentEdges = 0;
  const dihedrals = [];
  const stride = Math.max(1, Math.floor(topology.edgeCount / 60000));
  for (let e = 0; e < topology.edgeCount; e += 1) {
    const n = topology.edgeUse[e];
    if (n === 1) boundaryEdges += 1;
    else if (n > 2) nonManifoldEdges += 1;
    else {
      const h0 = topology.edgeHalves[topology.edgeStart[e]];
      const h1 = topology.edgeHalves[topology.edgeStart[e] + 1];
      // Consistent neighbours run along their shared edge in opposite directions.
      const same = topology.tri[h0] === topology.tri[h1];
      if (same) inconsistentEdges += 1;
      if (e % stride === 0) {
        const a = ((h0 / 3) | 0) * 3;
        const b = ((h1 / 3) | 0) * 3;
        let cos = normals[a] * normals[b] + normals[a + 1] * normals[b + 1] + normals[a + 2] * normals[b + 2];
        if (same) cos = -cos;
        dihedrals.push((Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI);
      }
    }
  }
  dihedrals.sort((a, b) => a - b);

  const { loops, open } = boundaryLoops(topology);
  const components = faceComponents(topology);
  const sizes = new Int32Array(components.count);
  for (let f = 0; f < faces; f += 1) sizes[components.component[f]] += 1;
  let largest = 0;
  for (const s of sizes) if (s > largest) largest = s;
  let floatingComponents = 0;
  for (const s of sizes) if (s < largest * 0.01) floatingComponents += 1;
  const nonManifoldVertices = countNonManifoldVertices(topology);
  onProgress?.(0.65);

  const selfIntersections = selfIntersectionSamples > 0
    ? sampleSelfIntersections(mesh, topology, selfIntersectionSamples) : null;
  onProgress?.(0.85);

  const metres = context.metresPerUnit;
  const cloud = await context.cloud();
  let unsupported = null;
  if (cloud) {
    const step = Math.max(1, Math.floor(faces / 40000));
    let sampledArea = 0;
    let bare = 0;
    let count = 0;
    let bareCount = 0;
    for (let f = 0; f < faces; f += step) {
      const g = faceGeometry(q, mesh.indices, f);
      const radius = Math.max(4 * cloud.spacing, Math.sqrt(g.area) * 1.5);
      count += 1;
      sampledArea += g.area;
      if (!supported(cloud, ...g.centroid, radius)) { bare += g.area; bareCount += 1; }
    }
    unsupported = {
      method: 'faces with no dense point within max(4 x cloud spacing, 1.5 x face size)',
      sampledFaces: count,
      unsupportedFaces: bareCount,
      unsupportedAreaPercent: sampledArea ? Math.round((bare / sampledArea) * 10000) / 100 : 0,
    };
  }

  return {
    vertices,
    triangles: faceCount(input),
    topologicalVertices: welded.count,
    connectedComponents: components.count,
    floatingComponents,
    degenerateFaces,
    duplicateVertices,
    duplicateFaces,
    isolatedVertices,
    invalidIndices,
    invalidCoordinates,
    boundaryEdges,
    boundaryLoops: loops.length,
    openBoundaryChains: open,
    nonManifoldEdges,
    nonManifoldVertices,
    inconsistentOrientationEdges: inconsistentEdges,
    normalsAgainstWinding: comparedStored ? againstStored : null,
    thinTriangles,
    selfIntersections,
    surfaceNoise: {
      method: 'angle between adjacent face normals, sampled',
      medianDegrees: dihedrals.length ? Math.round(dihedrals[dihedrals.length >> 1] * 10) / 10 : null,
      p90Degrees: dihedrals.length ? Math.round(dihedrals[Math.floor(dihedrals.length * 0.9)] * 10) / 10 : null,
    },
    unsupported,
    surfaceAreaM2: Math.round(area * metres * metres * 100) / 100,
    boundingDiagonalM: Math.round(boundingDiagonal(mesh) * metres * 100) / 100,
    medianEdgeM: Math.round(edgeLength * metres * 10000) / 10000,
    scaleReliable: context.scaleReliable,
    scaleSource: context.scaleSource,
    analysisMs: Date.now() - started,
  };
}
