/**
 * Mesh simplification with meshoptimizer's quadric edge collapse.
 *
 * The simplifier is the one three.js ships (meshoptimizer, MIT), so there is no
 * new dependency. It only produces new index buffers over the existing vertices,
 * so positions, texture coordinates and colours of surviving vertices are
 * exactly the originals.
 */

import { MeshoptSimplifier } from 'three/addons/libs/meshopt_simplifier.module.js';

export async function simplifierReady() {
  if (!MeshoptSimplifier.supported) throw new Error('WebAssembly is unavailable, so meshes cannot be simplified.');
  await MeshoptSimplifier.ready;
}

/** Weight of one unit of attribute change relative to geometric error. */
const UV_WEIGHT = 0.8;
const COLOR_WEIGHT = 0.4;

/** Interleaved UV or colour attributes for the attribute-aware metric. */
export function vertexAttributes(mesh) {
  const count = mesh.positions.length / 3;
  if (mesh.uvs && mesh.images.length) {
    return { data: mesh.uvs, stride: 2, weights: [UV_WEIGHT, UV_WEIGHT] };
  }
  if (mesh.colors) {
    const data = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      for (let k = 0; k < 3; k += 1) data[i * 3 + k] = mesh.colors[i * 4 + k] / 255;
    }
    return { data, stride: 3, weights: [COLOR_WEIGHT, COLOR_WEIGHT, COLOR_WEIGHT] };
  }
  return null;
}

/**
 * Collapse edges while keeping texture seams, borders and material boundaries.
 *
 * @param {number} errorLimit largest allowed error as a fraction of mesh extent
 * @returns {{indices:Uint32Array, error:number}}
 */
export function simplifyPreserving(mesh, indices, targetFaces, errorLimit, { prune = false } = {}) {
  const target = Math.max(3, Math.min(indices.length, Math.floor(targetFaces) * 3));
  const flags = prune ? ['Prune'] : [];
  const attributes = vertexAttributes(mesh);
  const [result, error] = attributes
    ? MeshoptSimplifier.simplifyWithAttributes(
      indices, mesh.positions, 3, attributes.data, attributes.stride, attributes.weights, null, target, errorLimit, flags,
    )
    : MeshoptSimplifier.simplify(indices, mesh.positions, 3, target, errorLimit, flags);
  return { indices: Uint32Array.from(result), error };
}

/**
 * Geometry-only collapse on position-welded indices, ignoring texture seams.
 *
 * Only the referenced vertices are handed to the simplifier: unused seam copies
 * sharing their positions would still make it treat each one as a seam.
 */
export function simplifyGeometry(positions, indices, targetFaces, errorLimit, { prune = false } = {}) {
  const remap = new Int32Array(positions.length / 3).fill(-1);
  const used = [];
  const compact = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    const v = indices[i];
    if (remap[v] < 0) { remap[v] = used.length; used.push(v); }
    compact[i] = remap[v];
  }
  const points = new Float32Array(used.length * 3);
  used.forEach((v, i) => points.set(positions.subarray(v * 3, v * 3 + 3), i * 3));
  const target = Math.max(3, Math.min(indices.length, Math.floor(targetFaces) * 3));
  const [result, error] = MeshoptSimplifier.simplify(compact, points, 3, target, errorLimit, prune ? ['Prune'] : []);
  return { indices: Uint32Array.from(result, (v) => used[v]), error };
}
