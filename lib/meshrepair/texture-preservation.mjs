/**
 * Appearance of generated geometry, and checks that existing textures survived.
 *
 * Original faces keep their vertices, UVs and images untouched. Generated faces
 * never receive invented detail: they take colour from a source photograph only
 * when dense points support the patch AND a camera demonstrably sees every one
 * of its vertices; otherwise they stay a flat neutral grey.
 */

import { readFile } from 'node:fs/promises';
import { MeshBVH } from '../bvh.mjs';
import { decodeImage, sampleImage } from '../image.mjs';
import { readMeshGlb } from '../glbmesh.mjs';

export const NEUTRAL = Object.freeze([148, 148, 148]);
const MIN_SUPPORT = 0.3;
const MIN_INCIDENCE = 0.35;

export function project(view, p) {
  const dx = p[0] - view.center[0];
  const dy = p[1] - view.center[1];
  const dz = p[2] - view.center[2];
  const R = view.R;
  const z = R[2][0] * dx + R[2][1] * dy + R[2][2] * dz;
  if (z <= 1e-9) return null;
  const x = (view.fx * (R[0][0] * dx + R[0][1] * dy + R[0][2] * dz)) / z + view.cx;
  const y = (view.fy * (R[1][0] * dx + R[1][1] * dy + R[1][2] * dz)) / z + view.cy;
  if (x < 1 || y < 1 || x > view.width - 2 || y > view.height - 2) return null;
  return [x, y, z];
}

/**
 * @param {Array<{points:number[][], normal:number[], support:number|null}>} patches
 * @returns {Promise<Array<{colors:number[][], appearance:'reprojected'|'neutral', camera:string|null}>>}
 */
export async function appearanceForPatches(context, patches) {
  const neutral = (patch) => ({ colors: patch.points.map(() => [...NEUTRAL]), appearance: 'neutral', camera: null });
  if (!patches.length) return [];
  const views = await context.views();
  if (!views.length) return patches.map(neutral);
  const { mesh } = context;
  const bvh = new MeshBVH(mesh.positions, mesh.indices);

  const chosen = patches.map((patch) => {
    if (patch.support == null || patch.support < MIN_SUPPORT) return null;
    const centre = [0, 1, 2].map((k) => patch.points.reduce((s, p) => s + p[k], 0) / patch.points.length);
    const candidates = [];
    for (const view of views) {
      const toCamera = [0, 1, 2].map((k) => view.center[k] - centre[k]);
      const distance = Math.hypot(...toCamera) || 1;
      const incidence = (toCamera[0] * patch.normal[0] + toCamera[1] * patch.normal[1] + toCamera[2] * patch.normal[2]) / distance;
      if (incidence < MIN_INCIDENCE) continue;
      if (!patch.points.every((p) => project(view, p))) continue;
      candidates.push({ view, score: incidence / distance });
    }
    candidates.sort((a, b) => b.score - a.score);
    for (const { view } of candidates.slice(0, 6)) {
      const visible = patch.points.every((p) => {
        const dir = [0, 1, 2].map((k) => view.center[k] - p[k]);
        const distance = Math.hypot(...dir);
        const unit = dir.map((d) => d / distance);
        const eps = distance * 2e-3;
        const hit = bvh.raycast(p[0] + unit[0] * eps, p[1] + unit[1] * eps, p[2] + unit[2] * eps, ...unit, distance - eps * 2);
        return !hit;
      });
      if (visible) return view;
    }
    return null;
  });

  // Decode each needed photograph once.
  const results = patches.map(neutral);
  const byView = new Map();
  chosen.forEach((view, i) => {
    if (!view) return;
    if (!byView.has(view)) byView.set(view, []);
    byView.get(view).push(i);
  });
  for (const [view, members] of byView) {
    let image;
    try { image = decodeImage(await readFile(view.path)); } catch { continue; }
    for (const i of members) {
      const colors = patches[i].points.map((p) => {
        const [x, y] = project(view, p);
        return sampleImage(image, (x + 0.5) / view.width, (y + 0.5) / view.height).map(Math.round);
      });
      results[i] = { colors, appearance: 'reprojected', camera: view.name };
    }
  }
  return results;
}

/**
 * Replace textures with per-vertex colour (the "preserve textures: off" path),
 * which allows welding across former seams.
 */
export function bakeVertexColors(mesh) {
  const vertices = mesh.positions.length / 3;
  const colors = mesh.colors || new Uint8Array(vertices * 4).fill(255);
  const primOf = new Int32Array(vertices).fill(-1);
  for (let f = 0; f < mesh.indices.length / 3; f += 1) {
    for (let k = 0; k < 3; k += 1) primOf[mesh.indices[f * 3 + k]] = mesh.facePrim[f];
  }
  const images = new Map();
  const sample = [0, 0, 0];
  for (let v = 0; v < vertices; v += 1) {
    const prim = mesh.prims[primOf[v]];
    if (!prim || prim.imageIndex == null || !prim.hasUv) continue;
    if (!images.has(prim.imageIndex)) images.set(prim.imageIndex, decodeImage(mesh.images[prim.imageIndex].bytes));
    sampleImage(images.get(prim.imageIndex), mesh.uvs[v * 2], mesh.uvs[v * 2 + 1], sample);
    colors.set([sample[0], sample[1], sample[2], 255], v * 4);
  }
  mesh.colors = colors;
  mesh.uvs = null;
  mesh.images = [];
  mesh.facePrim.fill(0);
  mesh.prims = [{ material: { name: 'SkyforgeVertexColour' }, imageIndex: null, hasUv: false, hasColor: true }];
}

/**
 * Read the written file back and confirm that texture coordinates, images and
 * material references are intact.
 */
export function validateTextures(originalMesh, glb) {
  const read = readMeshGlb(glb);
  const issues = [];
  let outOfRange = 0;
  let checked = 0;
  for (const primitive of read.primitives) {
    if (primitive.mode !== 4) continue;
    if (primitive.imageIndex != null && !primitive.uvs) issues.push('A textured primitive has no texture coordinates.');
    if (!primitive.uvs) continue;
    for (const value of primitive.uvs) {
      checked += 1;
      if (!Number.isFinite(value) || value < -0.01 || value > 1.01) outOfRange += 1;
    }
  }
  const before = originalMesh.images.reduce((sum, image) => sum + image.bytes.length, 0);
  const after = read.images.reduce((sum, image) => sum + image.bytes.length, 0);
  const preserved = read.images.length === originalMesh.images.length && before === after;
  if (!preserved) issues.push('Texture images differ from the original.');
  if (outOfRange) issues.push(`${outOfRange} texture coordinates fall outside the atlas.`);
  return {
    valid: !issues.length,
    issues,
    texturedOriginally: originalMesh.images.length > 0,
    images: read.images.length,
    imagesByteIdentical: preserved,
    uvValuesChecked: checked,
    uvOutOfRange: outOfRange,
  };
}
