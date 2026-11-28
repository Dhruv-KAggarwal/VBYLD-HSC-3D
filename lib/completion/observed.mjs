/**
 * Which faces of a reconstruction were actually photographed.
 *
 * Where the drone never looked, the mesher still closes the surface: it stretches
 * large triangles from a roof edge down to the ground, and the texture stage
 * leaves them flat grey or smears a few grazing-angle pixels across them. Those
 * are the faces that make the back of a building look broken.
 */

import { DepthRasterizer, visibleAt } from '../confidence/visibility.mjs';
import { decodeImage, sampleImage } from '../image.mjs';
import { buildTopology, faceComponents } from '../meshdata.mjs';

const MAX_VIEWS = 100;
/** Below this cosine (about 76 degrees off the normal) a photo only smears. */
const MIN_COSINE = 0.24;

/** Decoded base-colour image per primitive slot, null where there is none. */
export function decodeTextures(mesh) {
  const cache = new Map();
  return mesh.prims.map((prim) => {
    if (prim.imageIndex == null || !prim.hasUv) return null;
    if (!cache.has(prim.imageIndex)) cache.set(prim.imageIndex, decodeImage(mesh.images[prim.imageIndex].bytes));
    return cache.get(prim.imageIndex);
  });
}

/** Surface colour at barycentric weights (a, b, c) of a face, or null. */
export function faceColor(mesh, textures, face, a, b, c, out = [0, 0, 0]) {
  const i = mesh.indices[face * 3];
  const j = mesh.indices[face * 3 + 1];
  const k = mesh.indices[face * 3 + 2];
  const texture = textures[mesh.facePrim[face]];
  if (texture) {
    const u = a * mesh.uvs[i * 2] + b * mesh.uvs[j * 2] + c * mesh.uvs[k * 2];
    const v = a * mesh.uvs[i * 2 + 1] + b * mesh.uvs[j * 2 + 1] + c * mesh.uvs[k * 2 + 1];
    return sampleImage(texture, u, v, out);
  }
  if (!mesh.colors) return null;
  for (let n = 0; n < 3; n += 1) {
    out[n] = a * mesh.colors[i * 4 + n] + b * mesh.colors[j * 4 + n] + c * mesh.colors[k * 4 + n];
  }
  return out;
}

const PROBES = [[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]];

/**
 * TextureMesh paints faces no camera covers a flat grey (`--empty-color`
 * 0x808080), but the atlas stores it a few levels off, by an amount that differs
 * between bakes. So the exact shade is read from the model: the one near-grey
 * value that a noticeable number of faces share precisely.
 */
function emptyColours(mesh, textures) {
  const counts = new Map();
  const colour = [0, 0, 0];
  const faces = mesh.indices.length / 3;
  for (let f = 0; f < faces; f += 1) {
    if (!textures[mesh.facePrim[f]]) continue;
    faceColor(mesh, textures, f, 1 / 3, 1 / 3, 1 / 3, colour);
    if (!colour.every((v) => v >= 118 && v <= 138)) continue;
    const key = colour.map(Math.round).join(',');
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  const shared = [...counts].filter(([, n]) => n > Math.max(50, faces * 0.002)).map(([key]) => key.split(',').map(Number));
  return [[128, 128, 128], ...shared];
}

function isUntextured(mesh, textures, face, empty) {
  if (!textures[mesh.facePrim[face]]) return false;
  const colour = [0, 0, 0];
  return empty.some((grey) => PROBES.every(([a, b, c]) => {
    faceColor(mesh, textures, face, a, b, c, colour);
    return colour.every((v, k) => Math.abs(v - grey[k]) <= 1.5);
  }));
}

/** Largest |cos| between each face normal and any camera that can see the face. */
function bestViewingCosine(mesh, views, onProgress) {
  const { positions, indices } = mesh;
  const faces = indices.length / 3;
  const best = new Float32Array(faces);
  const centre = new Float32Array(faces * 3);
  const normal = new Float32Array(faces * 3);
  for (let f = 0; f < faces; f += 1) {
    const a = indices[f * 3] * 3;
    const b = indices[f * 3 + 1] * 3;
    const c = indices[f * 3 + 2] * 3;
    for (let k = 0; k < 3; k += 1) centre[f * 3 + k] = (positions[a + k] + positions[b + k] + positions[c + k]) / 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const length = Math.hypot(nx, ny, nz) || 1;
    normal[f * 3] = nx / length;
    normal[f * 3 + 1] = ny / length;
    normal[f * 3 + 2] = nz / length;
  }

  const step = Math.max(1, Math.ceil(views.length / MAX_VIEWS));
  const used = views.filter((_, i) => i % step === 0);
  const raster = new DepthRasterizer(positions, indices, 640);
  used.forEach((view, n) => {
    const frame = raster.render(view);
    const [r0, r1, r2] = view.R;
    const [cx, cy, cz] = view.center;
    for (let f = 0; f < faces; f += 1) {
      const dx = centre[f * 3] - cx;
      const dy = centre[f * 3 + 1] - cy;
      const dz = centre[f * 3 + 2] - cz;
      const z = r2[0] * dx + r2[1] * dy + r2[2] * dz;
      if (!(z > 1e-6)) continue;
      const x = ((view.fx * (r0[0] * dx + r0[1] * dy + r0[2] * dz)) / z + view.cx) * frame.scale;
      const y = ((view.fy * (r1[0] * dx + r1[1] * dy + r1[2] * dz)) / z + view.cy) * frame.scale;
      if (x < 0 || y < 0 || x >= frame.width || y >= frame.height) continue;
      const cosine = Math.abs(normal[f * 3] * dx + normal[f * 3 + 1] * dy + normal[f * 3 + 2] * dz)
        / (Math.hypot(dx, dy, dz) || 1);
      if (cosine <= best[f] || !visibleAt(frame, x, y, z)) continue;
      best[f] = cosine;
    }
    onProgress((n + 1) / used.length);
  });
  return best;
}

/**
 * @param {object} args
 * @param {object} args.mesh      unified mesh in its local frame
 * @param {Array}  args.views     solved cameras in the same frame
 * @param {Float32Array} args.world  vertex positions in metres, Y up
 * @param {(face:number)=>boolean} args.inRegion  faces outside are never removed
 * @returns {{keep: Uint8Array, untextured: number, unseen: number, grazing: number, fragments: number}}
 */
export function classifyFaces({
  mesh, textures, views, world, inRegion, minFragmentM2 = 6, onProgress = () => {},
}) {
  const best = bestViewingCosine(mesh, views, onProgress);
  const faces = mesh.indices.length / 3;
  const keep = new Uint8Array(faces).fill(1);
  const counts = { untextured: 0, unseen: 0, grazing: 0, fragments: 0 };
  const empty = emptyColours(mesh, textures);
  for (let f = 0; f < faces; f += 1) {
    if (!inRegion(f)) continue;
    let reason = null;
    if (isUntextured(mesh, textures, f, empty)) reason = 'untextured';
    else if (best[f] === 0) reason = 'unseen';
    else if (best[f] < MIN_COSINE) reason = 'grazing';
    if (reason) { keep[f] = 0; counts[reason] += 1; }
  }

  // Cutting those faces out leaves crumbs floating where they used to hang.
  const kept = [];
  for (let f = 0; f < faces; f += 1) if (keep[f]) kept.push(f);
  const indices = new Uint32Array(kept.length * 3);
  kept.forEach((f, i) => indices.set(mesh.indices.subarray(f * 3, f * 3 + 3), i * 3));
  const parts = faceComponents(buildTopology({ positions: mesh.positions, indices }));
  const area = new Float64Array(parts.count);
  kept.forEach((f, i) => {
    const [a, b, c] = [0, 1, 2].map((k) => mesh.indices[f * 3 + k] * 3);
    const u = [0, 1, 2].map((k) => world[b + k] - world[a + k]);
    const v = [0, 1, 2].map((k) => world[c + k] - world[a + k]);
    area[parts.component[i]] += Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]) / 2;
  });
  kept.forEach((f, i) => {
    if (area[parts.component[i]] < minFragmentM2 && inRegion(f)) { keep[f] = 0; counts.fragments += 1; }
  });
  return { keep, ...counts };
}
