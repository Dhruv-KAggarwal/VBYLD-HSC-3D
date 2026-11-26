/**
 * Transfer the dense point cloud's photographic colour onto a mesh's vertices.
 * ---------------------------------------------------------------------------
 * OpenMVS's TextureMesh atlas fails on a single forward aerial pass: most faces
 * see no good view and come back as flat "empty-colour", so the model renders
 * blank. But the dense point cloud OpenMVS produced *does* carry a real colour
 * per point (sampled straight from the drone frames). This walks every mesh
 * vertex, finds the nearest coloured cloud point through a voxel hash, and bakes
 * that colour onto the vertex — giving a dense, genuinely photographic,
 * robustly-coloured 3-D mesh without depending on the atlas at all.
 *
 *   node tools/colorize-mesh.mjs <mesh.ply> <cloud.ply> --out=colored.ply
 */

import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parsePly, writeBinaryPly } from '../lib/ply.mjs';

const args = process.argv.slice(2);
const meshPath = resolve(args[0]);
const cloudPath = resolve(args[1]);
const out = resolve((args.find((a) => a.startsWith('--out=')) || '--out=output/colored-mesh.ply').split('=')[1]);

const mesh = parsePly(await readFile(meshPath));
const cloud = parsePly(await readFile(cloudPath));
if (!cloud.colors) throw new Error('Cloud has no colours to transfer.');

const cp = cloud.positions;
const cc = cloud.colors;
const pointCount = cp.length / 3;

// Build a voxel hash of the cloud. Cell size ~= mean point spacing so each cell
// holds a handful of points and a 3x3x3 neighbourhood almost always contains the
// true nearest neighbour.
const box = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
for (let i = 0; i < pointCount; i += 1) {
  for (let k = 0; k < 3; k += 1) {
    box.min[k] = Math.min(box.min[k], cp[i * 3 + k]);
    box.max[k] = Math.max(box.max[k], cp[i * 3 + k]);
  }
}
const diag = Math.hypot(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]);
const cell = Math.max(1e-4, diag / Math.cbrt(pointCount) * 1.5);
const key = (x, y, z) => `${Math.floor((x - box.min[0]) / cell)},${Math.floor((y - box.min[1]) / cell)},${Math.floor((z - box.min[2]) / cell)}`;
const grid = new Map();
for (let i = 0; i < pointCount; i += 1) {
  const k = key(cp[i * 3], cp[i * 3 + 1], cp[i * 3 + 2]);
  let bucket = grid.get(k);
  if (!bucket) { bucket = []; grid.set(k, bucket); }
  bucket.push(i);
}

const vertexCount = mesh.positions.length / 3;
const colors = new Uint8Array(vertexCount * 3);
let matched = 0;
for (let v = 0; v < vertexCount; v += 1) {
  const x = mesh.positions[v * 3];
  const y = mesh.positions[v * 3 + 1];
  const z = mesh.positions[v * 3 + 2];
  const cx = Math.floor((x - box.min[0]) / cell);
  const cy = Math.floor((y - box.min[1]) / cell);
  const cz = Math.floor((z - box.min[2]) / cell);
  let best = -1;
  let bestDist = Infinity;
  // Expand the search ring until something is found (handles sparse regions).
  for (let ring = 0; ring <= 4 && best < 0; ring += 1) {
    for (let dx = -ring; dx <= ring; dx += 1) {
      for (let dy = -ring; dy <= ring; dy += 1) {
        for (let dz = -ring; dz <= ring; dz += 1) {
          if (ring > 0 && Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== ring) continue;
          const bucket = grid.get(`${cx + dx},${cy + dy},${cz + dz}`);
          if (!bucket) continue;
          for (const i of bucket) {
            const d = (cp[i * 3] - x) ** 2 + (cp[i * 3 + 1] - y) ** 2 + (cp[i * 3 + 2] - z) ** 2;
            if (d < bestDist) { bestDist = d; best = i; }
          }
        }
      }
    }
  }
  if (best >= 0) {
    colors[v * 3] = cc[best * 3];
    colors[v * 3 + 1] = cc[best * 3 + 1];
    colors[v * 3 + 2] = cc[best * 3 + 2];
    matched += 1;
  } else {
    colors[v * 3] = colors[v * 3 + 1] = colors[v * 3 + 2] = 140;
  }
}

await writeFile(out, writeBinaryPly(mesh.positions, colors, mesh.indices));
console.log(`Coloured ${matched}/${vertexCount} vertices from ${pointCount} cloud points -> ${out}`);
console.log(`faces: ${mesh.indices.length / 3}, cell: ${cell.toFixed(3)}`);
