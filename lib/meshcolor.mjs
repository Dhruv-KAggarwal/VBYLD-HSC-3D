/**
 * Vertex-colour a dense mesh from the photographic point cloud, and ship it as
 * an upright, metric, self-contained GLB.
 * ---------------------------------------------------------------------------
 * OpenMVS's TextureMesh atlas is unreliable on a single forward aerial pass:
 * most faces have no clean view, so the baked atlas comes back mostly
 * "empty-colour" and the model renders blank. The dense point cloud, however,
 * carries a real colour per point sampled from the drone frames. Transferring
 * those colours onto the (much denser, un-decimated) reconstructed mesh gives a
 * genuinely photographic model every time, with no dependence on the atlas.
 *
 * This is lower-frequency than a per-pixel texture, but on a 300k-vertex mesh it
 * reads as real terrain — roads, rooftops, vegetation and rock all keep their
 * true colour — and it never fails closed to a grey/black mesh.
 */

import { writeGlb } from './glb.mjs';
import { applyRotation, scalePositions, boundsOf } from './geometry.mjs';
import { PointGrid } from './spatial.mjs';

/**
 * Voxel-cluster decimation of a triangle mesh.
 *
 * OpenMVS TextureMesh fragments a single forward pass into hundreds of thousands
 * of tiny texture patches (grazing angles give every face a different "best
 * view"), and packing that many patches exhausts a 16 GB box. Collapsing nearby
 * vertices onto a voxel grid cuts the face count several-fold while the per-pixel
 * photographic texture — baked afterwards — keeps the fine detail. Representative
 * vertex per cell is the centroid; faces that collapse to a sliver are dropped.
 *
 * @param {Float32Array} positions
 * @param {Uint32Array|number[]} indices
 * @param {number} cell  voxel edge length in mesh units
 */
export function decimateMesh(positions, indices, cell) {
  const key = new Map();
  const accum = [];
  const remap = new Int32Array(positions.length / 3);
  for (let v = 0; v < positions.length / 3; v += 1) {
    const x = positions[v * 3];
    const y = positions[v * 3 + 1];
    const z = positions[v * 3 + 2];
    const k = `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
    let idx = key.get(k);
    if (idx === undefined) { idx = accum.length / 4; key.set(k, idx); accum.push(0, 0, 0, 0); }
    accum[idx * 4] += x; accum[idx * 4 + 1] += y; accum[idx * 4 + 2] += z; accum[idx * 4 + 3] += 1;
    remap[v] = idx;
  }
  const outPos = new Float32Array((accum.length / 4) * 3);
  for (let i = 0; i < accum.length / 4; i += 1) {
    const n = accum[i * 4 + 3] || 1;
    outPos[i * 3] = accum[i * 4] / n;
    outPos[i * 3 + 1] = accum[i * 4 + 1] / n;
    outPos[i * 3 + 2] = accum[i * 4 + 2] / n;
  }
  const outIdx = [];
  for (let f = 0; f < indices.length; f += 3) {
    const a = remap[indices[f]];
    const b = remap[indices[f + 1]];
    const c = remap[indices[f + 2]];
    if (a === b || b === c || a === c) continue;
    outIdx.push(a, b, c);
  }
  return { positions: outPos, indices: new Uint32Array(outIdx) };
}

/**
 * True for sky / cloud colours, judged on colour alone.
 *
 * Deliberately narrow. The previous version also deleted anything mid-bright
 * and low-saturation with blue >= red, which is exactly what white and cream
 * rendered facades, concrete, light roofs, asphalt and water look like from the
 * air — it was removing real surface, not haze. Colour can only ever be a hint;
 * the height gate in markSky is what actually makes this safe.
 *
 * Two genuine signatures survive:
 *   - blown-out overexposed cloud: near-white and essentially neutral;
 *   - open blue sky: genuinely blue, so blue leads red by a wide margin.
 */
function isSkyColor(r, g, b) {
  const lum = (r + g + b) / 3;
  const sat = Math.max(r, g, b) - Math.min(r, g, b);
  const blownOut = lum > 232 && sat < 18;
  const blueSky = b - r > 38 && b > 140 && sat > 28;
  return blownOut || blueSky;
}

/**
 * Flag sky vertices by colour AND height.
 *
 * Reconstructed sky is not just a colour, it is a sheet that sits above
 * everything the drone actually flew over. Requiring both means a white rooftop
 * in the middle of the scene is kept while the melted ceiling above the skyline
 * still goes. Positions must already be gravity-aligned (+Y up).
 *
 * The height cut is a high percentile of the vertex distribution rather than a
 * fraction of the bounding box, so one stray floater cannot drag the threshold
 * up and disable the filter entirely.
 */
function markSky(positions, colors, vertexCount) {
  const sky = new Uint8Array(vertexCount);
  const candidate = [];
  for (let v = 0; v < vertexCount; v += 1) {
    if (isSkyColor(colors[v * 3], colors[v * 3 + 1], colors[v * 3 + 2])) candidate.push(v);
  }
  if (!candidate.length) return sky;

  const heights = new Float64Array(vertexCount);
  for (let v = 0; v < vertexCount; v += 1) heights[v] = positions[v * 3 + 1];
  const sorted = Float64Array.from(heights).sort();
  const p90 = sorted[Math.floor(sorted.length * 0.9)];
  const p50 = sorted[Math.floor(sorted.length * 0.5)];
  // Anything above the 90th percentile, with a little slack, is plausibly the
  // sky sheet. A flat scene (p90 ~ p50) gets a threshold above everything, so
  // nothing is removed — the correct answer when there is no sky geometry.
  const cut = p90 + (p90 - p50) * 0.25;

  for (const v of candidate) {
    if (heights[v] >= cut) sky[v] = 1;
  }
  return sky;
}

/**
 * Drop sky/cloud faces and disconnected floaters from a vertex-coloured mesh.
 *
 * A single forward aerial pass films a lot of sky; OpenMVS happily triangulates
 * the overexposed horizon into a flat sheet and leaves speckled floaters above
 * the terrain. Two cheap passes clean almost all of it:
 *   1. colour — remove faces whose vertices are sky/cloud coloured;
 *   2. topology — union-find the survivors and keep only components large enough
 *      to be real surface, which deletes the hanging islands.
 *
 * @returns {{ positions: Float32Array, indices: Uint32Array, colors: Uint8Array, removed: number }}
 */
export function cleanColoredMesh({ positions, indices, colors }) {
  const vertexCount = positions.length / 3;
  const sky = markSky(positions, colors, vertexCount);

  // Pass 1: keep faces with at most one sky vertex (a fully-sky triangle is sky).
  const kept = [];
  for (let f = 0; f < indices.length; f += 3) {
    const a = indices[f];
    const b = indices[f + 1];
    const c = indices[f + 2];
    if (sky[a] + sky[b] + sky[c] >= 2) continue;
    kept.push(a, b, c);
  }

  // Pass 2: union-find connected components over the kept faces.
  const parent = new Int32Array(vertexCount);
  for (let i = 0; i < vertexCount; i += 1) parent[i] = i;
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const union = (x, y) => { const rx = find(x); const ry = find(y); if (rx !== ry) parent[rx] = ry; };
  for (let i = 0; i < kept.length; i += 3) {
    union(kept[i], kept[i + 1]);
    union(kept[i + 1], kept[i + 2]);
  }
  const compSize = new Map();
  for (let i = 0; i < kept.length; i += 3) {
    const root = find(kept[i]);
    compSize.set(root, (compSize.get(root) || 0) + 1);
  }
  let biggest = 0;
  for (const n of compSize.values()) biggest = Math.max(biggest, n);
  // 0.2%, not 2%. A reconstruction legitimately splits into several large
  // components (a courtyard, a block across a road, terrain either side of a
  // tree line); a 2% floor deleted those along with the floaters.
  const minComp = Math.max(50, biggest * 0.002);

  // Compact the surviving vertices into a fresh, dense buffer.
  const remap = new Int32Array(vertexCount).fill(-1);
  const outPos = [];
  const outCol = [];
  const outIdx = [];
  let removed = 0;
  for (let i = 0; i < kept.length; i += 3) {
    if ((compSize.get(find(kept[i])) || 0) < minComp) { removed += 1; continue; }
    const tri = [kept[i], kept[i + 1], kept[i + 2]];
    for (const v of tri) {
      if (remap[v] < 0) {
        remap[v] = outPos.length / 3;
        outPos.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
        outCol.push(colors[v * 3], colors[v * 3 + 1], colors[v * 3 + 2]);
      }
      outIdx.push(remap[v]);
    }
  }
  return {
    positions: new Float32Array(outPos),
    indices: new Uint32Array(outIdx),
    colors: new Uint8Array(outCol),
    removed: (indices.length - kept.length) / 3 + removed,
  };
}

/**
 * Nearest-neighbour colour transfer through a hashed grid.
 *
 * The cell size follows the cloud's point spacing as a SURFACE (area per point).
 * Sizing it from the bounding volume, as this once did, put thousands of points
 * in each cell of an aerial scene and took minutes on a large mesh.
 *
 * @returns {Uint8Array} one RGB triple per mesh vertex
 */
export function colorizeVertices(meshPositions, cloudPositions, cloudColors) {
  const pointCount = cloudPositions.length / 3;
  const box = boundsOf(cloudPositions);
  const diag = Math.hypot(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]);
  // The farthest a colour may be borrowed from.
  const reach = Math.max(1e-4, (diag / Math.cbrt(Math.max(1, pointCount))) * 1.5) * 4;
  const fine = new PointGrid(cloudPositions);
  let coarse = null;

  const vertexCount = meshPositions.length / 3;
  const colors = new Uint8Array(vertexCount * 3);
  for (let v = 0; v < vertexCount; v += 1) {
    const x = meshPositions[v * 3];
    const y = meshPositions[v * 3 + 1];
    const z = meshPositions[v * 3 + 2];
    let hit = fine.nearest(x, y, z, Math.min(reach, fine.cellSize * 4));
    if (!hit && reach > fine.cellSize * 4) {
      // Sparse neighbourhoods are rare; search them on a coarser grid.
      coarse ||= new PointGrid(cloudPositions, reach / 3);
      hit = coarse.nearest(x, y, z, reach);
    }
    if (hit) {
      colors[v * 3] = cloudColors[hit.index * 3];
      colors[v * 3 + 1] = cloudColors[hit.index * 3 + 1];
      colors[v * 3 + 2] = cloudColors[hit.index * 3 + 2];
    } else {
      colors[v * 3] = colors[v * 3 + 1] = colors[v * 3 + 2] = 140;
    }
  }
  return colors;
}

/**
 * Build an upright, metric, vertex-coloured GLB from a reconstructed mesh.
 *
 * @param {object} args
 * @param {Float32Array} args.positions  mesh vertices in COLMAP frame
 * @param {Uint32Array|number[]} args.indices  triangle indices
 * @param {Float32Array} args.cloudPositions   coloured cloud positions (COLMAP frame)
 * @param {Uint8Array}   args.cloudColors      coloured cloud RGB
 * @param {object} args.transform  { rotation (3x3 row-major), scale, origin }
 * @returns {{ glb: Buffer, colors: Uint8Array, bounds: object }}
 */
export function buildColoredMeshGlb({ positions, indices, cloudPositions, cloudColors, transform, clean = true }) {
  const vColors = colorizeVertices(positions, cloudPositions, cloudColors);
  // Strip sky/cloud sheets and floaters before anything else, so the exported
  // model is the terrain and nothing but the terrain.
  const rotation = transform?.rotation;
  const origin = transform?.origin || [0, 0, 0];
  const scale = transform?.scale || 1;
  // Stand the mesh upright (gravity -> +Y) BEFORE cleaning, and size it to
  // metres afterwards, matching the DSM so both products share one coordinate
  // frame. The order matters now: the sky filter is height-gated, so it has to
  // know which way is up. Rotation changes no topology and no colour, so moving
  // it ahead of the clean costs nothing.
  const upright = rotation ? applyRotation(positions, rotation, origin) : positions;
  const base = clean
    ? cleanColoredMesh({ positions: upright, indices, colors: vColors })
    : { positions: upright, indices, colors: vColors };
  const scaled = scalePositions(base.positions, scale);
  const mesh = {
    positions: scaled,
    colors: base.colors,
    indices: base.indices,
    ortho: null,
    textured: false, // vertex colours, no texture map
  };
  const glb = writeGlb(mesh, null);
  return { glb, colors: base.colors, bounds: boundsOf(scaled), faces: base.indices.length / 3, removed: base.removed || 0 };
}
