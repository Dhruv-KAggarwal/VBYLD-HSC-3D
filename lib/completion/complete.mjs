/**
 * Fill the sides the drone never saw.
 *
 * Writes products/model-completed.glb: the photographed surface with its grey
 * and smeared triangles removed, plus estimated walls and ground in their place.
 * The input model is never modified, and the estimated surface is kept in its
 * own primitive so it can always be told apart from the reconstruction.
 */

import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { activeModel, completedModelPath, productsDir, recordArtifact } from '../artifacts.mjs';
import { MeshBVH } from '../bvh.mjs';
import { loadContext } from '../enhance/context.mjs';
import { recomputeNormals, writeMeshGlb } from '../glbmesh.mjs';
import { encodeImage } from '../image.mjs';
import { splitMesh, worldPositions } from '../meshdata.mjs';
import { writeJson } from '../multivideo/manifest.mjs';
import { jobViews } from '../views.mjs';
import { findUp, levelRotation, rotatePositions, unrotate } from './level.mjs';
import { classifyFaces, decodeTextures } from './observed.mjs';
import { spreadColours, swatchImage } from './repaint.mjs';
import { analyseTop, dilate } from './topview.mjs';
import { paintWall, planWalls } from './walls.mjs';

const ATLAS = 4096;
const PAD = 2;

/** Horizontal bounds of the vertices `use` selects. */
function boundsXZ(positions, use = () => true) {
  const region = { minX: Infinity, minZ: Infinity, maxX: -Infinity, maxZ: -Infinity };
  for (let i = 0; i < positions.length; i += 3) {
    if (!use(i / 3)) continue;
    region.minX = Math.min(region.minX, positions[i]);
    region.maxX = Math.max(region.maxX, positions[i]);
    region.minZ = Math.min(region.minZ, positions[i + 2]);
    region.maxZ = Math.max(region.maxZ, positions[i + 2]);
  }
  return region;
}

/** The filmed area in exported coordinates: the part worth filling. */
function regionFor(views, world) {
  if (!views?.focus) return boundsXZ(world);
  const { min, max } = views.focus;
  return { minX: min[0] - 20, minZ: min[1] - 20, maxX: max[0] + 20, maxZ: max[1] + 20 };
}

/** Shelf packing, tallest first. Returns false when the images do not fit. */
function pack(images, top) {
  let x = 0;
  let y = top;
  let shelf = 0;
  for (const image of [...images].sort((a, b) => b.height - a.height)) {
    const w = image.width + PAD * 2;
    const h = image.height + PAD * 2;
    if (w > ATLAS) return false;
    if (x + w > ATLAS) { x = 0; y += shelf; shelf = 0; }
    if (y + h > ATLAS) return false;
    image.x = x + PAD;
    image.y = y + PAD;
    x += w;
    shelf = Math.max(shelf, h);
  }
  return true;
}

/** Copy an image into the atlas, repeating its border into the padding. */
function blit(atlas, image, x, y) {
  for (let row = -PAD; row < image.height + PAD; row += 1) {
    const sy = Math.max(0, Math.min(image.height - 1, row));
    for (let col = -PAD; col < image.width + PAD; col += 1) {
      const sx = Math.max(0, Math.min(image.width - 1, col));
      const to = ((y + row) * ATLAS + x + col) * 3;
      const from = (sy * image.width + sx) * 3;
      for (let c = 0; c < 3; c += 1) atlas[to + c] = image.data[from + c];
    }
  }
}

/** Quads over every hole cell, textured from the top-down ground image. */
function groundSurface(top, atlasUv, add) {
  const { cols, rows, cell, minX, minZ, height, dtm, hole, roofHole } = top;
  // Reach one cell under the surrounding surface so no seam shows through.
  const grown = dilate(hole, cols, rows, 2);
  let area = 0;
  const level = (c, r, roof) => {
    let sum = 0;
    let count = 0;
    for (let y = r - 1; y <= r; y += 1) {
      for (let x = c - 1; x <= c; x += 1) {
        if (x < 0 || y < 0 || x >= cols || y >= rows) continue;
        const i = y * cols + x;
        const value = roof && roofHole[i] ? height[i] : dtm[i];
        if (Number.isFinite(value)) { sum += value; count += 1; }
      }
    }
    return count ? sum / count : 0;
  };
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const i = r * cols + c;
      if (!grown[i]) continue;
      const roof = Boolean(roofHole[i]);
      // Slightly below the real surface wherever the two overlap.
      const drop = hole[i] ? 0.1 : 0.5;
      const corners = [[c, r], [c, r + 1], [c + 1, r + 1], [c + 1, r]].map(([x, z]) => ({
        position: [minX + x * cell, level(x, z, roof) - drop, minZ + z * cell],
        uv: atlasUv(x / cols, z / rows),
      }));
      add(corners);
      if (hole[i]) area += cell * cell;
    }
  }
  return area;
}

/** A coarse floor beneath everything, so a remaining gap never shows the sky. */
function floorSurface(top, atlasUv, add) {
  const { cols, rows, cell, minX, minZ, dtm, height, seen, hole } = top;
  const step = Math.max(2, Math.round(2.4 / cell));
  const level = (c, r) => {
    let lowest = Infinity;
    for (let y = Math.max(0, r - step); y < Math.min(rows, r + step); y += 1) {
      for (let x = Math.max(0, c - step); x < Math.min(cols, c + step); x += 1) {
        const i = y * cols + x;
        lowest = Math.min(lowest, dtm[i], seen[i] ? height[i] : Infinity);
      }
    }
    return lowest - 0.8;
  };
  for (let r = 0; r < rows; r += step) {
    for (let c = 0; c < cols; c += step) {
      const c1 = Math.min(cols, c + step);
      const r1 = Math.min(rows, r + step);
      let inside = false;
      for (let y = r; y < r1 && !inside; y += 1) {
        for (let x = c; x < c1 && !inside; x += 1) inside = Boolean(seen[y * cols + x] || hole[y * cols + x]);
      }
      if (!inside) continue;
      add([[c, r], [c, r1], [c1, r1], [c1, r]].map(([x, z]) => ({
        position: [minX + x * cell, level(x, z), minZ + z * cell],
        uv: atlasUv(x / cols, z / rows),
      })));
    }
  }
}

/**
 * Which removed faces stay, repainted. Large or bridging triangles, and steep
 * ones an estimated wall now stands in for, go.
 */
function facesToRepaint(mesh, level, removed, plans) {
  const key = (x, y, z) => `${Math.floor(x / 2.5)},${Math.floor(y / 2.5)},${Math.floor(z / 2.5)}`;
  const walled = new Set();
  for (const { edge, tiles } of plans) {
    for (const [from, to, bottom, upper] of tiles) {
      const t = (from + to) / 2;
      const centre = [edge.start[0] + edge.along[0] * t, (bottom + upper) / 2, edge.start[1] + edge.along[1] * t];
      for (let dx = -2.5; dx <= 2.5; dx += 2.5) {
        for (let dy = -2.5; dy <= 2.5; dy += 2.5) {
          for (let dz = -2.5; dz <= 2.5; dz += 2.5) walled.add(key(centre[0] + dx, centre[1] + dy, centre[2] + dz));
        }
      }
    }
  }
  return removed.filter((f) => {
    const [a, b, c] = [0, 1, 2].map((k) => {
      const v = mesh.indices[f * 3 + k] * 3;
      return [level[v], level[v + 1], level[v + 2]];
    });
    const u = [0, 1, 2].map((k) => b[k] - a[k]);
    const v = [0, 1, 2].map((k) => c[k] - a[k]);
    const longest = Math.max(Math.hypot(...u), Math.hypot(...v), Math.hypot(b[0] - c[0], b[1] - c[1], b[2] - c[2]));
    const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const steep = Math.abs(n[1]) < 0.75 * Math.hypot(...n);
    // A level triangle is a roof or ground and is worth keeping even when wide.
    if (longest > (steep ? 5 : 15)) return false;
    if (!steep) return true;
    return !walled.has(key((a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3));
  });
}

export async function runCompletion(work, _settings = {}, onProgress = () => {}) {
  const source = await activeModel(work);
  onProgress(0.02, 'Loading the mesh and solved cameras');
  const context = await loadContext(work, { modelPath: source.path });
  const { mesh } = context;
  const cameras = await context.views();
  if (cameras.length < 3) throw new Error('Filling unseen sides needs the solved camera positions of this job.');
  const world = worldPositions(mesh);
  const textures = decodeTextures(mesh);
  const region = regionFor(await jobViews(work).catch(() => null), world);
  const faces = mesh.indices.length / 3;
  const inRegion = (f) => {
    const i = mesh.indices[f * 3] * 3;
    return world[i] >= region.minX && world[i] <= region.maxX && world[i + 2] >= region.minZ && world[i + 2] <= region.maxZ;
  };

  onProgress(0.05, `Checking which of ${faces.toLocaleString('en-US')} faces ${cameras.length} cameras really saw`);
  const seen = classifyFaces({
    mesh, textures, views: cameras, world, inRegion, onProgress: (f) => onProgress(0.05 + f * 0.55),
  });
  const faceMap = [];
  for (let f = 0; f < faces; f += 1) if (seen.keep[f]) faceMap.push(f);
  if (!faceMap.length) throw new Error('No photographed surface was found to build on.');

  onProgress(0.62, 'Finding ground and building outlines');
  // Everything from here is built in a levelled copy of the scene.
  const up = findUp(world, mesh.indices, (f) => seen.keep[f] && inRegion(f));
  const rotation = levelRotation(up);
  const level = rotatePositions(world, rotation);
  const wanted = new Uint8Array(level.length / 3);
  for (const f of faceMap) {
    if (inRegion(f)) for (let k = 0; k < 3; k += 1) wanted[mesh.indices[f * 3 + k]] = 1;
  }
  const top = analyseTop(mesh, textures, level, seen.keep, boundsXZ(level, (v) => wanted[v]));

  onProgress(0.68, `Rebuilding unseen walls of ${top.footprints.length} structure(s)`);
  const keptIndices = new Uint32Array(faceMap.length * 3);
  faceMap.forEach((f, i) => keptIndices.set(mesh.indices.subarray(f * 3, f * 3 + 3), i * 3));
  const bvh = new MeshBVH(level, keptIndices);

  // Facades are painted at the finest scale at which everything still fits.
  let ppm = 8;
  let walls;
  let images;
  let repainted;
  let swatches;
  const ground = { width: top.ground.width, height: top.ground.height, data: top.ground.data };
  const removed = [];
  for (let f = 0; f < faces; f += 1) if (!seen.keep[f]) removed.push(f);
  for (;;) {
    walls = planWalls(top, mesh, textures, bvh, faceMap, ppm, (f) => onProgress(0.68 + f * 0.2));
    images = walls.plans.map((plan) => paintWall(plan.edge, plan.donor, plan.tone, ppm));
    repainted = facesToRepaint(mesh, level, removed, walls.plans);
    swatches = swatchImage(spreadColours(mesh, textures, seen.keep, repainted), ATLAS - PAD * 2);
    const strip = { width: swatches.width, height: swatches.height, data: swatches.data };
    if (pack([strip, ...images], ground.height + PAD * 2)) {
      swatches.x = strip.x;
      swatches.y = strip.y;
      images.push(strip);
      break;
    }
    ppm *= 0.75;
    if (ppm < 1.5) throw new Error('The estimated surfaces do not fit one texture atlas.');
  }

  onProgress(0.9, 'Writing the filled model');
  // Only as tall as it needs to be, in a power of two for mipmapping.
  const bottom = Math.max(ground.height + PAD * 2, ...images.map((image) => image.y + image.height + PAD));
  const atlasHeight = 2 ** Math.ceil(Math.log2(bottom));
  const atlas = new Uint8Array(ATLAS * atlasHeight * 3).fill(128);
  blit(atlas, ground, PAD, PAD);
  images.forEach((image) => blit(atlas, image, image.x, image.y));

  // Estimated geometry is built level and in metres, then moved back.
  const m = mesh.rootMatrix;
  const squared = m ? m[0] ** 2 + m[1] ** 2 + m[2] ** 2 : 1;
  const toLocal = (point) => {
    const p = unrotate(point, rotation);
    if (!m) return p;
    const d = [p[0] - m[12], p[1] - m[13], p[2] - m[14]];
    return [0, 4, 8].map((column) => (m[column] * d[0] + m[column + 1] * d[1] + m[column + 2] * d[2]) / squared);
  };
  const positions = [];
  const uvs = [];
  const indices = [];
  const add = (corners) => {
    const first = positions.length / 3;
    for (const corner of corners) {
      positions.push(...toLocal(corner.position));
      uvs.push(...corner.uv);
    }
    indices.push(first, first + 1, first + 2, first, first + 2, first + 3);
  };

  const groundUv = (u, v) => [PAD + u * (ground.width - 1) + 0.5, PAD + v * (ground.height - 1) + 0.5];
  const groundArea = groundSurface(top, groundUv, add);
  floorSurface(top, groundUv, add);
  const groundVertices = positions.length / 3;
  const groundIndices = indices.length;
  repainted.forEach((f, n) => {
    const first = positions.length / 3;
    for (let k = 0; k < 3; k += 1) {
      const v = mesh.indices[f * 3 + k] * 3;
      positions.push(mesh.positions[v], mesh.positions[v + 1], mesh.positions[v + 2]);
      const [x, y] = swatches.uv(n, k);
      uvs.push(swatches.x + x, swatches.y + y);
    }
    indices.push(first, first + 1, first + 2);
  });

  let wallArea = 0;
  walls.plans.forEach((plan, n) => {
    const { edge } = plan;
    const image = images[n];
    for (const [from, to, bottom, upper] of plan.tiles) {
      const corner = (t, y) => ({
        // A little inside the outline, so any real surface stays in front.
        position: [
          edge.start[0] + edge.along[0] * t - edge.out[0] * 0.35, y,
          edge.start[1] + edge.along[1] * t - edge.out[1] * 0.35,
        ],
        uv: [
          image.x + Math.min(image.width - 0.5, Math.max(0.5, t * ppm)),
          image.y + Math.min(image.height - 0.5, Math.max(0.5, (edge.roof - y) * ppm)),
        ],
      });
      add([corner(from, bottom), corner(to, bottom), corner(to, upper), corner(from, upper)]);
      wallArea += (to - from) * (upper - bottom);
    }
  });

  const kept = splitMesh({
    ...mesh,
    indices: keptIndices,
    facePrim: Uint16Array.from(faceMap, (f) => mesh.facePrim[f]),
  }).primitives;
  const outImages = [...mesh.images];
  const primitives = [...kept];
  if (indices.length) {
    const { bytes, mime } = await encodeImage({ width: ATLAS, height: atlasHeight, data: atlas }, { format: 'jpeg', quality: 90 });
    outImages.push({ bytes, mime });
    // Ground is kept apart from walls so a viewer can hide it from underneath,
    // where it would otherwise cover the model.
    const part = (name, firstVertex, endVertex, firstIndex, endIndex, extras) => {
      if (endIndex <= firstIndex) return;
      const local = Float32Array.from(positions.slice(firstVertex * 3, endVertex * 3));
      const faceIndices = Uint32Array.from(indices.slice(firstIndex, endIndex), (v) => v - firstVertex);
      primitives.push({
        mode: 4,
        positions: local,
        normals: recomputeNormals(local, faceIndices),
        uvs: Float32Array.from(uvs.slice(firstVertex * 2, endVertex * 2), (value, i) => value / (i % 2 ? atlasHeight : ATLAS)),
        colors: null,
        indices: faceIndices,
        material: { name },
        imageIndex: outImages.length - 1,
        extras: { skyforgeEstimated: true, ...extras },
      });
    };
    part('SkyforgeEstimatedGround', 0, groundVertices, 0, groundIndices, { skyforgeGround: true });
    part('SkyforgeEstimated', groundVertices, positions.length / 3, groundIndices, indices.length, {});
  }
  const target = completedModelPath(work);
  await writeFile(target, writeMeshGlb({
    primitives: [...primitives, ...(mesh.passthrough || [])],
    images: outImages,
    rootMatrix: mesh.rootMatrix,
    unlit: mesh.unlit,
    generator: 'Skyforge completion',
    extras: { skyforge: 'Unseen sides are estimated, not measured.' },
  }));

  const report = {
    createdAt: new Date().toISOString(),
    sourceModel: source.which,
    sourceTriangles: faces,
    removedTriangles: faces - faceMap.length,
    repaintedTriangles: repainted.length,
    removedBecause: {
      untextured: seen.untextured, neverSeen: seen.unseen, seenOnlyEdgeOn: seen.grazing, leftoverFragments: seen.fragments,
    },
    estimatedTriangles: indices.length / 3,
    sceneTiltDegrees: Number(((Math.acos(Math.min(1, up[1])) * 180) / Math.PI).toFixed(1)),
    estimatedGroundM2: Math.round(groundArea),
    estimatedWallM2: Math.round(wallArea),
    buildingsFilled: walls.buildings,
    buildingsWithCopiedFacade: walls.copied,
    facadePixelsPerMetre: Number(ppm.toFixed(2)),
    note: 'Estimated walls and ground are a visual stand-in for surfaces the drone never saw. Do not measure them.',
  };
  await writeJson(join(productsDir(work), 'completion.json'), report);
  await recordArtifact(work, 'completion', { sources: [source.path], model: source.which, followsActive: true });
  onProgress(1);
  return report;
}
