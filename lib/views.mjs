/**
 * Where the drone looked from, and which part of the scene it actually filmed.
 *
 * A forward-looking flight reconstructs a few hundred metres well and then
 * scraps of countryside out to the horizon. Framing the whole bounding box makes
 * the well-reconstructed buildings a few pixels wide, so the viewer instead
 * opens from a real camera position and can hide the poorly observed surroundings.
 */

import { join } from 'node:path';
import { readPoints3D, readUndistortedViews } from './colmap.mjs';
import { sceneTransform } from './enhance/context.mjs';
import { readJson, videoOfImage, writeJson } from './multivideo/manifest.mjs';
import { median } from './spatial.mjs';

const MAX_VIEWS = 120;

const percentile = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))];

export async function computeViews(work) {
  const cameras = await readUndistortedViews(work);
  if (cameras.length < 3) return null;
  const { rotation: R, scale, origin } = await sceneTransform(work);
  const toWorld = (p) => [0, 1, 2].map((k) => scale * (
    R[k][0] * (p[0] - origin[0]) + R[k][1] * (p[1] - origin[1]) + R[k][2] * (p[2] - origin[2])));
  const direction = (d) => [0, 1, 2].map((k) => R[k][0] * d[0] + R[k][1] * d[1] + R[k][2] * d[2]);

  // The sparse landmarks each image observed give the distance it was looking.
  const points = await readPoints3D(work);
  const seen = new Map(cameras.map((c) => [c.imageId, []]));
  const world = points ? new Float32Array(points.count * 3) : null;
  const nearest = points ? new Float32Array(points.count).fill(Infinity) : null;
  if (points) {
    const centres = new Map(cameras.map((c) => [c.imageId, toWorld(c.center)]));
    for (let i = 0; i < points.count; i += 1) {
      const p = toWorld([points.positions[i * 3], points.positions[i * 3 + 1], points.positions[i * 3 + 2]]);
      world.set(p, i * 3);
      for (let k = points.trackOffsets[i]; k < points.trackOffsets[i + 1]; k += 1) {
        const centre = centres.get(points.tracks[k]);
        if (!centre) continue;
        const d = Math.hypot(p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]);
        if (seen.get(points.tracks[k]).length < 4000) seen.get(points.tracks[k]).push(d);
        if (d < nearest[i]) nearest[i] = d;
      }
    }
  }

  const views = cameras.map((camera) => {
    const position = toWorld(camera.center);
    const forward = direction(camera.viewAxis);
    const length = Math.hypot(...forward) || 1;
    const unit = forward.map((v) => v / length);
    const distances = seen.get(camera.imageId);
    // Without landmarks, aim at the ground under the line of sight.
    const fallback = unit[1] < -0.05 ? Math.abs(position[1] / unit[1]) : Math.abs(position[1]) * 3;
    const depth = distances?.length > 10 ? median(distances) : fallback || 100;
    return {
      name: camera.name,
      video: videoOfImage(camera.name),
      position,
      target: [0, 1, 2].map((k) => position[k] + unit[k] * depth),
      depth,
      fov: (2 * Math.atan(camera.height / (2 * camera.fy)) * 180) / Math.PI,
    };
  }).sort((a, b) => a.name.localeCompare(b.name));

  const typicalDepth = median(views.map((v) => v.depth));
  const centre = [0, 1, 2].map((k) => median(views.map((v) => v.target[k])));

  // The filmed area: landmarks that at least one camera saw from reasonably
  // close, and the places the cameras were aimed at.
  const xs = views.map((v) => v.target[0]);
  const zs = views.map((v) => v.target[2]);
  if (points) {
    const close = typicalDepth * 1.25;
    for (let i = 0; i < points.count; i += 1) {
      if (nearest[i] <= close) { xs.push(world[i * 3]); zs.push(world[i * 3 + 2]); }
    }
  }
  xs.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  const margin = typicalDepth * 0.15;
  const focus = {
    center: centre,
    min: [percentile(xs, 0.03) - margin, percentile(zs, 0.03) - margin],
    max: [percentile(xs, 0.97) + margin, percentile(zs, 0.97) + margin],
  };
  focus.radius = Math.max(focus.max[0] - focus.min[0], focus.max[1] - focus.min[1]) / 2;

  // Open on the view that looks closest to the centre from a typical distance.
  const score = (v) => Math.hypot(v.target[0] - centre[0], v.target[2] - centre[2]) + Math.abs(v.depth - typicalDepth) * 0.5;
  const step = Math.max(1, Math.ceil(views.length / MAX_VIEWS));
  const listed = views.filter((_, i) => i % step === 0);
  let start = 0;
  listed.forEach((v, i) => { if (score(v) < score(listed[start])) start = i; });
  return { version: 1, frame: 'viewer world coordinates (Y up, metres)', focus, start, views: listed };
}

/** Cached in products/views.json; recomputed if the camera model is newer. */
export async function jobViews(work) {
  const path = join(work, 'products', 'views.json');
  const cached = await readJson(path);
  if (cached?.version === 1) return cached;
  const computed = await computeViews(work);
  if (computed) await writeJson(path, computed);
  return computed;
}
