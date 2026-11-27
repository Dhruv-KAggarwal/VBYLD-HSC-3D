/**
 * Which registered cameras really observe each face.
 *
 * A camera counts for a face only if the face centroid projects inside the
 * image with positive depth, the face is turned towards the camera, and nothing
 * in the mesh is in front of it.
 */

import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { faceGeometry } from '../glbmesh.mjs';
import { DepthRasterizer, visibleAt } from './visibility.mjs';
import { MIN_INCIDENCE_COS } from './viewing-angles.mjs';
import { depthMapDirectory, parseDepthMap, compareDepth } from './depth.mjs';

export function faceSamples(mesh) {
  const faces = mesh.indices.length / 3;
  const centroid = new Float32Array(faces * 3);
  const normal = new Float32Array(faces * 3);
  const area = new Float32Array(faces);
  for (let f = 0; f < faces; f += 1) {
    const g = faceGeometry(mesh.positions, mesh.indices, f);
    centroid.set(g.centroid, f * 3);
    normal.set(g.normal, f * 3);
    area[f] = g.area;
  }
  return { faces, centroid, normal, area };
}

/**
 * Photogrammetry meshes are normally wound outwards, but if this one is wound
 * the other way every visibility test would invert. Vote using what the depth
 * buffers show.
 */
function windingSign(rasterizer, views, samples) {
  let positive = 0;
  let negative = 0;
  const picks = views.filter((_, i) => i % Math.max(1, Math.floor(views.length / 6)) === 0).slice(0, 6);
  const step = Math.max(1, Math.floor(samples.faces / 4000));
  for (const view of picks) {
    const frame = rasterizer.render(view);
    for (let f = 0; f < samples.faces; f += step) {
      const hit = projectToFrame(view, frame, samples.centroid, f);
      if (!hit || !visibleAt(frame, hit.x, hit.y, hit.z)) continue;
      const dot = samples.normal[f * 3] * hit.dx + samples.normal[f * 3 + 1] * hit.dy + samples.normal[f * 3 + 2] * hit.dz;
      if (dot < 0) positive += 1; else negative += 1;
    }
  }
  return negative > positive * 1.5 ? -1 : 1;
}

/** (dx,dy,dz) is the vector from the camera to the sample. */
function projectToFrame(view, frame, centroid, f) {
  const dx = centroid[f * 3] - view.center[0];
  const dy = centroid[f * 3 + 1] - view.center[1];
  const dz = centroid[f * 3 + 2] - view.center[2];
  const R = view.R;
  const z = R[2][0] * dx + R[2][1] * dy + R[2][2] * dz;
  if (z <= 1e-6) return null;
  const u = (view.fx * (R[0][0] * dx + R[0][1] * dy + R[0][2] * dz)) / z + view.cx;
  const v = (view.fy * (R[1][0] * dx + R[1][1] * dy + R[1][2] * dz)) / z + view.cy;
  if (u < 2 || v < 2 || u > view.width - 2 || v > view.height - 2) return null;
  return { u, v, x: u * frame.scale, y: v * frame.scale, z, dx, dy, dz };
}

export async function accumulateCoverage(context, samples, { resolution = 640, quality = null, onProgress = null } = {}) {
  const { mesh } = context;
  const views = await context.views();
  const n = samples.faces;
  const videoIds = [...new Set(views.map((v) => v.video))].sort();
  const out = {
    views,
    videoIds,
    cameras: new Uint16Array(n),
    effective: new Float32Array(n),
    direction: new Float32Array(n * 3),
    cosSum: new Float32Array(n),
    bestCos: new Float32Array(n),
    occluded: new Uint16Array(n),
    grazing: new Uint16Array(n),
    videoMask: new Uint32Array(n),
    qualitySum: new Float32Array(n),
    qualityWeight: new Float32Array(n),
    depthAgree: new Uint16Array(n),
    depthTested: new Uint16Array(n),
    visibleFacesPerCamera: [],
    depthMapsUsed: 0,
    winding: 1,
  };
  if (!views.length) return out;
  const rasterizer = new DepthRasterizer(mesh.positions, mesh.indices, resolution);
  const sign = windingSign(rasterizer, views, samples);
  out.winding = sign;
  const depthDirectory = depthMapDirectory(context.work);
  for (let i = 0; i < views.length; i += 1) {
    const view = views[i];
    const bit = 1 << Math.min(31, videoIds.indexOf(view.video));
    const sharpness = quality?.get(view.name) ?? null;
    let depthMap = null;
    if (depthDirectory) {
      const path = join(depthDirectory, `${view.name}.geometric.bin`);
      if (existsSync(path)) {
        depthMap = parseDepthMap(await readFile(path));
        if (depthMap) out.depthMapsUsed += 1;
      }
    }
    const frame = rasterizer.render(view);
    let seen = 0;
    for (let f = 0; f < n; f += 1) {
      const hit = projectToFrame(view, frame, samples.centroid, f);
      if (!hit) continue;
      const distance = Math.hypot(hit.dx, hit.dy, hit.dz) || 1;
      const cos = (-sign * (samples.normal[f * 3] * hit.dx + samples.normal[f * 3 + 1] * hit.dy + samples.normal[f * 3 + 2] * hit.dz)) / distance;
      if (cos <= 0) continue;
      if (!visibleAt(frame, hit.x, hit.y, hit.z)) {
        if (out.occluded[f] < 65535) out.occluded[f] += 1;
        continue;
      }
      if (cos < MIN_INCIDENCE_COS) {
        if (out.grazing[f] < 65535) out.grazing[f] += 1;
        continue;
      }
      seen += 1;
      out.cameras[f] += 1;
      out.effective[f] += Math.min(1, cos / 0.5);
      out.direction[f * 3] += hit.dx / distance;
      out.direction[f * 3 + 1] += hit.dy / distance;
      out.direction[f * 3 + 2] += hit.dz / distance;
      out.cosSum[f] += cos;
      if (cos > out.bestCos[f]) out.bestCos[f] = cos;
      out.videoMask[f] |= bit;
      if (sharpness != null) {
        out.qualitySum[f] += sharpness * cos;
        out.qualityWeight[f] += cos;
      }
      if (depthMap) {
        const agreement = compareDepth(depthMap, view, hit.u, hit.v, hit.z / context.depthScale);
        if (agreement !== null) {
          out.depthTested[f] += 1;
          if (agreement) out.depthAgree[f] += 1;
        }
      }
    }
    out.visibleFacesPerCamera.push({ image: view.name, video: view.video, visibleFaces: seen });
    onProgress?.((i + 1) / views.length);
    // Yield so progress messages and cancellation are not starved.
    if (i % 4 === 3) await new Promise((resolve) => setImmediate(resolve));
  }
  return out;
}
