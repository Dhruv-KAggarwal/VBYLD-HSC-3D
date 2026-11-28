/**
 * Everything the enhancement stages know about a finished job, expressed in the
 * mesh's own coordinate frame.
 *
 * A textured OpenMVS mesh keeps its vertices in the reconstruction frame and
 * carries gravity and metric scale in a node matrix. A vertex-coloured mesh has
 * them baked into the vertices. Cameras, sparse landmarks and the dense cloud
 * are all in the reconstruction frame, so in the second case they are moved
 * into the mesh frame here and nothing downstream has to know the difference.
 */

import { existsSync, statSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { readPoints3D, readUndistortedViews } from '../colmap.mjs';
import { parsePly } from '../ply.mjs';
import { computeSceneTransform } from '../export3d.mjs';
import { loadMesh, meshScale } from '../meshdata.mjs';
import { PointGrid, estimateSpacing } from '../spatial.mjs';
import { originalModelPath } from '../artifacts.mjs';
import { readJson, videoOfImage, readInputs, framesPath } from '../multivideo/manifest.mjs';

const MAX_CLOUD_POINTS = 4_000_000;

export function denseCloudPath(work) {
  return [join(work, 'mvs', 'scene_dense.ply'), join(work, 'dense', 'fused.ply')]
    .find((path) => existsSync(path) && statSync(path).size > 1024) || null;
}

export async function sceneTransform(work, metadata = null) {
  const stored = await readJson(join(work, 'products', 'transform.json'));
  if (stored?.rotation) return stored;
  // Jobs made before transforms were persisted: the computation is
  // deterministic, so repeating it reproduces the exported frame.
  const cloudPath = denseCloudPath(work) || join(work, 'sparse_points.ply');
  const meta = metadata || await readJson(join(work, 'products', 'metadata.json'));
  return computeSceneTransform({
    cloudPath, meshPath: cloudPath, workDir: work, altitude: meta?.altitudeM || 82, cameraOnly: true,
  });
}

function transformPoints(positions, { rotation: R, scale, origin }) {
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i] - origin[0];
    const y = positions[i + 1] - origin[1];
    const z = positions[i + 2] - origin[2];
    for (let k = 0; k < 3; k += 1) positions[i + k] = scale * (R[k][0] * x + R[k][1] * y + R[k][2] * z);
  }
  return positions;
}

/**
 * @param {object} [options]
 * @param {object} [options.mesh]  a mesh already loaded from `modelPath`
 */
export async function loadContext(work, { modelPath = originalModelPath(work), mesh = null } = {}) {
  if (!modelPath || !existsSync(modelPath)) {
    throw new Error('This job has no 3D mesh. Mesh stages need products/model.glb (or terrain.glb).');
  }
  const loaded = mesh || await loadMesh(modelPath);
  const metadata = await readJson(join(work, 'products', 'metadata.json'));
  const frameBaked = !loaded.rootMatrix;
  const transform = frameBaked ? await sceneTransform(work, metadata) : null;
  const m = loaded.rootMatrix;
  const metresPerUnit = meshScale(loaded);
  const up = frameBaked ? [0, 1, 0] : [m[1] / metresPerUnit, m[5] / metresPerUnit, m[9] / metresPerUnit];
  const scaleSource = metadata?.scaleSource || transform?.scaleSource || 'unknown';
  const toLocal = (positions) => (frameBaked ? transformPoints(positions, transform) : positions);

  let cloudCache;
  let viewCache;
  let pointCache;
  return {
    work,
    mesh: loaded,
    modelPath,
    metadata,
    metresPerUnit,
    /** Mesh units per reconstruction unit, for comparing camera depths. */
    depthScale: frameBaked ? transform.scale : 1,
    up,
    scaleSource,
    scaleReliable: scaleSource === 'flight-altitude',

    /** Dense cloud with a spatial index, or null if the workspace has none. */
    async cloud() {
      if (cloudCache !== undefined) return cloudCache;
      const path = denseCloudPath(work);
      if (!path) { cloudCache = null; return null; }
      const parsed = parsePly(await readFile(path));
      let { positions } = parsed;
      const total = positions.length / 3;
      if (total > MAX_CLOUD_POINTS) {
        const stride = Math.ceil(total / MAX_CLOUD_POINTS);
        const kept = new Float32Array(Math.floor(total / stride) * 3);
        for (let i = 0; i < kept.length / 3; i += 1) {
          kept.set(positions.subarray(i * stride * 3, i * stride * 3 + 3), i * 3);
        }
        positions = kept;
      }
      toLocal(positions);
      const grid = new PointGrid(positions);
      const spacing = estimateSpacing(grid);
      // Re-index at a cell size matched to the measured spacing.
      cloudCache = { path, grid: new PointGrid(positions, spacing * 3), spacing, points: positions.length / 3, totalPoints: total };
      return cloudCache;
    },

    /** Registered cameras with intrinsics, undistorted image path and source video. */
    async views() {
      if (viewCache) return viewCache;
      const inputs = await readInputs(work);
      const frames = await readJson(framesPath(work));
      const labels = new Map((inputs?.videos || []).map((v) => [v.id, v.settings?.label || v.originalName || v.id]));
      const sharpness = new Map((frames?.frames || []).filter((f) => f.fileName).map((f) => [f.fileName, f.sharpnessScore]));
      const views = await readUndistortedViews(work);
      viewCache = views.map((view) => {
        let { R, center } = view;
        if (frameBaked) {
          const { rotation: Q, scale, origin } = transform;
          // R_local = R_cam * Q^T ; centre moves like any other point.
          R = view.R.map((row) => [0, 1, 2].map((k) => row[0] * Q[k][0] + row[1] * Q[k][1] + row[2] * Q[k][2]));
          center = [0, 1, 2].map((k) => scale * (
            Q[k][0] * (view.center[0] - origin[0]) + Q[k][1] * (view.center[1] - origin[1]) + Q[k][2] * (view.center[2] - origin[2])));
        }
        const video = videoOfImage(view.name) || 'video_001';
        return {
          name: view.name,
          imageId: view.imageId,
          path: view.path,
          R,
          center,
          fx: view.fx,
          fy: view.fy,
          cx: view.cx,
          cy: view.cy,
          width: view.width,
          height: view.height,
          video,
          videoLabel: labels.get(video) || metadata?.source || video,
          sharpness: sharpness.get(view.name) ?? null,
        };
      });
      return viewCache;
    },

    /** Sparse landmarks in the mesh frame, or null. */
    async sparse() {
      if (pointCache !== undefined) return pointCache;
      const points = await readPoints3D(work);
      if (points) toLocal(points.positions);
      pointCache = points;
      return points;
    },
  };
}
