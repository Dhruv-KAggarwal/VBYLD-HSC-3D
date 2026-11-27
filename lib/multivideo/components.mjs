/**
 * Dense reconstruction of the sub-models that could not be merged.
 *
 * Each gets its own workspace, its own scene transform and its own model file.
 * They share no coordinate frame with the main model and are never presented as
 * if they did. Everything here is best-effort: the main model already exists.
 */

import { existsSync, statSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { computeSceneTransform } from '../export3d.mjs';
import { buildTexturedMesh, openmvsCommands, isCancellation } from '../openmvs.mjs';
import { embedExternalImages, reorientGlb } from '../glbtransform.mjs';
import { buildColoredMeshGlb } from '../meshcolor.mjs';
import { parsePly } from '../ply.mjs';
import { videoOfImage, writeJson } from './manifest.mjs';

const MIN_IMAGES = 8;
const MAX_COMPONENTS = 6;

export async function reconstructComponents({
  job, work, frames, components, registration, quality, altitude, useMvs, run, commands, logger, denseFallback,
}) {
  const results = [];
  // No tracker: the main run's stage bar is already complete for this work.
  const sub = {
    logger,
    cancelHooks: job.cancelHooks,
    get cancelled() { return job.cancelled; },
    warning: '',
  };
  const noTracker = { start() {}, complete() {}, skip() {}, fraction() {} };
  for (let i = 0; i < components.length && i < MAX_COMPONENTS; i += 1) {
    const component = components[i];
    const number = i + 2;
    const videos = (registration.components.find((c) => c.component === number)?.videos || []).map((v) => v.id);
    const entry = { component: number, images: component.count, videos, model: null, status: 'skipped', reason: null };
    results.push(entry);
    if (component.count < MIN_IMAGES) { entry.reason = `Only ${component.count} registered images; too few for a dense model.`; continue; }
    if (job.cancelled) break;
    const dir = join(work, 'components', `component_${number}`);
    const dense = join(dir, 'dense');
    const mvsDir = join(dir, 'mvs');
    const out = join(work, 'products', 'components', `component_${number}`);
    try {
      await mkdir(dense, { recursive: true });
      await mkdir(mvsDir, { recursive: true });
      await mkdir(out, { recursive: true });
      logger.step(`Reconstructing separate component ${number} (${component.count} images from ${videos.join(', ')})`);
      await run(commands.colmap, [
        'image_undistorter', '--image_path', frames, '--input_path', component.path, '--output_path', dense,
        '--output_type', 'COLMAP', '--max_image_size', quality.undistortSize,
      ], dir, sub, `Undistorting component ${number}`, {});
      let cloud = null;
      let textured = null;
      let meshPly = null;
      if (useMvs) {
        const mvs = await buildTexturedMesh({
          run, commands: openmvsCommands(commands.openmvs), dense, mvsDir, job: sub, tracker: null,
          mvs: quality.mvs, logger, imageCount: component.count,
        });
        ({ denseCloud: cloud, modelGlb: textured, meshPly } = mvs);
      } else {
        cloud = await denseFallback({ dense, work: dir, job: sub, tracker: noTracker, quality, chosen: { images: component.count }, frameCount: component.count });
        if (cloud && existsSync(`${cloud}.vis`)) {
          const delaunay = join(dense, 'meshed-delaunay.ply');
          await run(commands.colmap, [
            'delaunay_mesher', '--input_path', dense, '--input_type', 'dense', '--output_path', delaunay,
          ], dir, sub, `Meshing component ${number}`, { timeoutMs: 120 * 60 * 1000 });
          meshPly = delaunay;
        }
      }
      if (!cloud) throw new Error('No dense point cloud was produced.');
      const altitudes = registration.videos.filter((v) => videos.includes(v.id)).map((v) => ({
        id: v.id, label: v.label, altitudeM: job.videoAltitudes?.[v.id] ?? null,
      }));
      const transform = await computeSceneTransform({ cloudPath: cloud, meshPath: cloud, workDir: dir, altitude, videos: altitudes });
      let glb = null;
      if (textured && existsSync(textured)) {
        const embedded = embedExternalImages(await readFile(textured), dirname(textured));
        glb = reorientGlb(embedded.buffer, transform);
      } else if (meshPly && existsSync(meshPly) && statSync(meshPly).size > 1024) {
        const mesh = parsePly(await readFile(meshPly));
        const points = parsePly(await readFile(cloud));
        if (mesh.indices?.length && points.colors) {
          glb = buildColoredMeshGlb({
            positions: mesh.positions, indices: mesh.indices, cloudPositions: points.positions, cloudColors: points.colors, transform,
          }).glb;
        }
      }
      if (!glb) throw new Error('No mesh was produced.');
      await writeFile(join(out, 'model.glb'), glb);
      await writeJson(join(out, 'transform.json'), transform);
      entry.status = 'complete';
      entry.model = `components/component_${number}/model.glb`;
      entry.videoLabels = [...new Set([...component.images].map(videoOfImage))];
      entry.scaleSource = transform.scaleSource;
      logger.ok(`Component ${number} written as ${entry.model}`);
    } catch (error) {
      if (isCancellation(error)) throw error;
      entry.status = 'failed';
      entry.reason = error.message.slice(0, 400);
      logger.warn(`Component ${number} could not be reconstructed: ${entry.reason}`);
    }
  }
  return results;
}
