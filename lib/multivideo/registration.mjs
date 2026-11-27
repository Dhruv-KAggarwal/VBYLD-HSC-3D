/**
 * Unified registration: collect the sparse models the mapper produced, merge the
 * ones that provably share cameras, and report what registered where.
 *
 * Merging uses COLMAP's `model_merger`, which estimates a Sim(3) from images
 * registered in both models, followed by bundle adjustment. Two models with no
 * shared images are NOT aligned by any other means: there is no verified
 * evidence of how they relate, and ICP on non-overlapping geometry would invent
 * it. They are reported as separate components instead.
 */

import { existsSync, readdirSync } from 'node:fs';
import { mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { readModelPoses } from '../colmap.mjs';
import { videoOfImage } from './manifest.mjs';

const MIN_SHARED_IMAGES = 3;
const ALTITUDE_TOLERANCE = 1.15;

export async function describeModel(path) {
  const poses = await readModelPoses(path);
  return { path, images: new Set(poses.map((p) => p.name)), count: poses.length, poses };
}

export async function listSparseModels(sparse) {
  if (!existsSync(sparse)) return [];
  const models = [];
  for (const entry of readdirSync(sparse, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const path = join(sparse, entry.name);
    if (!existsSync(join(path, 'images.bin')) && !existsSync(join(path, 'images.txt'))) continue;
    models.push(await describeModel(path));
  }
  return models.sort((a, b) => b.count - a.count);
}

/**
 * Fold models into the largest one wherever shared cameras allow it.
 *
 * @returns {{models:Array, merges:Array}} models[0] is the unified model.
 */
export async function mergeModels({ run, colmap, models, work, job, logger }) {
  const merges = [];
  if (models.length < 2) return { models, merges };
  let base = models[0];
  let remaining = models.slice(1);
  let step = 0;
  let progressed = true;
  // Repeat: absorbing one model can create the overlap another one needed.
  while (progressed && remaining.length) {
    progressed = false;
    for (const other of [...remaining]) {
      let shared = 0;
      for (const name of other.images) if (base.images.has(name)) shared += 1;
      if (shared < MIN_SHARED_IMAGES) continue;
      step += 1;
      const merged = join(work, 'sparse_merged', `merge_${step}`);
      const adjusted = join(work, 'sparse_merged', `merge_${step}_ba`);
      await rm(merged, { recursive: true, force: true });
      await rm(adjusted, { recursive: true, force: true });
      await mkdir(merged, { recursive: true });
      await mkdir(adjusted, { recursive: true });
      try {
        await run(colmap, [
          'model_merger', '--input_path1', base.path, '--input_path2', other.path, '--output_path', merged,
          '--max_reproj_error', '8',
        ], work, job, 'Merging sparse models on shared cameras', { stage: 'mapper' });
        const candidate = await describeModel(merged);
        // model_merger writes the first input unchanged when alignment fails.
        if (candidate.count <= base.count) {
          merges.push({ sharedImages: shared, merged: false, reason: 'Alignment on shared cameras was not consistent.' });
          continue;
        }
        await run(colmap, ['bundle_adjuster', '--input_path', merged, '--output_path', adjusted],
          work, job, 'Global bundle adjustment of the merged model', { stage: 'mapper', timeoutMs: 90 * 60 * 1000 });
        base = await describeModel(adjusted);
        remaining = remaining.filter((m) => m !== other);
        merges.push({ sharedImages: shared, merged: true, images: base.count });
        logger?.ok(`Merged a sub-model on ${shared} shared cameras; unified model now has ${base.count} images`);
        progressed = true;
      } catch (error) {
        if (/cancelled/i.test(error.message)) throw error;
        merges.push({ sharedImages: shared, merged: false, reason: error.message.slice(0, 300) });
        logger?.warn(`Could not merge a sub-model: ${error.message.slice(0, 300)}`);
      }
    }
  }
  return { models: [base, ...remaining.sort((a, b) => b.count - a.count)], merges };
}

/**
 * Per-video registration and the landmarks videos share.
 *
 * @param {Array<{id:string,label:string,frames:Array<{fileName:string}>}>} videos
 * @param {Array} models  from listSparseModels/mergeModels; models[0] is primary
 * @param {object|null} points  sparse points of the primary model
 */
export function registrationReport(videos, models, points) {
  const primary = models[0] || { images: new Set(), poses: [] };
  const perVideo = videos.map((video) => {
    const total = video.frames.length;
    const registered = video.frames.filter((f) => primary.images.has(f.fileName)).length;
    const elsewhere = models.slice(1).map((m, i) => ({
      component: i + 2,
      images: video.frames.filter((f) => m.images.has(f.fileName)).length,
    })).filter((e) => e.images > 0);
    const share = total ? registered / total : 0;
    return {
      id: video.id,
      label: video.label,
      selectedFrames: total,
      registeredFrames: registered,
      registrationPercent: Math.round(share * 1000) / 10,
      otherComponents: elsewhere,
      merged: registered >= Math.min(3, total) && share >= 0.1,
    };
  });

  const videoById = new Map(primary.poses.map((p) => [p.imageId, videoOfImage(p.name)]));
  const shared = new Map();
  const anchored = new Map(videos.map((v) => [v.id, 0]));
  if (points?.count) {
    const seen = new Set();
    for (let i = 0; i < points.count; i += 1) {
      seen.clear();
      for (let k = points.trackOffsets[i]; k < points.trackOffsets[i + 1]; k += 1) {
        const video = videoById.get(points.tracks[k]);
        if (video) seen.add(video);
      }
      if (seen.size < 2) continue;
      const list = [...seen].sort();
      for (const id of list) anchored.set(id, (anchored.get(id) || 0) + 1);
      for (let a = 0; a < list.length; a += 1) {
        for (let b = a + 1; b < list.length; b += 1) {
          const key = `${list[a]}|${list[b]}`;
          shared.set(key, (shared.get(key) || 0) + 1);
        }
      }
    }
  }
  for (const entry of perVideo) {
    entry.sharedLandmarks = anchored.get(entry.id) || 0;
    entry.weaklyAnchored = entry.merged && videos.length > 1 && entry.sharedLandmarks < 30;
  }

  const components = models.map((model, i) => {
    const counts = videos.map((v) => ({ id: v.id, images: v.frames.filter((f) => model.images.has(f.fileName)).length }))
      .filter((c) => c.images > 0);
    return { component: i + 1, images: model.count, videos: counts };
  });
  const unmerged = perVideo.filter((v) => !v.merged);
  const messages = unmerged.map((v) => {
    if (v.otherComponents.length) {
      return `${v.label} could not be merged: it solved only as a separate component (no verified overlap strong enough to place it in the main model).`;
    }
    return `${v.label} could not be registered: too few verified correspondences with itself or the other videos.`;
  });
  for (const v of perVideo.filter((x) => x.weaklyAnchored)) {
    messages.push(`${v.label} is attached by only ${v.sharedLandmarks} shared landmarks; check its alignment before relying on it.`);
  }
  return {
    videos: perVideo,
    components,
    registeredImages: primary.count,
    sharedLandmarks: [...shared.entries()].map(([key, count]) => {
      const [a, b] = key.split('|');
      return { a, b, points: count };
    }).sort((x, y) => y.points - x.points),
    unmergedVideos: unmerged.map((v) => v.id),
    messages,
  };
}

const medianOf = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};

/**
 * Decide the metric scale when several videos each imply one.
 *
 * @param {Array<{video:string, clearance:number}>} cameras  height of every
 *   registered camera above the reconstructed ground, in model units
 * @param {Array<{id:string,label:string,altitudeM:number|null}>} videos  in operator order
 * @param {number} altitude  the job-wide altitude
 */
export function resolveAltitudeScale(cameras, videos, altitude) {
  const entries = videos.map((video) => {
    const own = cameras.filter((c) => c.video === video.id).map((c) => c.clearance);
    const clearance = own.length >= 3 ? medianOf(own) : null;
    return {
      id: video.id,
      label: video.label,
      altitudeM: video.altitudeM ?? null,
      cameras: own.length,
      clearanceUnits: clearance,
      scale: clearance && video.altitudeM ? video.altitudeM / clearance : null,
    };
  });
  const supplied = entries.filter((e) => e.scale);
  if (supplied.length) {
    const scales = supplied.map((e) => e.scale);
    const spread = Math.max(...scales) / Math.min(...scales);
    const weighted = supplied.flatMap((e) => Array(Math.min(500, e.cameras)).fill(e.scale));
    const consistent = spread <= ALTITUDE_TOLERANCE;
    return {
      scale: medianOf(weighted),
      reliable: consistent,
      scaleSource: consistent ? 'flight-altitude' : 'flight-altitude-inconsistent',
      spread,
      entries,
      message: consistent
        ? (supplied.length > 1 ? `Per-video altitudes agree within ${((spread - 1) * 100).toFixed(0)}%.` : `Scale set from the altitude of ${supplied[0].label}.`)
        : `Per-video altitudes disagree: they imply scales ${((spread - 1) * 100).toFixed(0)}% apart (${supplied.map((e) => `${e.label}: ${e.altitudeM} m`).join(', ')}). The median scale was applied and measurements are marked unreliable. Check the altitudes.`,
    };
  }
  // Only the job-wide figure: it can describe one flight height, so check that
  // the videos actually flew at one.
  const measured = entries.filter((e) => e.clearanceUnits);
  if (!measured.length) return null;
  const clearances = measured.map((e) => e.clearanceUnits);
  const spread = Math.max(...clearances) / Math.min(...clearances);
  const reference = measured[0];
  const consistent = spread <= 1.25;
  return {
    scale: altitude / (consistent ? medianOf(cameras.map((c) => c.clearance)) : reference.clearanceUnits),
    reliable: consistent,
    scaleSource: consistent ? 'flight-altitude' : 'flight-altitude-inconsistent',
    spread,
    entries,
    message: consistent
      ? `One altitude (${altitude} m) applied; the videos flew at similar heights.`
      : `The videos flew at heights differing by ${((spread - 1) * 100).toFixed(0)}%, so the single altitude of ${altitude} m was applied to ${reference.label} only and measurements are marked unreliable. Enter per-video altitudes to fix this.`,
  };
}
