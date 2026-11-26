/**
 * Blur / keyframe selection.
 * ---------------------------------------------------------------------------
 * Motion blur is poison for feature matching: a smeared frame contributes bad
 * keypoints that drag the whole pose solve off. Dropping the blurriest frames
 * before COLMAP/hloc ever sees them is the single cheapest quality win in the
 * SkyForge_Upgrade_Plan (§6.1), and it needs no new dependency — the bundled
 * FFmpeg decodes each frame to a tiny greyscale thumbnail and we score its
 * sharpness with the variance of the Laplacian (the classic blur metric).
 *
 * Frames far below the run's own median sharpness are deleted, but never so many
 * that coverage collapses: we keep at least `minKeep` frames and never drop more
 * than `maxDropFraction`, so a uniformly soft clip is left intact rather than
 * gutted.
 */

import { spawn } from 'node:child_process';
import { readdirSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';

const THUMB_W = 192;
const THUMB_H = 108;
const FRAME_BYTES = THUMB_W * THUMB_H;

/** Variance of the 4-neighbour Laplacian over a greyscale thumbnail. */
function laplacianVariance(gray) {
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = 1; y < THUMB_H - 1; y += 1) {
    for (let x = 1; x < THUMB_W - 1; x += 1) {
      const i = y * THUMB_W + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - THUMB_W] - gray[i + THUMB_W];
      sum += lap;
      sumSq += lap * lap;
      count += 1;
    }
  }
  if (!count) return 0;
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

/** Decode the whole frame sequence to packed greyscale thumbnails in one pass. */
function decodeThumbnails(ffmpeg, framesDir) {
  return new Promise((resolveDecode) => {
    const child = spawn(ffmpeg, [
      '-v', 'error',
      '-i', join(framesDir, 'frame_%06d.jpg'),
      '-vf', `scale=${THUMB_W}:${THUMB_H},format=gray`,
      '-f', 'rawvideo', 'pipe:1',
    ], { windowsHide: true });
    const chunks = [];
    child.stdout.on('data', (c) => chunks.push(c));
    child.on('error', () => resolveDecode(null));
    child.on('close', () => resolveDecode(chunks.length ? Buffer.concat(chunks) : null));
  });
}

/**
 * Score every extracted frame and delete the blurriest outliers in place.
 *
 * @param {string} ffmpeg    path to the ffmpeg binary
 * @param {string} framesDir directory of frame_%06d.jpg
 * @param {object} [opts]
 * @param {object} [logger]
 * @returns {Promise<{ total:number, kept:number, dropped:number, medianSharpness:number }>}
 */
export async function selectSharpFrames(ffmpeg, framesDir, { minKeep = 24, maxDropFraction = 0.3, relThreshold = 0.45 } = {}, logger = null) {
  const files = readdirSync(framesDir).filter((f) => f.endsWith('.jpg')).sort();
  const total = files.length;
  if (total <= minKeep) return { total, kept: total, dropped: 0, medianSharpness: 0 };

  const raw = await decodeThumbnails(ffmpeg, framesDir);
  // If decode failed or the frame count does not line up, do nothing rather than
  // risk deleting the wrong frames.
  if (!raw || Math.floor(raw.length / FRAME_BYTES) !== total) {
    logger?.warn?.('Blur filter skipped (could not score frames); keeping all.');
    return { total, kept: total, dropped: 0, medianSharpness: 0 };
  }

  const scored = files.map((name, i) => ({
    name,
    score: laplacianVariance(raw.subarray(i * FRAME_BYTES, (i + 1) * FRAME_BYTES)),
  }));
  const sortedScores = scored.map((s) => s.score).sort((a, b) => a - b);
  const median = sortedScores[Math.floor(sortedScores.length / 2)] || 0;
  const cutoff = median * relThreshold;

  // Candidates to drop: clearly below the run's own sharpness, blurriest first.
  const droppable = scored.filter((s) => s.score < cutoff).sort((a, b) => a.score - b.score);
  const maxDrop = Math.min(droppable.length, Math.floor(total * maxDropFraction), total - minKeep);
  const toDrop = droppable.slice(0, Math.max(0, maxDrop));

  for (const f of toDrop) {
    try { unlinkSync(join(framesDir, f.name)); } catch { /* already gone */ }
  }
  const dropped = toDrop.length;
  logger?.info?.(`Blur filter: kept ${total - dropped}/${total} frames (dropped ${dropped} below ${Math.round(relThreshold * 100)}% of median sharpness)`);
  return { total, kept: total - dropped, dropped, medianSharpness: median };
}
