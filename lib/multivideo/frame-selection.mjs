/**
 * Adaptive frame sampling, one video at a time.
 *
 * Pass 1 decodes oversampled greyscale thumbnails and measures sharpness and
 * how much the view changes between neighbours. Frames are then spaced evenly
 * in "visual change" rather than in time, so a hover contributes a few frames
 * and a fast orbit contributes many, and the sharpest candidate in each slot
 * wins. Pass 2 decodes only the chosen frames at working resolution.
 */

import { readdir, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { capture } from './video-input.mjs';
import {
  THUMB_BYTES, THUMB_H, THUMB_W, differenceHash, downsample, estimateShift, laplacianVariance,
} from './imaging.mjs';

const RATE = { fast: 1.5, balanced: 2, detailed: 2.5 };
const TOTAL_CAP = { fast: 180, balanced: 300, detailed: 420 };
const MIN_PER_VIDEO = 24;
const MAX_CANDIDATES = 1500;
const BLUR_THRESHOLD = 0.45;

export function trimmedDuration(video) {
  const { trimStart = 0, trimEnd = null } = video.settings || {};
  const end = trimEnd != null ? Math.min(trimEnd, video.probe.durationSeconds) : video.probe.durationSeconds;
  return Math.max(0.5, end - Math.min(trimStart, end - 0.5));
}

/** Total frame ceiling for the whole job, from profile, hardware and overrides. */
export function totalFrameBudget(quality, hardware = {}, requested = null) {
  const forced = Number(process.env.SKYFORGE_MAX_FRAMES) || Number(requested) || 0;
  if (forced > 0) return Math.min(2000, Math.max(24, Math.round(forced)));
  const base = TOTAL_CAP[quality] || TOTAL_CAP.balanced;
  const gpu = Math.min(1.25, Math.max(0.6, hardware.frameScale || 1));
  const memory = hardware.ramGb && hardware.ramGb < 12 ? 0.7 : 1;
  return Math.round(base * gpu * memory);
}

/**
 * Share the budget between videos in proportion to how much each one wants.
 * Short clips keep a floor so a four-second rooftop pass is not starved by a
 * ten-minute orbit.
 */
export function allocateBudget(videos, quality, total) {
  const rate = RATE[quality] || RATE.balanced;
  const demand = videos.map((video) => {
    const duration = trimmedDuration(video);
    const available = Math.max(1, Math.floor(duration * Math.min(video.probe.fps, 12)));
    const weight = video.settings?.weight || 1;
    return {
      id: video.id,
      available,
      floor: Math.min(MIN_PER_VIDEO, available),
      want: Math.min(available, Math.max(MIN_PER_VIDEO, Math.round(duration * rate * weight))),
    };
  });
  const floors = demand.reduce((sum, d) => sum + d.floor, 0);
  const wanted = demand.reduce((sum, d) => sum + d.want, 0);
  // Every video keeps its floor even if that exceeds the requested total.
  const spare = Math.max(0, total - floors);
  const extraWanted = Math.max(1, wanted - floors);
  const factor = Math.min(1, spare / extraWanted);
  return new Map(demand.map((d) => [d.id, d.floor + Math.floor((d.want - d.floor) * factor)]));
}

export function candidateRate(video, budget) {
  const duration = trimmedDuration(video);
  const wanted = Math.max(3, (budget / duration) * 3);
  return Math.min(video.probe.fps, wanted, MAX_CANDIDATES / duration);
}

function inputArgs(video) {
  const { trimStart = 0, trimEnd = null } = video.settings || {};
  const args = [];
  if (trimStart > 0) args.push('-ss', String(trimStart));
  if (trimEnd != null) args.push('-t', String(trimmedDuration(video)));
  args.push('-i', video.path);
  return args;
}

/** Pass 1: thumbnails and per-candidate measurements. */
export async function analyzeVideo(ffmpeg, video, budget, signal = null) {
  const rate = candidateRate(video, budget);
  const raw = await capture(ffmpeg, [
    '-v', 'error', '-an', '-sn', '-dn', ...inputArgs(video),
    '-vf', `fps=${rate.toFixed(6)},scale=${THUMB_W}:${THUMB_H}:flags=area,format=gray`,
    '-f', 'rawvideo', '-',
  ], { timeoutMs: 40 * 60 * 1000, maxBytes: THUMB_BYTES * (MAX_CANDIDATES + 20), signal });
  const count = Math.floor(raw.length / THUMB_BYTES);
  if (!count) throw new Error(`No frames could be decoded from ${video.originalName}.`);
  const candidates = [];
  let previous = null;
  for (let i = 0; i < count; i += 1) {
    const thumb = raw.subarray(i * THUMB_BYTES, (i + 1) * THUMB_BYTES);
    const small = downsample(thumb, 4);
    const motion = previous ? estimateShift(previous, small, 8) : { magnitude: 0, residual: 0 };
    const number = String(i + 1).padStart(6, '0');
    candidates.push({
      index: i,
      frameId: `${video.id}_frame_${number}`,
      sourceVideoId: video.id,
      fileName: `${video.id}_frame_${number}.jpg`,
      timestampSeconds: Math.round(((video.settings?.trimStart || 0) + i / rate) * 1000) / 1000,
      thumb,
      small,
      sharpness: laplacianVariance(thumb),
      change: motion.magnitude + 0.6 * motion.residual,
      motion: motion.magnitude,
      hash: differenceHash(thumb),
      selected: false,
      duplicate: false,
      rejectedReason: null,
    });
    previous = small;
  }
  return { candidates, rate };
}

/**
 * Choose up to `budget` frames, evenly spaced in a blend of visual change (70%)
 * and time (30%); the time term keeps coverage through a static stretch.
 */
export function chooseFrames(candidates, budget) {
  const sorted = candidates.map((c) => c.sharpness).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] || 0;
  const reference = sorted[Math.floor(sorted.length * 0.95)] || median || 1;
  const totalChange = candidates.reduce((sum, c) => sum + c.change, 0);
  const slots = Math.max(1, Math.min(budget, candidates.length));
  const bins = Array.from({ length: slots }, () => []);
  let travelled = 0;
  candidates.forEach((c, i) => {
    travelled += c.change;
    const time = candidates.length > 1 ? i / (candidates.length - 1) : 0;
    const progress = totalChange > 0 ? 0.7 * (travelled / totalChange) + 0.3 * time : time;
    bins[Math.min(slots - 1, Math.floor(progress * slots))].push(c);
    c.sharpnessScore = Math.round(Math.min(1, c.sharpness / reference) * 1000) / 1000;
  });
  let gap = false;
  for (const bin of bins) {
    if (!bin.length) continue;
    let best = bin[0];
    for (const c of bin) if (c.sharpness > best.sharpness) best = c;
    const blurred = best.sharpness < median * BLUR_THRESHOLD;
    // A blurry slot is skipped, but never two in a row: losing coverage breaks
    // tracking, which is worse than one soft frame.
    if (blurred && !gap) {
      gap = true;
    } else {
      best.selected = true;
      gap = false;
    }
  }
  for (const c of candidates) {
    if (c.selected) continue;
    c.rejectedReason = c.sharpness < median * BLUR_THRESHOLD ? 'blur' : 'redundant';
  }
  return { medianSharpness: median };
}

/** An `eq()` sum as a balanced tree, which FFmpeg's expression parser accepts at any length. */
function selectExpression(indices) {
  if (indices.length === 1) return `eq(n\\,${indices[0]})`;
  const middle = indices.length >> 1;
  return `(${selectExpression(indices.slice(0, middle))}+${selectExpression(indices.slice(middle))})`;
}

/**
 * Pass 2: decode the selected frames at working resolution.
 *
 * The same `fps` filter precedes `select`, so index n here is the same frame as
 * candidate n in pass 1.
 */
export async function extractFrames({
  ffmpeg, video, candidates, rate, width, outDir, run, job, label, stage,
}) {
  const chosen = candidates.filter((c) => c.selected);
  if (!chosen.length) return 0;
  const filter = `fps=${rate.toFixed(6)},select=${selectExpression(chosen.map((c) => c.index))},`
    + `scale=${width}:-2:flags=lanczos,unsharp=5:5:0.45:5:5:0.0`;
  const temporary = `${video.id}_tmp_%06d.jpg`;
  const args = (hardware) => [
    '-y', '-an', '-sn', '-dn',
    ...(hardware ? ['-hwaccel', 'cuda'] : []),
    ...inputArgs(video),
    '-vf', filter, '-fps_mode', 'vfr', '-q:v', '2',
    join(outDir, temporary),
  ];
  const collect = async () => (await readdir(outDir))
    .filter((name) => name.startsWith(`${video.id}_tmp_`)).sort();
  let produced = [];
  if (process.env.SKYFORGE_NO_NVDEC !== '1') {
    try {
      await run(ffmpeg, args(true), outDir, job, label, { stage });
      produced = await collect();
    } catch (error) {
      if (/cancelled|timeout/i.test(error.message)) throw error;
      job.logger?.warn(`${video.id}: hardware decode failed, retrying in software.`);
    }
  }
  if (!produced.length) {
    await run(ffmpeg, args(false), outDir, job, label, { stage });
    produced = await collect();
  }
  // Output order is input order, so the k-th file is the k-th selected frame.
  for (let i = 0; i < produced.length && i < chosen.length; i += 1) {
    await rename(join(outDir, produced[i]), join(outDir, chosen[i].fileName));
  }
  for (let i = produced.length; i < chosen.length; i += 1) {
    chosen[i].selected = false;
    chosen[i].rejectedReason = 'decode-failed';
  }
  return Math.min(produced.length, chosen.length);
}
