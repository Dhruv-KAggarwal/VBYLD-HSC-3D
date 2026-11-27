/**
 * Near-duplicate frame removal.
 *
 * Buildings repeat themselves, so appearance similarity alone must never drop a
 * frame: two different windows of one facade can hash identically while being
 * exactly the parallax the solver needs. A frame is a duplicate only when all
 * three agree that the camera did not move and nothing changed:
 *   1. the perceptual hashes nearly match,
 *   2. the pixels correlate almost perfectly, and
 *   3. the best aligning translation is (almost) zero.
 */

import { correlation, downsample, estimateShift, hammingDistance } from './imaging.mjs';

export const DEDUP_DEFAULTS = Object.freeze({
  // Within one video: temporal neighbours, e.g. a hovering drone.
  window: 6,
  maxHamming: 6,
  minCorrelation: 0.975,
  maxShift: 0.012,
  // Across videos only a re-upload or re-encode of the same footage qualifies.
  crossMaxHamming: 2,
  crossMinCorrelation: 0.995,
  crossMaxShift: 0.001,
});

function isDuplicate(a, b, maxHamming, minCorrelation, maxShift) {
  if (hammingDistance(a.hash, b.hash) > maxHamming) return false;
  if (correlation(a.thumb, b.thumb) < minCorrelation) return false;
  a.small ||= downsample(a.thumb, 4);
  b.small ||= downsample(b.thumb, 4);
  return estimateShift(a.small, b.small, 4).magnitude <= maxShift;
}

/**
 * Mark duplicates among selected candidates of one video, in temporal order.
 * The sharper of a duplicate pair is the one kept.
 *
 * @returns {number} how many were marked
 */
export function markDuplicatesWithinVideo(candidates, options = {}) {
  const o = { ...DEDUP_DEFAULTS, ...options };
  const kept = [];
  let removed = 0;
  for (const frame of candidates) {
    if (!frame.selected) continue;
    let match = -1;
    for (let i = kept.length - 1; i >= Math.max(0, kept.length - o.window); i -= 1) {
      if (isDuplicate(kept[i], frame, o.maxHamming, o.minCorrelation, o.maxShift)) { match = i; break; }
    }
    if (match < 0) { kept.push(frame); continue; }
    const other = kept[match];
    const loser = frame.sharpness > other.sharpness ? other : frame;
    if (loser === other) kept[match] = frame;
    loser.selected = false;
    loser.duplicate = true;
    loser.duplicateOf = loser === other ? frame.frameId : other.frameId;
    loser.rejectedReason = 'duplicate';
    removed += 1;
  }
  return removed;
}

/** Remove frames of later videos that are the same footage as an earlier one. */
export function markDuplicatesAcrossVideos(videos, options = {}) {
  const o = { ...DEDUP_DEFAULTS, ...options };
  // Bucketing on one hash byte at a time finds every pair within
  // crossMaxHamming (<= 3) without comparing all frames with all frames.
  const buckets = Array.from({ length: 8 }, () => new Map());
  const keys = (hash) => Array.from({ length: 8 }, (_, i) => (hash[i >> 2] >>> ((i & 3) * 8)) & 255);
  let removed = 0;
  for (const candidates of videos) {
    const added = [];
    for (const frame of candidates) {
      if (!frame.selected) continue;
      const k = keys(frame.hash);
      const seen = new Set();
      let duplicate = null;
      for (let i = 0; i < 8 && !duplicate; i += 1) {
        for (const other of buckets[i].get(k[i]) || []) {
          if (seen.has(other)) continue;
          seen.add(other);
          if (isDuplicate(other, frame, o.crossMaxHamming, o.crossMinCorrelation, o.crossMaxShift)) {
            duplicate = other;
            break;
          }
        }
      }
      if (duplicate) {
        frame.selected = false;
        frame.duplicate = true;
        frame.duplicateOf = duplicate.frameId;
        frame.rejectedReason = 'duplicate-across-videos';
        removed += 1;
      } else {
        added.push([frame, k]);
      }
    }
    for (const [frame, k] of added) {
      for (let i = 0; i < 8; i += 1) {
        if (!buckets[i].has(k[i])) buckets[i].set(k[i], []);
        buckets[i].get(k[i]).push(frame);
      }
    }
  }
  return removed;
}
