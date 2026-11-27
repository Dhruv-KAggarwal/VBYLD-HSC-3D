/**
 * Hybrid pair selection: temporal and loop-closure pairs inside each video,
 * retrieval-ranked pairs between videos.
 *
 * Sequential matching over a concatenated frame list would only ever connect the
 * last frame of one flight to the first of the next, which says nothing about
 * where two flights actually look at the same thing.
 */

import { writeFile } from 'node:fs/promises';
import { videoOfImage } from './manifest.mjs';

export const PROFILE_MATCHING = {
  fast: { minWindow: 5, maxWindow: 16, loops: 1, crossPerImage: 8, perVideoPair: 20 },
  balanced: { minWindow: 6, maxWindow: 20, loops: 2, crossPerImage: 14, perVideoPair: 30 },
  detailed: { minWindow: 8, maxWindow: 24, loops: 3, crossPerImage: 20, perVideoPair: 40 },
};

const keyOf = (a, b) => (a < b ? `${a}\n${b}` : `${b}\n${a}`);

export class PairSet {
  constructor() { this.map = new Map(); }

  add(a, b, why) {
    if (a === b) return false;
    const key = keyOf(a, b);
    if (this.map.has(key)) return false;
    this.map.set(key, { a, b, why });
    return true;
  }

  has(a, b) { return this.map.has(keyOf(a, b)); }

  get size() { return this.map.size; }

  values() { return [...this.map.values()]; }
}

export async function writePairs(path, pairs) {
  await writeFile(path, `${pairs.map((p) => `${p.a} ${p.b}`).join('\n')}\n`);
}

/**
 * Temporal pairs with a window that widens while the camera moves little, plus
 * power-of-two skips and retrieval-based loop closures.
 *
 * @param {Array<{id:string, frames:Array<{fileName:string, travel:number}>}>} videos
 *   `travel` is cumulative visual change, in image widths.
 */
export function intraVideoPairs(videos, profile, index = null) {
  const options = PROFILE_MATCHING[profile] || PROFILE_MATCHING.balanced;
  const pairs = new PairSet();
  const position = new Map(index?.names.map((name, i) => [name, i]) || []);
  for (const video of videos) {
    const frames = video.frames;
    for (let i = 0; i < frames.length; i += 1) {
      let reach = 0;
      for (let j = i + 1; j < frames.length && j - i <= options.maxWindow; j += 1) {
        const still = frames[j].travel - frames[i].travel < 0.6;
        if (j - i > options.minWindow && !still) break;
        pairs.add(frames[i].fileName, frames[j].fileName, 'sequential');
        reach = j - i;
      }
      for (let skip = 2; i + skip < frames.length; skip *= 2) {
        if (skip > reach) pairs.add(frames[i].fileName, frames[i + skip].fileName, 'quadratic');
      }
    }
    if (!index) continue;
    for (let i = 0; i < frames.length; i += 1) {
      const a = position.get(frames[i].fileName);
      if (a === undefined) continue;
      const far = [];
      for (let j = 0; j < frames.length; j += 1) {
        const b = position.get(frames[j].fileName);
        if (b !== undefined && Math.abs(i - j) > options.maxWindow) far.push({ j, score: index.similarity(a, b) });
      }
      far.sort((x, y) => y.score - x.score);
      for (const candidate of far.slice(0, options.loops)) {
        pairs.add(frames[i].fileName, frames[candidate.j].fileName, 'loop');
      }
    }
  }
  return pairs;
}

/**
 * For every image, its most similar images in each other video; and for every
 * pair of videos, the globally best candidates, so no video pair is left
 * untested just because a third video looks more alike.
 *
 * `expand` widens both quotas on a retry; `only` restricts to video pairs the
 * predicate accepts.
 */
export function interVideoPairs(videos, profile, index, { expand = 1, only = null, skip = null } = {}) {
  const options = PROFILE_MATCHING[profile] || PROFILE_MATCHING.balanced;
  const pairs = new PairSet();
  const position = new Map(index.names.map((name, i) => [name, i]));
  const others = Math.max(1, videos.length - 1);
  const perVideo = Math.max(2, Math.round(options.crossPerImage / others)) * expand;
  for (let x = 0; x < videos.length; x += 1) {
    for (let y = x + 1; y < videos.length; y += 1) {
      if (only && !only(videos[x].id, videos[y].id)) continue;
      const scored = [];
      for (const fa of videos[x].frames) {
        const a = position.get(fa.fileName);
        if (a === undefined) continue;
        for (const fb of videos[y].frames) {
          const b = position.get(fb.fileName);
          if (b !== undefined) scored.push([fa.fileName, fb.fileName, index.similarity(a, b)]);
        }
      }
      scored.sort((p, q) => q[2] - p[2]);
      const taken = new Map();
      let global = 0;
      for (const [a, b] of scored) {
        const forGlobal = global < options.perVideoPair * expand;
        const forA = (taken.get(a) || 0) < perVideo;
        const forB = (taken.get(b) || 0) < perVideo;
        if (!forGlobal && !forA && !forB) continue;
        if (skip?.has(a, b)) continue;
        if (pairs.add(a, b, 'retrieval')) {
          taken.set(a, (taken.get(a) || 0) + 1);
          taken.set(b, (taken.get(b) || 0) + 1);
          global += 1;
        }
      }
    }
  }
  return pairs;
}

/** Every cross-video pair; `stride` thins both sides when that is too many. */
export function exhaustiveCrossPairs(videos, { only = null, skip = null, limit = Infinity } = {}) {
  for (let stride = 1; ; stride += 1) {
    const pairs = new PairSet();
    for (let x = 0; x < videos.length; x += 1) {
      for (let y = x + 1; y < videos.length; y += 1) {
        if (only && !only(videos[x].id, videos[y].id)) continue;
        for (let i = 0; i < videos[x].frames.length; i += stride) {
          for (let j = 0; j < videos[y].frames.length; j += stride) {
            const a = videos[x].frames[i].fileName;
            const b = videos[y].frames[j].fileName;
            if (!skip?.has(a, b)) pairs.add(a, b, stride === 1 ? 'exhaustive' : 'strided');
          }
        }
      }
    }
    if (pairs.size <= limit || stride > 64) return pairs;
  }
}

export const isCrossPair = (pair) => videoOfImage(pair.a) !== videoOfImage(pair.b);

/** COLMAP flags selecting the matcher that suits the extracted descriptors. */
export function matcherArgs(learned, fallback = false) {
  if (learned) return ['--FeatureMatching.type', fallback ? 'ALIKED_BRUTEFORCE' : 'ALIKED_LIGHTGLUE'];
  // SIFT's fallback is guided matching: a second search along the epipolar
  // geometry, which recovers matches across wide viewpoint changes.
  return ['--FeatureMatching.type', 'SIFT_BRUTEFORCE', '--FeatureMatching.guided_matching', fallback ? '1' : '0'];
}
