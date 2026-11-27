/**
 * Carry confidence and provenance from the canonical mesh onto a simplified one.
 *
 * Nothing is re-analysed. Each simplified face samples several points on itself,
 * finds the canonical face nearest each, and takes a deliberately cautious
 * aggregate: the mean of the known scores (unknown if most samples are unknown)
 * and the WORST provenance among them.
 */

import { HEADER_SIZE, RECORD_SIZE, UNKNOWN, createRecords, recordCount } from '../confidence/scoring.mjs';

const SAMPLES = [[1 / 3, 1 / 3, 1 / 3], [0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]];

/** For every level face, the canonical face under each sample point. */
export function nearestSources(level, bvh) {
  const faces = level.indices.length / 3;
  const nearest = new Int32Array(faces * SAMPLES.length);
  let hint = -1;
  for (let f = 0; f < faces; f += 1) {
    for (let s = 0; s < SAMPLES.length; s += 1) {
      const p = [0, 1, 2].map((k) => SAMPLES[s].reduce(
        (sum, w, corner) => sum + w * level.positions[level.indices[f * 3 + corner] * 3 + k], 0,
      ));
      const hit = bvh.closestPoint(p[0], p[1], p[2], Infinity, hint);
      hint = hit ? hit.face : -1;
      nearest[f * SAMPLES.length + s] = hint;
    }
  }
  return { nearest, perFace: SAMPLES.length };
}

/** @param {Int32Array} order level face for each output face */
export function transferProvenance(sources, order, codes) {
  const out = new Uint8Array(order.length);
  const source = new Int32Array(order.length);
  for (let i = 0; i < order.length; i += 1) {
    const f = order[i];
    let worst = 0;
    for (let s = 0; s < sources.perFace; s += 1) {
      const c = sources.nearest[f * sources.perFace + s];
      if (c >= 0 && codes[c] > worst) worst = codes[c];
    }
    out[i] = worst;
    source[i] = sources.nearest[f * sources.perFace];
  }
  return { codes: out, source };
}

export function transferConfidence(sources, order, records) {
  const canonical = recordCount(records);
  const out = createRecords(order.length);
  const videoBase = HEADER_SIZE + canonical * RECORD_SIZE;
  const outBase = HEADER_SIZE + order.length * RECORD_SIZE;
  for (let i = 0; i < order.length; i += 1) {
    const f = order[i];
    const first = sources.nearest[f * sources.perFace];
    if (first < 0) { out[HEADER_SIZE + i * RECORD_SIZE] = UNKNOWN; continue; }
    // Component values come from the centre sample; the headline is aggregated.
    records.copy(out, HEADER_SIZE + i * RECORD_SIZE, HEADER_SIZE + first * RECORD_SIZE, HEADER_SIZE + (first + 1) * RECORD_SIZE);
    let sum = 0;
    let known = 0;
    let total = 0;
    let worst = 0;
    let availability = 0;
    let mask = 0;
    for (let s = 0; s < sources.perFace; s += 1) {
      const c = sources.nearest[f * sources.perFace + s];
      if (c < 0) continue;
      const at = HEADER_SIZE + c * RECORD_SIZE;
      total += 1;
      availability += records[at + 1];
      worst = Math.max(worst, records[at + 13]);
      mask |= records.readUInt32LE(videoBase + c * 4);
      if (records[at] !== UNKNOWN) { sum += records[at]; known += 1; }
    }
    const at = HEADER_SIZE + i * RECORD_SIZE;
    out[at] = known * 2 >= total && known ? Math.round(sum / known) : UNKNOWN;
    out[at + 1] = Math.round(availability / Math.max(1, total));
    out[at + 13] = worst;
    out.writeUInt32LE(mask >>> 0, outBase + i * 4);
  }
  return out;
}
