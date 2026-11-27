/**
 * The confidence score and its binary record.
 *
 * The score is a RELATIVE, weighted summary of the photographic and geometric
 * evidence behind a face. It is an engineering heuristic, not a calibrated
 * probability and not a statement of dimensional accuracy.
 */

export const DEFAULT_WEIGHTS = Object.freeze({
  coverage: 0.25, // how many cameras see the face (oblique views count less)
  diversity: 0.2, // angular spread of those viewing rays
  reprojection: 0.2, // pixel consistency of nearby sparse landmarks
  support: 0.15, // dense points lying on the face
  quality: 0.1, // sharpness of the observing frames
  depth: 0.1, // agreement with the dense depth maps
});

export const COMPONENTS = Object.keys(DEFAULT_WEIGHTS);
export const UNKNOWN = 255;
/** Cameras (head-on equivalents) at which coverage is considered full. */
export const FULL_COVERAGE_CAMERAS = 6;
/** Below this share of total weight there is not enough evidence to score. */
export const MIN_AVAILABILITY = 0.5;
/** Highest score a face of each provenance class may receive. */
export const PROVENANCE_CAPS = Object.freeze([100, 100, 100, 35, 20, 25]);
export const SMOOTHED_FACTOR = 0.95;

export const BANDS = Object.freeze([
  { key: 'high', label: 'High', min: 80, max: 100 },
  { key: 'good', label: 'Good', min: 60, max: 79 },
  { key: 'moderate', label: 'Moderate', min: 40, max: 59 },
  { key: 'low', label: 'Low', min: 20, max: 39 },
  { key: 'veryLow', label: 'Very low', min: 0, max: 19 },
]);

export function normalizeWeights(raw = null) {
  const weights = { ...DEFAULT_WEIGHTS };
  for (const key of COMPONENTS) {
    const value = Number(raw?.[key]);
    if (raw && raw[key] !== '' && Number.isFinite(value) && value >= 0) weights[key] = value;
  }
  const total = COMPONENTS.reduce((sum, key) => sum + weights[key], 0);
  if (!(total > 0)) return { ...DEFAULT_WEIGHTS };
  for (const key of COMPONENTS) weights[key] /= total;
  return weights;
}

/**
 * @param {Record<string, number|null>} values 0..1 per component, null if unavailable
 * @returns {{score:number, availability:number}} score 0..100 or UNKNOWN
 */
export function combine(values, weights, minAvailability = MIN_AVAILABILITY) {
  let sum = 0;
  let available = 0;
  for (const key of COMPONENTS) {
    const value = values[key];
    if (value == null || !(value >= 0)) continue;
    sum += weights[key] * Math.min(1, value);
    available += weights[key];
  }
  // Missing evidence is never filled in: the average runs over what exists and
  // the availability says how much that was.
  if (available <= 0 || available + 1e-9 < minAvailability) return { score: UNKNOWN, availability: available };
  return { score: Math.round((sum / available) * 100), availability: available };
}

export function applyProvenance(score, code) {
  if (score === UNKNOWN) return UNKNOWN;
  const adjusted = code === 2 ? Math.round(score * SMOOTHED_FACTOR) : score;
  return Math.min(adjusted, PROVENANCE_CAPS[code] ?? 100);
}

// ---------------------------------------------------------------------------
// Binary schema: "SFCF", u16 version, u16 recordSize, u32 faceCount, u32 flags,
// faceCount records, then (flags & 1) u32 videoMask[faceCount].
// ---------------------------------------------------------------------------

export const RECORD_SIZE = 16;
export const HEADER_SIZE = 16;
export const RECORD_FIELDS = Object.freeze([
  { offset: 0, type: 'u8', name: 'score', unit: '0..100, 255 unknown' },
  { offset: 1, type: 'u8', name: 'availability', unit: 'percent of total weight with evidence' },
  { offset: 2, type: 'u8', name: 'coverage', unit: '0..100, 255 unavailable' },
  { offset: 3, type: 'u8', name: 'diversity', unit: '0..100, 255 unavailable' },
  { offset: 4, type: 'u8', name: 'reprojection', unit: '0..100, 255 unavailable' },
  { offset: 5, type: 'u8', name: 'support', unit: '0..100, 255 unavailable' },
  { offset: 6, type: 'u8', name: 'quality', unit: '0..100, 255 unavailable' },
  { offset: 7, type: 'u8', name: 'depth', unit: '0..100, 255 unavailable' },
  { offset: 8, type: 'u16', name: 'cameras', unit: 'unoccluded front-facing observers, 65535 unavailable' },
  { offset: 10, type: 'u16', name: 'reprojectionError', unit: 'median pixels x100, 65535 unavailable' },
  { offset: 12, type: 'u8', name: 'videos', unit: 'distinct supporting source videos' },
  { offset: 13, type: 'u8', name: 'provenance', unit: 'provenance code' },
  { offset: 14, type: 'u8', name: 'incidence', unit: 'mean degrees off the surface normal, 255 unavailable' },
  { offset: 15, type: 'u8', name: 'spread', unit: 'angular spread of viewing rays in degrees, 255 unavailable' },
]);

export function createRecords(faces) {
  const buffer = Buffer.alloc(HEADER_SIZE + faces * RECORD_SIZE + faces * 4);
  buffer.write('SFCF', 0, 'ascii');
  buffer.writeUInt16LE(1, 4);
  buffer.writeUInt16LE(RECORD_SIZE, 6);
  buffer.writeUInt32LE(faces, 8);
  buffer.writeUInt32LE(1, 12);
  return buffer;
}

export function writeRecord(buffer, face, record) {
  const at = HEADER_SIZE + face * RECORD_SIZE;
  const byte = (v) => (v == null || v < 0 ? UNKNOWN : Math.min(254, Math.round(v)));
  const percent = (v) => (v == null || v < 0 ? UNKNOWN : Math.round(Math.min(1, v) * 100));
  buffer[at] = record.score;
  buffer[at + 1] = Math.round(record.availability * 100);
  buffer[at + 2] = percent(record.coverage);
  buffer[at + 3] = percent(record.diversity);
  buffer[at + 4] = percent(record.reprojection);
  buffer[at + 5] = percent(record.support);
  buffer[at + 6] = percent(record.quality);
  buffer[at + 7] = percent(record.depth);
  buffer.writeUInt16LE(record.cameras == null ? 65535 : Math.min(65534, record.cameras), at + 8);
  buffer.writeUInt16LE(record.reprojectionError == null || record.reprojectionError < 0 ? 65535 : Math.min(65534, Math.round(record.reprojectionError * 100)), at + 10);
  buffer[at + 12] = record.videos || 0;
  buffer[at + 13] = record.provenance || 0;
  buffer[at + 14] = byte(record.incidence);
  buffer[at + 15] = byte(record.spread);
  const faces = buffer.readUInt32LE(8);
  buffer.writeUInt32LE((record.videoMask || 0) >>> 0, HEADER_SIZE + faces * RECORD_SIZE + face * 4);
}

export function recordCount(buffer) {
  if (buffer.toString('ascii', 0, 4) !== 'SFCF') throw new Error('Not a confidence file.');
  return buffer.readUInt32LE(8);
}

export function readRecord(buffer, face) {
  const at = HEADER_SIZE + face * RECORD_SIZE;
  const optional = (v) => (v === UNKNOWN ? null : v);
  const cameras = buffer.readUInt16LE(at + 8);
  const error = buffer.readUInt16LE(at + 10);
  const faces = buffer.readUInt32LE(8);
  return {
    score: buffer[at],
    availability: buffer[at + 1] / 100,
    coverage: optional(buffer[at + 2]),
    diversity: optional(buffer[at + 3]),
    reprojection: optional(buffer[at + 4]),
    support: optional(buffer[at + 5]),
    quality: optional(buffer[at + 6]),
    depth: optional(buffer[at + 7]),
    cameras: cameras === 65535 ? null : cameras,
    reprojectionError: error === 65535 ? null : error / 100,
    videos: buffer[at + 12],
    provenance: buffer[at + 13],
    incidence: optional(buffer[at + 14]),
    spread: optional(buffer[at + 15]),
    videoMask: buffer.readUInt32LE(HEADER_SIZE + faces * RECORD_SIZE + face * 4),
  };
}
