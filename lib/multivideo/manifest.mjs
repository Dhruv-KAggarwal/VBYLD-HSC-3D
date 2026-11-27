/** Persistent descriptions of the inputs and frames of a reconstruction job. */

import { existsSync } from 'node:fs';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

export const CAMERA_MODELS = ['SIMPLE_RADIAL', 'RADIAL', 'OPENCV', 'OPENCV_FISHEYE', 'SIMPLE_PINHOLE'];

/** Write through a temporary file so a reader never sees half a document. */
export async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2));
  await rename(temporary, path);
}

export async function readJson(path, fallback = null) {
  if (!existsSync(path)) return fallback;
  try { return JSON.parse(await readFile(path, 'utf8')); } catch { return fallback; }
}

const number = (value, min, max) => {
  if (value === '' || value == null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : null;
};
const text = (value, length = 200) => (value == null ? '' : String(value).replace(/[\u0000-\u001f]/g, ' ').slice(0, length));

/** Clamp operator-supplied per-video settings; unknown keys are dropped. */
export function normalizeVideoSettings(raw = {}) {
  const trimStart = number(raw.trimStart, 0, 1e6) ?? 0;
  let trimEnd = number(raw.trimEnd, 0, 1e6);
  if (trimEnd != null && trimEnd <= trimStart + 0.5) trimEnd = null;
  return {
    label: text(raw.label, 80),
    altitudeM: number(raw.altitudeM, 2, 2000),
    cameraModel: CAMERA_MODELS.includes(raw.cameraModel) ? raw.cameraModel : 'SIMPLE_RADIAL',
    horizontalFovDeg: number(raw.horizontalFovDeg, 10, 170),
    weight: number(raw.weight, 0.25, 4) ?? 1,
    trimStart,
    trimEnd,
    enabled: raw.enabled !== false,
    cameraName: text(raw.cameraName, 120),
    flightNotes: text(raw.flightNotes, 500),
  };
}

export const videoId = (n) => `video_${String(n).padStart(3, '0')}`;

export const inputsPath = (work) => join(work, 'manifests', 'inputs.json');
export const framesPath = (work) => join(work, 'manifests', 'frames.json');
export const connectivityPath = (work) => join(work, 'manifests', 'connectivity.json');
export const registrationPath = (work) => join(work, 'manifests', 'registration.json');

export const writeInputs = (work, videos) => writeJson(inputsPath(work), { version: 1, videos });
export const readInputs = (work) => readJson(inputsPath(work));

/** Frame records without the in-memory thumbnails. */
export function frameRecord(candidate, size) {
  return {
    frameId: candidate.frameId,
    sourceVideoId: candidate.sourceVideoId,
    fileName: candidate.selected ? candidate.fileName : null,
    timestampSeconds: candidate.timestampSeconds,
    width: size.width,
    height: size.height,
    sharpnessScore: candidate.sharpnessScore,
    motion: Math.round(candidate.motion * 1000) / 1000,
    duplicate: candidate.duplicate,
    duplicateOf: candidate.duplicateOf || null,
    selected: candidate.selected,
    rejectedReason: candidate.rejectedReason,
  };
}

/** `video_002_frame_000042.jpg` -> `video_002`; null for single-video names. */
export function videoOfImage(name) {
  return /^(video_\d{3,})_/.exec(String(name))?.[1] || null;
}
