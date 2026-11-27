/** LOD options and which compression methods this installation can really produce. */

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

export const DEFAULT_LEVELS = Object.freeze([
  { level: 1, ratio: 0.6, texture: 4096 },
  { level: 2, ratio: 0.3, texture: 2048 },
  { level: 3, ratio: 0.1, texture: 2048 },
  { level: 4, ratio: 0.03, texture: 1024 },
]);

const clamp = (value, min, max, fallback) => {
  const n = Number(value);
  return value !== '' && value != null && Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
};

export function normalizeLodSettings(raw = {}) {
  const requested = new Map((Array.isArray(raw.levels) ? raw.levels : []).map((l) => [Number(l.level), l]));
  const textureSizes = {};
  let previous = 0.95;
  const levels = DEFAULT_LEVELS.map((d) => {
    const r = requested.get(d.level) || {};
    // Each level must keep less than the one before it.
    const ratio = Math.min(previous * 0.9, clamp(r.ratio, 0.01, 0.9, d.ratio));
    previous = ratio;
    textureSizes[d.level] = Math.round(clamp(r.texture, 256, 8192, d.texture));
    return { level: d.level, ratio, enabled: r.enabled !== false };
  });
  return {
    levels,
    textureSizes,
    textureFormat: ['auto', 'jpeg', 'png', 'webp'].includes(raw.textureFormat) ? raw.textureFormat : 'auto',
    textureQuality: Math.round(clamp(raw.textureQuality, 50, 100, 90)),
    quantize: raw.quantize === true,
    // Multiplies the per-level error limits: below 1 is stricter.
    errorScale: clamp(raw.errorScale, 0.25, 4, 1),
  };
}

let webp = null;
export function compressionSupport(ffmpeg) {
  if (webp === null) {
    webp = false;
    if (ffmpeg && existsSync(ffmpeg)) {
      const probe = spawnSync(ffmpeg, ['-hide_banner', '-encoders'], { windowsHide: true, timeout: 20000 });
      webp = String(probe.stdout || '').includes('libwebp');
    }
  }
  return {
    quantization: { available: true },
    webp: webp ? { available: true } : { available: false, reason: 'the FFmpeg build has no libwebp encoder' },
    meshopt: { available: false, reason: 'no meshopt encoder is installed (three.js ships the decoder only)' },
    draco: { available: false, reason: 'no Draco encoder is installed' },
    ktx2: { available: false, reason: 'no KTX2/Basis encoder (toktx) is installed' },
  };
}
