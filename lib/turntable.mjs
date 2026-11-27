/**
 * Turntable video of a finished model.
 * ---------------------------------------------------------------------------
 * A reconstruction is hard to judge from a still, and the interactive viewer
 * needs WebGL, a GPU and a user who knows how to orbit. A 360-degree flyaround
 * encoded as an MP4 is the one artefact that plays everywhere — in the browser
 * panel, in a chat message, in a slide — and shows at a glance whether the
 * geometry and the texture came out.
 *
 * Frames are rasterised on the CPU by lib/meshrender.mjs and piped straight
 * into FFmpeg as raw RGB, so nothing is written to disk except the finished
 * MP4 and a poster still. Encoding is explicitly H.264 yuv420p with
 * +faststart: browsers will not play yuv444 or a file whose moov atom is at
 * the end of the stream.
 *
 * Every entry point here is best-effort by contract. A turntable is a nicety
 * attached to the end of a job that has already produced its real deliverable,
 * so a failure must be reported and swallowed, never propagated.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, writeFile, rm, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { encodePng } from './png.mjs';
import { loadRenderModel, renderView } from './meshrender.mjs';

export const TURNTABLE_DEFAULTS = {
  frames: 150,
  width: 1280,
  height: 720,
  fps: 30,
  elevation: 30,
  crf: 23,
  distanceScale: 2.4,
};

/**
 * Render `glbPath` as an orbiting MP4 at `outPath`.
 *
 * Resolves with `{ ok: true, ... }` on success or `{ ok: false, reason }` on
 * any failure. It never throws and never rejects: callers treat a missing
 * turntable as a cosmetic gap, not an error.
 */
export async function renderTurntable({
  glbPath,
  outPath,
  ffmpeg,
  frames = TURNTABLE_DEFAULTS.frames,
  width = TURNTABLE_DEFAULTS.width,
  height = TURNTABLE_DEFAULTS.height,
  fps = TURNTABLE_DEFAULTS.fps,
  elevation = TURNTABLE_DEFAULTS.elevation,
  crf = TURNTABLE_DEFAULTS.crf,
  distanceScale = TURNTABLE_DEFAULTS.distanceScale,
  poster = true,
  onProgress = null,
  signal = null,
} = {}) {
  const started = Date.now();
  try {
    if (!glbPath || !existsSync(glbPath)) {
      return { ok: false, reason: `model not found at ${glbPath}` };
    }
    if (!ffmpeg || !existsSync(ffmpeg)) {
      return { ok: false, reason: 'FFmpeg is not available, so the turntable cannot be encoded' };
    }
    // H.264 needs even dimensions; an odd width silently fails the encode.
    const w = Math.max(16, Math.round(width / 2) * 2);
    const h = Math.max(16, Math.round(height / 2) * 2);
    const total = Math.max(8, Math.round(frames));

    await mkdir(dirname(outPath), { recursive: true });
    const model = await loadRenderModel(glbPath);
    if (!model.faceCount) return { ok: false, reason: 'model has no faces to render' };

    const args = [
      '-hide_banner', '-loglevel', 'error', '-y',
      '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${w}x${h}`, '-r', String(fps),
      '-i', 'pipe:0',
      '-an',
      '-c:v', 'libx264',
      '-preset', 'veryfast',
      '-crf', String(crf),
      '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart',
      outPath,
    ];

    const encoder = spawn(ffmpeg, args, { stdio: ['pipe', 'ignore', 'pipe'] });
    let stderr = '';
    encoder.stderr?.on('data', (chunk) => { stderr += chunk.toString(); });

    const finished = new Promise((resolveDone) => {
      encoder.on('error', (error) => resolveDone({ code: -1, message: error.message }));
      encoder.on('close', (code) => resolveDone({ code, message: stderr.trim() }));
    });

    // Back-pressure matters: a 1280x720 RGB frame is 2.7 MB, so writing 150 of
    // them without waiting for drain would buffer 400 MB in the parent.
    const writeFrame = (buffer) => new Promise((resolveWrite, rejectWrite) => {
      if (encoder.stdin.destroyed) { rejectWrite(new Error('encoder closed early')); return; }
      encoder.stdin.write(buffer, (error) => (error ? rejectWrite(error) : resolveWrite()));
    });

    let posterBuffer = null;
    let cancelled = false;
    for (let index = 0; index < total; index += 1) {
      if (signal?.aborted) { cancelled = true; break; }
      const az = (360 * index) / total;
      const { pixels } = renderView(model, {
        width: w, height: h, az, el: elevation, distanceScale,
      });
      // A quarter turn in is a more representative thumbnail than frame zero,
      // which is always the same dead-on azimuth for every model.
      if (poster && index === Math.floor(total / 8)) posterBuffer = encodePng(pixels, w, h);
      try {
        await writeFrame(pixels);
      } catch (error) {
        encoder.stdin.destroy();
        await finished;
        return { ok: false, reason: `encoder stopped after ${index} frames: ${error.message}` };
      }
      if (onProgress && (index % 5 === 0 || index === total - 1)) {
        onProgress({ frame: index + 1, total, fraction: (index + 1) / total });
      }
    }

    encoder.stdin.end();
    const result = await finished;

    if (cancelled) {
      await rm(outPath, { force: true }).catch(() => {});
      return { ok: false, reason: 'cancelled' };
    }
    if (result.code !== 0) {
      await rm(outPath, { force: true }).catch(() => {});
      return { ok: false, reason: `FFmpeg exited ${result.code}: ${result.message || 'no output'}` };
    }
    if (!existsSync(outPath)) return { ok: false, reason: 'FFmpeg produced no file' };

    let posterPath = null;
    if (posterBuffer) {
      posterPath = outPath.replace(/\.mp4$/i, '.png');
      try { await writeFile(posterPath, posterBuffer); } catch { posterPath = null; }
    }

    const info = await stat(outPath);
    return {
      ok: true,
      path: outPath,
      posterPath,
      bytes: info.size,
      frames: total,
      width: w,
      height: h,
      fps,
      durationSec: total / fps,
      faceCount: model.faceCount,
      textured: Boolean(model.texture),
      elapsedMs: Date.now() - started,
    };
  } catch (error) {
    return { ok: false, reason: error.message };
  }
}

/**
 * Standard location of a job's turntable, so the pipeline, the server and the
 * on-demand endpoint all agree on one filename.
 */
export const turntablePath = (productsDir) => join(productsDir, 'turning.mp4');
export const turntablePosterPath = (productsDir) => join(productsDir, 'turning.png');

/**
 * Frame budget for a profile. Lower profiles are usually being run for speed,
 * and the turntable should not become a noticeable tax on a rapid pass.
 */
export function turntableSettings(quality = 'balanced') {
  if (quality === 'rapid' || quality === 'fast') {
    return { frames: 96, width: 960, height: 540, fps: 24 };
  }
  if (quality === 'detailed' || quality === 'high') {
    return { frames: 180, width: 1280, height: 720, fps: 30 };
  }
  return { frames: 144, width: 1280, height: 720, fps: 30 };
}
