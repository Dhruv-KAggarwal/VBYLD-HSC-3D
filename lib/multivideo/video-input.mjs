/**
 * Source-video inspection: FFprobe validation, metadata, quality indicators and
 * thumbnails. Everything here reads the file through the installed FFmpeg, so
 * "which formats are supported" is simply "whatever this FFmpeg can decode".
 */

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync } from 'node:fs';
import { laplacianVariance } from './imaging.mjs';

const STILL_CODECS = new Set(['png', 'bmp', 'gif', 'tiff', 'webp', 'jpegls', 'targa', 'ppm']);

/** Run a binary and collect stdout, with a hard timeout. */
export function capture(command, args, { timeoutMs = 120_000, maxBytes = 256 * 1024 * 1024, signal = null } = {}) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new Error('Operation was cancelled.')); return; }
    const child = spawn(command, args, { windowsHide: true });
    const chunks = [];
    let length = 0;
    let stderr = '';
    let settled = false;
    const finish = (error, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      if (error) reject(error);
      else resolve(value);
    };
    const onAbort = () => { child.kill('SIGKILL'); finish(new Error('Operation was cancelled.')); };
    signal?.addEventListener('abort', onAbort);
    const timer = setTimeout(() => { child.kill('SIGKILL'); finish(new Error(`${command} timed out.`)); }, timeoutMs);
    child.stdout.on('data', (chunk) => {
      length += chunk.length;
      if (length > maxBytes) { child.kill('SIGKILL'); finish(new Error(`${command} produced too much output.`)); return; }
      chunks.push(chunk);
    });
    child.stderr.on('data', (chunk) => { stderr = (stderr + chunk.toString()).slice(-4000); });
    child.on('error', (error) => finish(error));
    child.on('close', (code) => {
      if (code === 0) finish(null, Buffer.concat(chunks));
      else finish(new Error(stderr.trim().split('\n').slice(-3).join(' ') || `${command} exited with code ${code}`));
    });
  });
}

function parseRate(value) {
  const [num, den] = String(value || '0/1').split('/').map(Number);
  const rate = num / (den || 1);
  return Number.isFinite(rate) && rate > 0 ? rate : 0;
}

/**
 * Validate a file as a usable video and describe it.
 *
 * `width`/`height` are the DISPLAY dimensions: FFmpeg auto-rotates on decode,
 * so a portrait phone clip stored sideways is reported the way its frames will
 * actually come out.
 */
export async function probeMedia(ffprobe, path) {
  let json;
  try {
    const out = await capture(ffprobe, ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', path], { timeoutMs: 90_000 });
    json = JSON.parse(out.toString('utf8'));
  } catch (error) {
    // The server's storage path is no business of whoever uploaded the file.
    return { ok: false, error: `FFprobe could not read this file: ${error.message.split(path).join('the file')}` };
  }
  const stream = (json.streams || []).find((s) => s.codec_type === 'video' && !s.disposition?.attached_pic);
  if (!stream) return { ok: false, error: 'The file contains no video stream.' };
  if (STILL_CODECS.has(stream.codec_name) && Number(stream.nb_frames || 1) <= 1) {
    return { ok: false, error: 'This is a still image, not a video.' };
  }
  const duration = Number(json.format?.duration) || Number(stream.duration) || 0;
  if (!(duration >= 1)) return { ok: false, error: 'The video is shorter than one second or reports no duration.' };
  const fps = parseRate(stream.avg_frame_rate) || parseRate(stream.r_frame_rate) || 24;
  const rotation = Number(
    (stream.side_data_list || []).find((d) => d.rotation !== undefined)?.rotation
    ?? stream.tags?.rotate ?? 0,
  ) || 0;
  const sideways = Math.abs(rotation) % 180 === 90;
  const codedWidth = Number(stream.width) || 0;
  const codedHeight = Number(stream.height) || 0;
  if (!codedWidth || !codedHeight) return { ok: false, error: 'The video reports no resolution.' };
  const tags = { ...(json.format?.tags || {}), ...(stream.tags || {}) };
  const tag = (...names) => {
    for (const name of names) {
      const key = Object.keys(tags).find((k) => k.toLowerCase() === name);
      if (key) return String(tags[key]).slice(0, 200);
    }
    return null;
  };
  return {
    ok: true,
    durationSeconds: duration,
    width: sideways ? codedHeight : codedWidth,
    height: sideways ? codedWidth : codedHeight,
    fps: Math.round(fps * 1000) / 1000,
    codec: stream.codec_name || 'unknown',
    container: json.format?.format_name || null,
    bitrate: Number(json.format?.bit_rate) || null,
    rotation,
    totalFrames: Number(stream.nb_frames) || Math.round(duration * fps),
    pixelFormat: stream.pix_fmt || null,
    metadata: {
      creationTime: tag('creation_time'),
      make: tag('make', 'com.apple.quicktime.make'),
      model: tag('model', 'com.apple.quicktime.model'),
      encoder: tag('encoder'),
    },
  };
}

export function hashFile(path) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    createReadStream(path)
      .on('data', (chunk) => hash.update(chunk))
      .on('error', reject)
      .on('end', () => resolve(hash.digest('hex')));
  });
}

export async function makeThumbnail(ffmpeg, video, out, atSeconds = 1) {
  await capture(ffmpeg, [
    '-v', 'error', '-y', '-ss', String(Math.max(0, atSeconds)), '-i', video,
    '-frames:v', '1', '-vf', 'scale=320:-2', '-q:v', '4', out,
  ], { timeoutMs: 90_000 });
  return existsSync(out);
}

const SAMPLE_W = 320;
const SAMPLE_H = 180;

/**
 * Cheap quality indicator from a handful of frames spread across the clip.
 *
 * It grades what hurts reconstruction and can be seen without solving anything:
 * blur, exposure, resolution and bitrate. It is an indicator, not a prediction
 * of whether the video will register against the others.
 */
export async function assessQuality(ffmpeg, video, probe, samples = 6) {
  const frames = [];
  for (let i = 0; i < samples; i += 1) {
    const time = probe.durationSeconds * ((i + 0.5) / samples);
    try {
      const raw = await capture(ffmpeg, [
        '-v', 'error', '-ss', time.toFixed(3), '-i', video, '-frames:v', '1',
        '-vf', `scale=${SAMPLE_W}:${SAMPLE_H},format=gray`, '-f', 'rawvideo', '-',
      ], { timeoutMs: 60_000 });
      if (raw.length >= SAMPLE_W * SAMPLE_H) frames.push(raw.subarray(0, SAMPLE_W * SAMPLE_H));
    } catch { /* a bad seek is not a quality verdict */ }
  }
  if (!frames.length) return { rating: 'unknown', reasons: ['No sample frames could be decoded.'], samples: 0 };
  const sharpness = frames.map((f) => laplacianVariance(f, SAMPLE_W, SAMPLE_H)).sort((a, b) => a - b);
  let dark = 0;
  let bright = 0;
  let mean = 0;
  for (const frame of frames) {
    let sum = 0;
    for (let i = 0; i < frame.length; i += 1) {
      sum += frame[i];
      if (frame[i] < 16) dark += 1;
      else if (frame[i] > 239) bright += 1;
    }
    mean += sum / frame.length;
  }
  mean /= frames.length;
  const pixels = frames.length * SAMPLE_W * SAMPLE_H;
  const medianSharpness = sharpness[Math.floor(sharpness.length / 2)];
  const reasons = [];
  let penalty = 0;
  if (medianSharpness < 40) { reasons.push('Frames look soft or motion-blurred.'); penalty += 2; } else if (medianSharpness < 110) { reasons.push('Moderate sharpness.'); penalty += 1; }
  if (mean < 50 || dark / pixels > 0.35) { reasons.push('Underexposed or large black areas.'); penalty += 1; }
  if (mean > 205 || bright / pixels > 0.3) { reasons.push('Overexposed or large blown-out areas.'); penalty += 1; }
  if (Math.min(probe.width, probe.height) < 720) { reasons.push('Below 720p; fine detail will be limited.'); penalty += 1; }
  if (probe.bitrate && probe.bitrate / (probe.width * probe.height * probe.fps) < 0.04) {
    reasons.push('Low bitrate for the resolution; compression artefacts are likely.');
    penalty += 1;
  }
  return {
    rating: penalty >= 3 ? 'poor' : penalty >= 1 ? 'fair' : 'good',
    reasons,
    samples: frames.length,
    sharpness: Math.round(medianSharpness),
    meanBrightness: Math.round(mean),
  };
}

/** Extension safe to put on disk; the original name is only ever metadata. */
export function safeExtension(name) {
  const match = /\.([A-Za-z0-9]{1,5})$/.exec(String(name || ''));
  return match ? `.${match[1].toLowerCase()}` : '.mp4';
}

export function displayName(name) {
  // Control characters and path separators stripped; this is shown, never used
  // as a path.
  return String(name || 'video').replace(/[\u0000-\u001f\\/]/g, '_').slice(-180) || 'video';
}
