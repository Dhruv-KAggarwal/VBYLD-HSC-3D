/**
 * Texture decoding, resizing and encoding.
 *
 * PNG and JPEG are handled in JavaScript (the project's PNG reader and jpeg-js).
 * WebP output uses the bundled FFmpeg's libwebp encoder when it is available.
 */

import { spawn } from 'node:child_process';
import jpeg from 'jpeg-js';
import { decodePng, encodePng } from './png.mjs';

export function sniffMime(bytes) {
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return 'image/png';
  if (bytes[0] === 0xff && bytes[1] === 0xd8) return 'image/jpeg';
  if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[8] === 0x57) return 'image/webp';
  return null;
}

/** @returns {{width:number,height:number,data:Uint8Array}} packed RGB, top row first */
export function decodeImage(bytes) {
  const mime = sniffMime(bytes);
  if (mime === 'image/png') return decodePng(bytes);
  if (mime === 'image/jpeg') {
    const { width, height, data } = jpeg.decode(bytes, {
      useTArray: true, formatAsRGBA: false, maxMemoryUsageInMB: 2048, maxResolutionInMP: 300,
    });
    return { width, height, data };
  }
  throw new Error('Unsupported texture format (only PNG and JPEG can be re-processed).');
}

/** Dimensions from the header alone, without decoding. */
export function imageSize(bytes) {
  const mime = sniffMime(bytes);
  if (mime === 'image/png') return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
  if (mime === 'image/jpeg') {
    let offset = 2;
    while (offset + 9 < bytes.length) {
      if (bytes[offset] !== 0xff) { offset += 1; continue; }
      const marker = bytes[offset + 1];
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { height: bytes.readUInt16BE(offset + 5), width: bytes.readUInt16BE(offset + 7) };
      }
      offset += 2 + bytes.readUInt16BE(offset + 2);
    }
  }
  return null;
}

/** Area-averaging downscale. Never upscales. */
export function resizeImage(image, maxEdge) {
  const scale = Math.min(1, maxEdge / Math.max(image.width, image.height));
  if (scale >= 1) return image;
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const out = new Uint8Array(width * height * 3);
  const sx = image.width / width;
  const sy = image.height / height;
  for (let y = 0; y < height; y += 1) {
    const y0 = Math.floor(y * sy);
    const y1 = Math.min(image.height, Math.max(y0 + 1, Math.floor((y + 1) * sy)));
    for (let x = 0; x < width; x += 1) {
      const x0 = Math.floor(x * sx);
      const x1 = Math.min(image.width, Math.max(x0 + 1, Math.floor((x + 1) * sx)));
      let r = 0;
      let g = 0;
      let b = 0;
      for (let j = y0; j < y1; j += 1) {
        for (let i = x0; i < x1; i += 1) {
          const o = (j * image.width + i) * 3;
          r += image.data[o];
          g += image.data[o + 1];
          b += image.data[o + 2];
        }
      }
      const n = (x1 - x0) * (y1 - y0);
      const o = (y * width + x) * 3;
      out[o] = r / n;
      out[o + 1] = g / n;
      out[o + 2] = b / n;
    }
  }
  return { width, height, data: out };
}

function encodeJpeg(image, quality) {
  const rgba = Buffer.alloc(image.width * image.height * 4);
  for (let i = 0, j = 0; i < image.data.length; i += 3, j += 4) {
    rgba[j] = image.data[i];
    rgba[j + 1] = image.data[i + 1];
    rgba[j + 2] = image.data[i + 2];
    rgba[j + 3] = 255;
  }
  return jpeg.encode({ data: rgba, width: image.width, height: image.height }, quality).data;
}

function encodeWebp(image, quality, ffmpeg) {
  return new Promise((resolve, reject) => {
    const child = spawn(ffmpeg, [
      '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${image.width}x${image.height}`, '-i', '-',
      '-frames:v', '1', '-c:v', 'libwebp', '-quality', String(quality), '-f', 'webp', '-',
    ], { windowsHide: true });
    const out = [];
    let error = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), 180_000);
    child.stdout.on('data', (c) => out.push(c));
    child.stderr.on('data', (c) => { error += c.toString(); });
    child.on('error', (e) => { clearTimeout(timer); reject(e); });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0 && out.length) resolve(Buffer.concat(out));
      else reject(new Error(`WebP encoding failed: ${error.slice(-300)}`));
    });
    child.stdin.on('error', () => {});
    child.stdin.end(Buffer.from(image.data.buffer, image.data.byteOffset, image.data.byteLength));
  });
}

/** @returns {Promise<{bytes:Buffer, mime:string}>} */
export async function encodeImage(image, { format = 'jpeg', quality = 90, ffmpeg = null } = {}) {
  if (format === 'png') return { bytes: encodePng(image.data, image.width, image.height), mime: 'image/png' };
  if (format === 'webp') {
    if (!ffmpeg) throw new Error('WebP output needs FFmpeg.');
    return { bytes: await encodeWebp(image, quality, ffmpeg), mime: 'image/webp' };
  }
  return { bytes: encodeJpeg(image, quality), mime: 'image/jpeg' };
}

/** Bilinear sample at glTF texture coordinates (origin top-left, clamped). */
export function sampleImage(image, u, v, out = [0, 0, 0]) {
  const x = Math.min(image.width - 1, Math.max(0, u * image.width - 0.5));
  const y = Math.min(image.height - 1, Math.max(0, v * image.height - 0.5));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  for (let k = 0; k < 3; k += 1) {
    const a = image.data[(y0 * image.width + x0) * 3 + k];
    const b = image.data[(y0 * image.width + x1) * 3 + k];
    const c = image.data[(y1 * image.width + x0) * 3 + k];
    const d = image.data[(y1 * image.width + x1) * 3 + k];
    out[k] = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
  }
  return out;
}
