/**
 * Sky masking for the feature stage.
 * ---------------------------------------------------------------------------
 * An aerial frame is often a third sky. Sky has no stable texture, so SIFT and
 * ALIKED still find thousands of keypoints in cloud edges and haze gradients,
 * the matcher pairs them across frames that share no real geometry, and the
 * mapper triangulates them into a ceiling of floating points above the scene —
 * the white haze artefact. Masking the sky out of the *input* is strictly better
 * than deleting sky-coloured faces afterwards: the bad points never enter the
 * solve, so they cannot bend the camera poses either.
 *
 * COLMAP reads a per-image mask PNG (`--ImageReader.mask_path`) where a zero
 * pixel means "ignore this pixel". We write one `<frame>.jpg.png` per frame.
 *
 * This replaces `tools/python/mask_frames.py`. It is a faithful port of that
 * script's heuristic back-end with one upgrade — a real 8-connected flood from
 * the top band, instead of the per-column run the Python used when OpenCV was
 * missing — and it needs no Python, no OpenCV and no model weights.
 */

import { readFile, writeFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import jpeg from 'jpeg-js';
import { encodePng } from './png.mjs';

/**
 * Is this pixel plausibly sky?
 *
 * Two bands, both from the original heuristic: bright-and-unsaturated with a
 * blue bias (open sky), or simply very bright and grey (blown cloud). The blue
 * bias is what keeps pale concrete roofs out of the mask — they are bright and
 * unsaturated too, but not blue.
 */
function skyPixel(r, g, b) {
  const lum = (r + g + b) / 3;
  const sat = Math.max(r, g, b) - Math.min(r, g, b);
  return (lum > 175 && sat < 46 && b >= r - 12) || (lum > 200 && sat < 30);
}

/**
 * Keep only the sky that touches the top of the frame.
 *
 * Without this, a bright rooftop or a sun-struck road passes the colour test and
 * gets masked out — exactly the geometry we are trying to reconstruct. Real sky
 * in a drone frame is connected to the upper edge; a bright roof is not.
 */
function floodFromTop(sky, width, height) {
  const ignore = new Uint8Array(width * height);
  const topBand = Math.max(1, Math.floor(height / 18));
  const queue = new Int32Array(width * height);
  let head = 0;
  let tail = 0;
  for (let y = 0; y < topBand; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      if (sky[i] && !ignore[i]) {
        ignore[i] = 1;
        queue[tail] = i;
        tail += 1;
      }
    }
  }
  while (head < tail) {
    const i = queue[head];
    head += 1;
    const x = i % width;
    const y = (i - x) / width;
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (!dx && !dy) continue;
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const ni = ny * width + nx;
        if (!sky[ni] || ignore[ni]) continue;
        ignore[ni] = 1;
        queue[tail] = ni;
        tail += 1;
      }
    }
  }
  return ignore;
}

/**
 * Morphological close (dilate then erode) with a square kernel.
 *
 * A ragged mask edge is its own source of spurious features: the boundary
 * between masked and kept pixels is a strong gradient that the detector happily
 * fires on. Closing fills the pinholes and smooths the seam.
 */
function closeMask(mask, width, height, radius = 4) {
  const dilate = (src) => {
    const out = new Uint8Array(src.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let hit = 0;
        for (let dy = -radius; dy <= radius && !hit; dy += 1) {
          const ny = y + dy;
          if (ny < 0 || ny >= height) continue;
          for (let dx = -radius; dx <= radius; dx += 1) {
            const nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            if (src[ny * width + nx]) { hit = 1; break; }
          }
        }
        out[y * width + x] = hit;
      }
    }
    return out;
  };
  // Erode = dilate the complement.
  const complement = (src) => {
    const out = new Uint8Array(src.length);
    for (let i = 0; i < src.length; i += 1) out[i] = src[i] ? 0 : 1;
    return out;
  };
  return complement(dilate(complement(dilate(mask))));
}

/**
 * Builds one COLMAP ignore-mask per frame.
 *
 * Best-effort by contract: a frame that cannot be decoded simply gets no mask,
 * and COLMAP then uses the whole image. Masking is an improvement, never a
 * prerequisite, so nothing here may fail the run.
 *
 * @param {string} framesDir  directory of frame_*.jpg
 * @param {string} outDir     directory to write <name>.png into
 * @param {object} [opts]
 * @param {number} [opts.maxSkyFraction=0.75] refuse a mask that would hide most
 *   of the frame — that means the colour test misfired (fog, a white roof filling
 *   the view, an overexposed pass), and masking it would leave nothing to match.
 * @param {object} [logger]
 * @returns {Promise<{written:number, frames:number, skipped:number, meanSkyPercent:number}>}
 */
export async function buildSkyMasks(framesDir, outDir, opts = {}, logger = null) {
  const maxSkyFraction = opts.maxSkyFraction ?? 0.75;
  const names = (await readdir(framesDir)).filter((n) => /\.jpe?g$/i.test(n)).sort();
  let written = 0;
  let skipped = 0;
  let skySum = 0;
  for (const name of names) {
    try {
      const raw = await readFile(join(framesDir, name));
      const { width, height, data } = jpeg.decode(raw, { useTArray: true, formatAsRGBA: false });
      const cells = width * height;
      const sky = new Uint8Array(cells);
      for (let i = 0; i < cells; i += 1) {
        const o = i * 3;
        if (skyPixel(data[o], data[o + 1], data[o + 2])) sky[i] = 1;
      }
      let ignore = floodFromTop(sky, width, height);
      let hidden = 0;
      for (let i = 0; i < cells; i += 1) hidden += ignore[i];
      if (!hidden) { skipped += 1; continue; }
      if (hidden / cells > maxSkyFraction) {
        skipped += 1;
        continue;
      }
      ignore = closeMask(ignore, width, height, 4);
      hidden = 0;
      for (let i = 0; i < cells; i += 1) hidden += ignore[i];
      skySum += hidden / cells;
      const rgb = Buffer.alloc(cells * 3);
      for (let i = 0; i < cells; i += 1) {
        const v = ignore[i] ? 0 : 255;
        rgb[i * 3] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
      }
      await writeFile(join(outDir, `${name}.png`), encodePng(rgb, width, height));
      written += 1;
    } catch (error) {
      skipped += 1;
      logger?.warn?.(`Sky mask skipped for ${name}: ${error.message}`);
    }
  }
  const meanSkyPercent = written ? Math.round((skySum / written) * 100) : 0;
  return { written, frames: names.length, skipped, meanSkyPercent };
}

export const _internals = { skyPixel, floodFromTop, closeMask };
