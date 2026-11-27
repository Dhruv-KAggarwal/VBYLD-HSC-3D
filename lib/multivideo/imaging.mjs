/** Greyscale thumbnail measurements shared by frame selection and quality checks. */

export const THUMB_W = 160;
export const THUMB_H = 90;
export const THUMB_BYTES = THUMB_W * THUMB_H;

/** Variance of the 4-neighbour Laplacian: the standard blur metric. */
export function laplacianVariance(gray, width = THUMB_W, height = THUMB_H) {
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - width] - gray[i + width];
      sum += lap;
      sumSq += lap * lap;
      count += 1;
    }
  }
  if (!count) return 0;
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

/** Box-average a thumbnail down by an integer factor. */
export function downsample(gray, factor, width = THUMB_W, height = THUMB_H) {
  const w = Math.floor(width / factor);
  const h = Math.floor(height / factor);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let sum = 0;
      for (let dy = 0; dy < factor; dy += 1) {
        for (let dx = 0; dx < factor; dx += 1) sum += gray[(y * factor + dy) * width + x * factor + dx];
      }
      out[y * w + x] = sum / (factor * factor);
    }
  }
  return { data: out, width: w, height: h };
}

/**
 * Dominant image translation between two small images, by exhaustive search.
 *
 * `magnitude` is the shift as a fraction of image width and `residual` is what
 * remains after compensating it (0..1). Together they separate "the camera
 * panned" from "the content changed", which a plain frame difference cannot.
 */
export function estimateShift(a, b, maxShift = 8) {
  const { width: w, height: h } = a;
  let best = Infinity;
  let bx = 0;
  let by = 0;
  for (let dy = -maxShift; dy <= maxShift; dy += 1) {
    for (let dx = -maxShift; dx <= maxShift; dx += 1) {
      let sum = 0;
      let count = 0;
      for (let y = Math.max(0, -dy); y < Math.min(h, h - dy); y += 1) {
        for (let x = Math.max(0, -dx); x < Math.min(w, w - dx); x += 1) {
          sum += Math.abs(a.data[y * w + x] - b.data[(y + dy) * w + x + dx]);
          count += 1;
        }
      }
      if (!count) continue;
      // A slight preference for small shifts breaks ties on flat content.
      const score = sum / count + 0.02 * Math.hypot(dx, dy);
      if (score < best) { best = score; bx = dx; by = dy; }
    }
  }
  return { dx: bx, dy: by, magnitude: Math.hypot(bx, by) / w, residual: Math.min(1, best / 255) };
}

/** 64-bit difference hash as two unsigned 32-bit words. */
export function differenceHash(gray, width = THUMB_W, height = THUMB_H) {
  const cells = new Float32Array(9 * 8);
  for (let cy = 0; cy < 8; cy += 1) {
    for (let cx = 0; cx < 9; cx += 1) {
      const x0 = Math.floor((cx * width) / 9);
      const x1 = Math.floor(((cx + 1) * width) / 9);
      const y0 = Math.floor((cy * height) / 8);
      const y1 = Math.floor(((cy + 1) * height) / 8);
      let sum = 0;
      for (let y = y0; y < y1; y += 1) for (let x = x0; x < x1; x += 1) sum += gray[y * width + x];
      cells[cy * 9 + cx] = sum / ((x1 - x0) * (y1 - y0));
    }
  }
  const words = [0, 0];
  for (let bit = 0; bit < 64; bit += 1) {
    const x = bit % 8;
    const y = (bit / 8) | 0;
    if (cells[y * 9 + x] > cells[y * 9 + x + 1]) words[bit >> 5] |= 1 << (bit & 31);
  }
  return [words[0] >>> 0, words[1] >>> 0];
}

function popcount(value) {
  let v = value - ((value >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

export function hammingDistance(a, b) {
  return popcount((a[0] ^ b[0]) >>> 0) + popcount((a[1] ^ b[1]) >>> 0);
}

/** Zero-mean normalised cross-correlation of two equally sized images. */
export function correlation(a, b) {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i += 1) { ma += a[i]; mb += b[i]; }
  ma /= n;
  mb /= n;
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < n; i += 1) {
    const x = a[i] - ma;
    const y = b[i] - mb;
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  if (aa <= 0 || bb <= 0) return aa === bb && Math.abs(ma - mb) < 1 ? 1 : 0;
  return ab / Math.sqrt(aa * bb);
}
