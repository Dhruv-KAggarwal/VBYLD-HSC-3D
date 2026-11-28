/**
 * The observed surface seen from straight above: a height grid, the bare ground
 * beneath it, the building footprints standing on that ground, and a top-down
 * colour image used to paint patches of ground the drone could not see.
 */

import { connectedComponents, percentile } from '../buildings.mjs';
import { faceColor } from './observed.mjs';

/** Rasterise kept faces into XZ, keeping the highest surface per sample. */
function rasterTop(mesh, textures, world, keep, grid, perCell) {
  const width = grid.cols * perCell;
  const height = grid.rows * perCell;
  const step = grid.cell / perCell;
  const top = new Float32Array(width * height).fill(-Infinity);
  const rgb = new Uint8Array(width * height * 3);
  const colour = [0, 0, 0];
  const faces = mesh.indices.length / 3;
  for (let f = 0; f < faces; f += 1) {
    if (!keep[f]) continue;
    const ids = [mesh.indices[f * 3] * 3, mesh.indices[f * 3 + 1] * 3, mesh.indices[f * 3 + 2] * 3];
    const x = ids.map((i) => (world[i] - grid.minX) / step);
    const z = ids.map((i) => (world[i + 2] - grid.minZ) / step);
    const x0 = Math.max(0, Math.floor(Math.min(...x)));
    const x1 = Math.min(width - 1, Math.ceil(Math.max(...x)));
    const z0 = Math.max(0, Math.floor(Math.min(...z)));
    const z1 = Math.min(height - 1, Math.ceil(Math.max(...z)));
    if (x0 > x1 || z0 > z1) continue;
    const area = (x[1] - x[0]) * (z[2] - z[0]) - (z[1] - z[0]) * (x[2] - x[0]);
    if (Math.abs(area) < 1e-9) continue;
    for (let row = z0; row <= z1; row += 1) {
      for (let col = x0; col <= x1; col += 1) {
        const px = col + 0.5;
        const pz = row + 0.5;
        const wc = ((x[1] - x[0]) * (pz - z[0]) - (z[1] - z[0]) * (px - x[0])) / area;
        const wb = ((px - x[0]) * (z[2] - z[0]) - (pz - z[0]) * (x[2] - x[0])) / area;
        const wa = 1 - wb - wc;
        if (wa < -0.02 || wb < -0.02 || wc < -0.02) continue;
        const y = wa * world[ids[0] + 1] + wb * world[ids[1] + 1] + wc * world[ids[2] + 1];
        const at = row * width + col;
        if (y <= top[at]) continue;
        top[at] = y;
        if (faceColor(mesh, textures, f, wa, wb, wc, colour)) rgb.set(colour, at * 3);
      }
    }
  }
  return { width, height, top, rgb };
}

/** Cells within `radius` steps of a set cell. */
export function dilate(mask, cols, rows, radius) {
  let current = mask;
  for (let pass = 0; pass < radius; pass += 1) {
    const next = Uint8Array.from(current);
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        const i = y * cols + x;
        if (current[i]) continue;
        if ((x > 0 && current[i - 1]) || (x < cols - 1 && current[i + 1])
          || (y > 0 && current[i - cols]) || (y < rows - 1 && current[i + cols])) next[i] = 1;
      }
    }
    current = next;
  }
  return current;
}

const invert = (mask) => Uint8Array.from(mask, (v) => (v ? 0 : 1));
const erode = (mask, cols, rows, radius) => invert(dilate(invert(mask), cols, rows, radius));

/** Push-pull: fill invalid texels of a multi-channel image from coarser levels. */
export function pushPull(values, valid, width, height, channels) {
  if (width <= 1 && height <= 1) return values;
  const w = Math.ceil(width / 2);
  const h = Math.ceil(height / 2);
  const coarse = new Float32Array(w * h * channels);
  const count = new Uint8Array(w * h);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      if (!valid[i]) continue;
      const j = (y >> 1) * w + (x >> 1);
      for (let c = 0; c < channels; c += 1) coarse[j * channels + c] += values[i * channels + c];
      count[j] += 1;
    }
  }
  for (let j = 0; j < w * h; j += 1) {
    if (count[j]) for (let c = 0; c < channels; c += 1) coarse[j * channels + c] /= count[j];
  }
  const filled = pushPull(coarse, count, w, h, channels);
  for (let y = 0; y < height; y += 1) {
    const gy = Math.max(0, Math.min(h - 1, (y - 0.5) / 2));
    const y0 = Math.floor(gy);
    const y1 = Math.min(h - 1, y0 + 1);
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      if (valid[i]) continue;
      const gx = Math.max(0, Math.min(w - 1, (x - 0.5) / 2));
      const x0 = Math.floor(gx);
      const x1 = Math.min(w - 1, x0 + 1);
      const fx = gx - x0;
      const fy = gy - y0;
      for (let c = 0; c < channels; c += 1) {
        const a = filled[(y0 * w + x0) * channels + c];
        const b = filled[(y0 * w + x1) * channels + c];
        const d = filled[(y1 * w + x0) * channels + c];
        const e = filled[(y1 * w + x1) * channels + c];
        values[i * channels + c] = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
      }
    }
  }
  return values;
}

/**
 * Colour for unseen ground: the smooth colour of what surrounds it, plus the
 * fine grain borrowed from a seen patch nearby, so it does not read as a blur.
 */
function paintGround(raster, groundSample) {
  const { width, height, rgb } = raster;
  const total = width * height;
  const smooth = pushPull(Float32Array.from(rgb), groundSample, width, height, 3);
  // Local mean of the seen pixels, to separate grain from colour.
  const sum = new Float64Array((width + 1) * (height + 1) * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      for (let c = 0; c < 4; c += 1) {
        const value = groundSample[i] ? (c < 3 ? rgb[i * 3 + c] : 1) : 0;
        const s = (y + 1) * (width + 1) + x + 1;
        sum[s * 4 + c] = value + sum[(s - 1) * 4 + c] + sum[(s - width - 1) * 4 + c] - sum[(s - width - 2) * 4 + c];
      }
    }
  }
  const boxMean = (x, y, c) => {
    const at = (px, py, n) => sum[(py * (width + 1) + px) * 4 + n];
    const x0 = Math.max(0, x - 4);
    const y0 = Math.max(0, y - 4);
    const x1 = Math.min(width, x + 5);
    const y1 = Math.min(height, y + 5);
    const box = (n) => at(x1, y1, n) - at(x0, y1, n) - at(x1, y0, n) + at(x0, y0, n);
    return box(c) / (box(3) || 1);
  };
  const out = new Uint8Array(total * 3);
  const offsets = [[23, 7], [-19, 13], [11, -29], [-31, -17], [47, 3], [-5, 53], [71, -41], [-67, 37]];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      if (groundSample[i]) { out.set(rgb.subarray(i * 3, i * 3 + 3), i * 3); continue; }
      let source = -1;
      for (const [dx, dy] of offsets) {
        const sx = x + dx;
        const sy = y + dy;
        if (sx >= 0 && sy >= 0 && sx < width && sy < height && groundSample[sy * width + sx]) { source = sy * width + sx; break; }
      }
      for (let c = 0; c < 3; c += 1) {
        const grain = source < 0 ? 0 : rgb[source * 3 + c] - boxMean(source % width, (source / width) | 0, c);
        out[i * 3 + c] = Math.max(0, Math.min(255, smooth[i * 3 + c] + grain * 0.7));
      }
    }
  }
  return { width, height, data: out };
}

/** Sliding minimum (sign 1) or maximum (sign -1) over a square window. */
function slide(values, cols, rows, radius, sign) {
  const pass = (source, count, length, indexOf) => {
    const out = Float32Array.from(source);
    const queue = new Int32Array(length);
    for (let line = 0; line < count; line += 1) {
      let head = 0;
      let tail = 0;
      let next = 0;
      for (let i = 0; i < length; i += 1) {
        for (; next < Math.min(length, i + radius + 1); next += 1) {
          const v = sign * source[indexOf(line, next)];
          while (tail > head && sign * source[indexOf(line, queue[tail - 1])] >= v) tail -= 1;
          queue[tail] = next;
          tail += 1;
        }
        while (queue[head] < i - radius) head += 1;
        out[indexOf(line, i)] = source[indexOf(line, queue[head])];
      }
    }
    return out;
  };
  const across = pass(values, rows, cols, (line, i) => line * cols + i);
  return pass(across, cols, rows, (line, i) => i * cols + line);
}

/**
 * Bare ground: a morphological opening wider than a building. Unlike a low
 * percentile over a window, it follows sloping ground instead of calling the
 * uphill side of a slope a structure.
 */
function groundModel(height, seen, cols, rows, cell) {
  // A median first, so stray points below the surface do not drag the ground down.
  const despiked = new Float32Array(cols * rows);
  const window = [];
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      if (!seen[y * cols + x]) continue;
      window.length = 0;
      for (let dy = -2; dy <= 2; dy += 1) {
        for (let dx = -2; dx <= 2; dx += 1) {
          if (x + dx < 0 || x + dx >= cols || y + dy < 0 || y + dy >= rows) continue;
          const i = (y + dy) * cols + x + dx;
          if (seen[i]) window.push(height[i]);
        }
      }
      despiked[y * cols + x] = window.sort((a, b) => a - b)[window.length >> 1];
    }
  }
  const surface = pushPull(despiked, seen, cols, rows, 1);
  const radius = Math.max(4, Math.round(24 / cell));
  return slide(slide(surface, cols, rows, radius, 1), cols, rows, radius, -1);
}

/**
 * @param {object} region  { minX, minZ, maxX, maxZ } in metres
 */
export function analyseTop(mesh, textures, world, keep, region) {
  const extent = Math.max(region.maxX - region.minX, region.maxZ - region.minZ);
  const cell = Math.max(0.6, extent / 900);
  const grid = {
    cell,
    minX: region.minX,
    minZ: region.minZ,
    cols: Math.ceil((region.maxX - region.minX) / cell),
    rows: Math.ceil((region.maxZ - region.minZ) / cell),
  };
  const { cols, rows } = grid;
  const total = cols * rows;
  const perCell = Math.max(1, Math.min(3, Math.floor(1800 / Math.max(cols, rows))));
  const raster = rasterTop(mesh, textures, world, keep, grid, perCell);

  const height = new Float32Array(total).fill(Number.NaN);
  const seen = new Uint8Array(total);
  const green = new Uint8Array(total);
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const i = y * cols + x;
      let highest = -Infinity;
      let plants = 0;
      let samples = 0;
      for (let sy = 0; sy < perCell; sy += 1) {
        for (let sx = 0; sx < perCell; sx += 1) {
          const at = (y * perCell + sy) * raster.width + x * perCell + sx;
          if (raster.top[at] === -Infinity) continue;
          highest = Math.max(highest, raster.top[at]);
          const [r, g, b] = [raster.rgb[at * 3], raster.rgb[at * 3 + 1], raster.rgb[at * 3 + 2]];
          if (g > r * 1.04 && g > b * 1.08) plants += 1;
          samples += 1;
        }
      }
      if (!samples) continue;
      height[i] = highest;
      seen[i] = 1;
      green[i] = plants * 2 > samples ? 1 : 0;
    }
  }

  const dtm = groundModel(height, seen, cols, rows, cell);

  // Voids further than this from anything photographed are outside the survey,
  // not hidden behind something, and are left empty.
  const reach = dilate(seen, cols, rows, Math.round(20 / cell));
  const closed = erode(dilate(seen, cols, rows, Math.round(12 / cell)), cols, rows, Math.round(12 / cell));
  const hole = Uint8Array.from({ length: total }, (_, i) => (!seen[i] && reach[i] && closed[i] ? 1 : 0));

  // A hole ringed by high surface is a gap in a roof; anything else is ground.
  const filled = Float32Array.from(height);
  const roofHole = new Uint8Array(total);
  for (const component of connectedComponents(hole, cols, rows)) {
    const rim = [];
    for (const i of component) {
      const x = i % cols;
      for (const n of [x > 0 ? i - 1 : -1, x < cols - 1 ? i + 1 : -1, i - cols, i + cols]) {
        if (n >= 0 && n < total && seen[n]) rim.push(height[n]);
      }
    }
    const ground = percentile(component.map((i) => dtm[i]), 0.5);
    const low = percentile(rim, 0.15);
    const isRoof = component.length * cell * cell < 150 && Number.isFinite(low) && low - ground > 3;
    const level = percentile(rim, 0.5);
    for (const i of component) {
      filled[i] = isRoof ? level : dtm[i];
      if (isRoof) roofHole[i] = 1;
    }
  }

  const structure = new Uint8Array(total);
  for (let i = 0; i < total; i += 1) {
    const standing = Number.isFinite(filled[i]) && filled[i] - dtm[i] >= 3.5;
    if (standing && (roofHole[i] || (seen[i] && !green[i]))) structure[i] = 1;
  }
  // Open then close: drop slivers of noise, bridge single-cell notches.
  const opened = dilate(erode(structure, cols, rows, 1), cols, rows, 1);
  const tidy = erode(dilate(opened, cols, rows, 1), cols, rows, 1);
  const footprints = connectedComponents(tidy, cols, rows)
    .filter((c) => c.length * cell * cell >= 60);

  const groundSample = new Uint8Array(raster.width * raster.height);
  for (let at = 0; at < groundSample.length; at += 1) {
    const i = ((at / raster.width) | 0) / perCell | 0;
    const x = (at % raster.width) / perCell | 0;
    const cellIndex = i * cols + x;
    if (raster.top[at] !== -Infinity && raster.top[at] - dtm[cellIndex] < 2) groundSample[at] = 1;
  }

  return {
    ...grid, height: filled, seen, hole, roofHole, dtm, footprints, ground: paintGround(raster, groundSample),
  };
}
