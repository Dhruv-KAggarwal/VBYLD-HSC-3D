/**
 * Spatial indices over point sets, built on typed arrays.
 *
 * The dense cloud runs to millions of points, so a Map of arrays is not an
 * option: it costs hundreds of megabytes of object headers. This is a hashed
 * uniform grid in compressed-sparse-row form (two Int32 arrays), which indexes
 * two million points in well under a second.
 */

const PRIME_X = 73856093;
const PRIME_Y = 19349663;
const PRIME_Z = 83492791;

function nextPowerOfTwo(value) {
  let power = 1;
  while (power < value) power *= 2;
  return power;
}

/** Robust bounds: percentiles over a sample, so a few flyaway points cannot set the scale. */
export function robustBounds(positions, lo = 0.02, hi = 0.98, maxSamples = 20000) {
  const count = positions.length / 3;
  const step = Math.max(1, Math.floor(count / maxSamples));
  const axes = [[], [], []];
  for (let i = 0; i < count; i += step) {
    for (let k = 0; k < 3; k += 1) {
      const value = positions[i * 3 + k];
      if (Number.isFinite(value)) axes[k].push(value);
    }
  }
  const min = [0, 0, 0];
  const max = [0, 0, 0];
  for (let k = 0; k < 3; k += 1) {
    const sorted = Float64Array.from(axes[k]).sort();
    if (!sorted.length) continue;
    min[k] = sorted[Math.floor(lo * (sorted.length - 1))];
    max[k] = sorted[Math.floor(hi * (sorted.length - 1))];
  }
  return { min, max, size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]] };
}

/** A starting cell size for surface-like data: a couple of point spacings. */
export function suggestCellSize(positions) {
  const { size } = robustBounds(positions);
  const [a, b] = [...size].sort((x, y) => y - x);
  const area = Math.max(a * b, 1e-12);
  const count = Math.max(1, positions.length / 3);
  return Math.max(1e-9, 2.5 * Math.sqrt(area / count));
}

export class PointGrid {
  constructor(positions, cellSize = suggestCellSize(positions)) {
    this.positions = positions;
    this.count = positions.length / 3;
    this.cellSize = cellSize;
    const tableSize = nextPowerOfTwo(Math.max(16, this.count));
    this.mask = tableSize - 1;
    const bucketOf = new Int32Array(this.count);
    const starts = new Int32Array(tableSize + 1);
    for (let i = 0; i < this.count; i += 1) {
      const bucket = this.bucket(
        Math.floor(positions[i * 3] / cellSize),
        Math.floor(positions[i * 3 + 1] / cellSize),
        Math.floor(positions[i * 3 + 2] / cellSize),
      );
      bucketOf[i] = bucket;
      starts[bucket + 1] += 1;
    }
    for (let i = 1; i <= tableSize; i += 1) starts[i] += starts[i - 1];
    const cursor = starts.slice(0, tableSize);
    const order = new Int32Array(this.count);
    for (let i = 0; i < this.count; i += 1) {
      order[cursor[bucketOf[i]]] = i;
      cursor[bucketOf[i]] += 1;
    }
    this.starts = starts;
    this.order = order;
    // Coordinates and cell indices copied in bucket order: a query then reads
    // contiguous memory and tests cell membership without recomputing it.
    this.sorted = new Float32Array(this.count * 3);
    this.cells = new Int32Array(this.count * 3);
    for (let slot = 0; slot < this.count; slot += 1) {
      const i = order[slot] * 3;
      for (let k = 0; k < 3; k += 1) {
        this.sorted[slot * 3 + k] = positions[i + k];
        this.cells[slot * 3 + k] = Math.floor(positions[i + k] / cellSize);
      }
    }
  }

  bucket(ix, iy, iz) {
    return (Math.imul(ix, PRIME_X) ^ Math.imul(iy, PRIME_Y) ^ Math.imul(iz, PRIME_Z)) & this.mask;
  }

  /**
   * Call `visit(index, distanceSquared)` for every point within `radius`.
   * Returning `false` from `visit` stops the search early.
   */
  forEachWithin(x, y, z, radius, visit) {
    const s = this.cellSize;
    const r2 = radius * radius;
    const p = this.sorted;
    const { cells, starts, order } = this;
    const x1 = Math.floor((x + radius) / s);
    const y1 = Math.floor((y + radius) / s);
    const z1 = Math.floor((z + radius) / s);
    for (let ix = Math.floor((x - radius) / s); ix <= x1; ix += 1) {
      for (let iy = Math.floor((y - radius) / s); iy <= y1; iy += 1) {
        for (let iz = Math.floor((z - radius) / s); iz <= z1; iz += 1) {
          const bucket = this.bucket(ix, iy, iz);
          for (let slot = starts[bucket], end = starts[bucket + 1]; slot < end; slot += 1) {
            const o = slot * 3;
            if (cells[o] !== ix || cells[o + 1] !== iy || cells[o + 2] !== iz) continue;
            const dx = p[o] - x;
            const dy = p[o + 1] - y;
            const dz = p[o + 2] - z;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= r2 && visit(order[slot], d2) === false) return;
          }
        }
      }
    }
  }

  /** Nearest point within `maxRadius`, or null. `exclude` skips one index. */
  nearest(x, y, z, maxRadius, exclude = -1) {
    const s = this.cellSize;
    const cx = Math.floor(x / s);
    const cy = Math.floor(y / s);
    const cz = Math.floor(z / s);
    const rings = Math.ceil(maxRadius / s);
    const p = this.sorted;
    const { cells, starts, order } = this;
    let best = maxRadius * maxRadius;
    let index = -1;
    for (let ring = 0; ring <= rings; ring += 1) {
      // A point in ring k is at least (k-1) cells away, so once the best hit is
      // closer than that no further ring can improve on it.
      if (index >= 0 && (ring - 1) * s >= Math.sqrt(best)) break;
      for (let dx = -ring; dx <= ring; dx += 1) {
        for (let dy = -ring; dy <= ring; dy += 1) {
          const edge = Math.abs(dx) === ring || Math.abs(dy) === ring;
          // Only the shell of the cube: interior cells were visited earlier.
          const step = edge ? 1 : Math.max(1, ring * 2);
          for (let dz = -ring; dz <= ring; dz += step) {
            const ix = cx + dx;
            const iy = cy + dy;
            const iz = cz + dz;
            // Skip cells that lie wholly beyond the best hit so far.
            const gx = ix * s > x ? ix * s - x : x > (ix + 1) * s ? x - (ix + 1) * s : 0;
            const gy = iy * s > y ? iy * s - y : y > (iy + 1) * s ? y - (iy + 1) * s : 0;
            const gz = iz * s > z ? iz * s - z : z > (iz + 1) * s ? z - (iz + 1) * s : 0;
            if (gx * gx + gy * gy + gz * gz > best) continue;
            const bucket = this.bucket(ix, iy, iz);
            for (let slot = starts[bucket], end = starts[bucket + 1]; slot < end; slot += 1) {
              const o = slot * 3;
              if (cells[o] !== ix || cells[o + 1] !== iy || cells[o + 2] !== iz) continue;
              const ex = p[o] - x;
              const ey = p[o + 1] - y;
              const ez = p[o + 2] - z;
              const d2 = ex * ex + ey * ey + ez * ez;
              if (d2 <= best && order[slot] !== exclude) { best = d2; index = order[slot]; }
            }
          }
        }
      }
    }
    return index < 0 ? null : { index, distance: Math.sqrt(best) };
  }
}

/** Median nearest-neighbour distance: the cloud's own sampling resolution. */
export function estimateSpacing(grid, samples = 1500) {
  const step = Math.max(1, Math.floor(grid.count / samples));
  const distances = [];
  const p = grid.positions;
  for (let i = 0; i < grid.count; i += step) {
    const hit = grid.nearest(p[i * 3], p[i * 3 + 1], p[i * 3 + 2], grid.cellSize * 4, i);
    if (hit && hit.distance > 0) distances.push(hit.distance);
  }
  if (!distances.length) return grid.cellSize / 2.5;
  distances.sort((a, b) => a - b);
  return distances[Math.floor(distances.length / 2)];
}

export function median(values) {
  if (!values.length) return Number.NaN;
  const sorted = Float64Array.from(values).sort();
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
