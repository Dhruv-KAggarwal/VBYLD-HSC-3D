/**
 * Walls for the sides of buildings that were never photographed.
 *
 * Each footprint becomes a straightened polygon. Along every edge, short tiles
 * of wall are added only where the reconstruction has no surface of its own, and
 * they are painted with a facade of the same building that was photographed.
 * The result is an estimate of what is there, never a measurement.
 */

import { percentile } from '../buildings.mjs';
import { faceColor } from './observed.mjs';
import { pushPull } from './topview.mjs';

const TILE = 1.5;

/** Closed outlines of a set of cells, as lists of grid-corner points. */
function outlines(cells, cols) {
  const member = new Set(cells);
  const next = new Map();
  const add = (ax, ay, bx, by) => {
    const key = `${ax},${ay}`;
    if (!next.has(key)) next.set(key, []);
    next.get(key).push([bx, by]);
  };
  for (const i of cells) {
    const x = i % cols;
    const y = (i / cols) | 0;
    // Edges run so that the building is always on the same side.
    if (!member.has(i - cols)) add(x, y, x + 1, y);
    if (x === cols - 1 || !member.has(i + 1)) add(x + 1, y, x + 1, y + 1);
    if (!member.has(i + cols)) add(x + 1, y + 1, x, y + 1);
    if (x === 0 || !member.has(i - 1)) add(x, y + 1, x, y);
  }
  const loops = [];
  for (const [start, targets] of next) {
    while (targets.length) {
      const loop = [start.split(',').map(Number)];
      let point = targets.pop();
      while (`${point[0]},${point[1]}` !== start) {
        loop.push(point);
        const options = next.get(`${point[0]},${point[1]}`);
        if (!options?.length) break;
        point = options.pop();
      }
      if (loop.length >= 4) loops.push(loop);
    }
  }
  return loops;
}

function douglasPeucker(points, tolerance) {
  if (points.length < 3) return points;
  const [ax, ay] = points[0];
  const [bx, by] = points[points.length - 1];
  const length = Math.hypot(bx - ax, by - ay);
  let worst = 0;
  let at = 0;
  for (let i = 1; i < points.length - 1; i += 1) {
    const [px, py] = points[i];
    const d = length > 1e-9
      ? Math.abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / length
      : Math.hypot(px - ax, py - ay);
    if (d > worst) { worst = d; at = i; }
  }
  if (worst <= tolerance) return [points[0], points[points.length - 1]];
  return [...douglasPeucker(points.slice(0, at + 1), tolerance).slice(0, -1), ...douglasPeucker(points.slice(at), tolerance)];
}

function simplifyLoop(loop, tolerance) {
  // Split at the point farthest from the start so both halves have real chords.
  let far = 0;
  let distance = 0;
  loop.forEach(([x, y], i) => {
    const d = Math.hypot(x - loop[0][0], y - loop[0][1]);
    if (d > distance) { distance = d; far = i; }
  });
  const a = douglasPeucker(loop.slice(0, far + 1), tolerance);
  const b = douglasPeucker([...loop.slice(far), loop[0]], tolerance);
  return [...a.slice(0, -1), ...b.slice(0, -1)];
}

/** Snap edges to the building's own two axes where that moves nothing far. */
function squareUp(loop, tolerance) {
  const n = loop.length;
  let sx = 0;
  let sy = 0;
  const edges = loop.map(([ax, ay], i) => {
    const [bx, by] = loop[(i + 1) % n];
    const angle = Math.atan2(by - ay, bx - ax);
    const length = Math.hypot(bx - ax, by - ay);
    sx += Math.cos(4 * angle) * length;
    sy += Math.sin(4 * angle) * length;
    return { angle, mx: (ax + bx) / 2, my: (ay + by) / 2 };
  });
  const axis = Math.atan2(sy, sx) / 4;
  for (const edge of edges) {
    const snapped = axis + Math.round((edge.angle - axis) / (Math.PI / 2)) * (Math.PI / 2);
    if (Math.abs(snapped - edge.angle) < Math.PI / 6) edge.angle = snapped;
  }
  return loop.map((point, i) => {
    const a = edges[(i + n - 1) % n];
    const b = edges[i];
    const d = Math.sin(b.angle - a.angle);
    if (Math.abs(d) < 0.2) return point;
    const t = ((b.mx - a.mx) * Math.sin(b.angle) - (b.my - a.my) * Math.cos(b.angle)) / d;
    const x = a.mx + t * Math.cos(a.angle);
    const y = a.my + t * Math.sin(a.angle);
    return Math.hypot(x - point[0], y - point[1]) <= tolerance ? [x, y] : point;
  });
}

/** Straight wall runs of one footprint, with their roof and ground heights. */
function buildingEdges(top, cells) {
  const { cols, rows, cell, minX, minZ, height, dtm } = top;
  const member = new Set(cells);
  const indexAt = (x, z) => {
    const c = Math.floor((x - minX) / cell);
    const r = Math.floor((z - minZ) / cell);
    return c >= 0 && r >= 0 && c < cols && r < rows ? r * cols + c : -1;
  };
  const edges = [];
  for (const raw of outlines(cells, cols)) {
    const loop = squareUp(simplifyLoop(raw, 1.8 / cell), 3 / cell)
      .map(([x, y]) => [minX + x * cell, minZ + y * cell]);
    for (let i = 0; i < loop.length; i += 1) {
      const a = loop[i];
      const b = loop[(i + 1) % loop.length];
      const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (length < 1.5) continue;
      const along = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
      let out = [along[1], -along[0]];
      // Point the normal away from the footprint.
      let inside = 0;
      for (const t of [0.25, 0.5, 0.75]) {
        const x = a[0] + along[0] * length * t;
        const z = a[1] + along[1] * length * t;
        if (member.has(indexAt(x + out[0] * 1.5, z + out[1] * 1.5))) inside += 1;
        if (member.has(indexAt(x - out[0] * 1.5, z - out[1] * 1.5))) inside -= 1;
      }
      if (inside > 0) out = [-out[0], -out[1]];
      const pieces = Math.ceil(length / 12);
      for (let p = 0; p < pieces; p += 1) {
        const span = length / pieces;
        const start = [a[0] + along[0] * span * p, a[1] + along[1] * span * p];
        const roofs = [];
        const grounds = [];
        for (let t = 0.5; t < span; t += 1) {
          const x = start[0] + along[0] * t;
          const z = start[1] + along[1] * t;
          for (const depth of [0.8, 1.8]) {
            const roof = indexAt(x - out[0] * depth, z - out[1] * depth);
            if (member.has(roof) && Number.isFinite(height[roof])) roofs.push(height[roof]);
          }
          const ground = indexAt(x + out[0] * 3, z + out[1] * 3);
          if (ground >= 0) grounds.push(dtm[ground]);
        }
        if (!roofs.length || !grounds.length) continue;
        const roof = percentile(roofs, 0.4);
        const base = percentile(grounds, 0.3);
        if (!(roof - base >= 3)) continue;
        edges.push({ start, along, out, length: span, base, roof });
      }
    }
  }
  return edges;
}

/** Vertical component of a face's unit normal: 0 for a wall, 1 for flat ground. */
function normalY(bvh, face) {
  const [a, b, c] = [0, 1, 2].map((k) => bvh.indices[face * 3 + k] * 3);
  const p = bvh.positions;
  const u = [0, 1, 2].map((k) => p[b + k] - p[a + k]);
  const v = [0, 1, 2].map((k) => p[c + k] - p[a + k]);
  const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  return n[1] / (Math.hypot(...n) || 1);
}

/** Does the reconstruction already have surface at, or behind, this wall point? */
function covered(bvh, point, out) {
  if (bvh.closestPoint(point[0], point[1], point[2], 1.1)) return true;
  // A real wall further in would be hidden by an estimate placed in front.
  const hit = bvh.raycast(point[0], point[1], point[2], -out[0], 0, -out[1], 5);
  return Boolean(hit) && Math.abs(normalY(bvh, hit.face)) < 0.6;
}

/** Orthographic photograph of whatever the reconstruction has on a wall run. */
function rectify(mesh, textures, bvh, faceMap, edge, ppm) {
  const width = Math.max(2, Math.round(edge.length * ppm));
  const height = Math.max(2, Math.round((edge.roof - edge.base) * ppm));
  const values = new Float32Array(width * height * 3);
  const valid = new Uint8Array(width * height);
  const colour = [0, 0, 0];
  let hits = 0;
  for (let y = 0; y < height; y += 1) {
    const wy = edge.roof - (y + 0.5) / ppm;
    for (let x = 0; x < width; x += 1) {
      const t = (x + 0.5) / ppm;
      const hit = bvh.raycast(
        edge.start[0] + edge.along[0] * t + edge.out[0] * 2.5, wy, edge.start[1] + edge.along[1] * t + edge.out[1] * 2.5,
        -edge.out[0], 0, -edge.out[1], 5,
      );
      // Roofs and ground seen through a gap are not part of the facade.
      if (!hit || Math.abs(normalY(bvh, hit.face)) > 0.7
        || !faceColor(mesh, textures, faceMap[hit.face], 1 - hit.u - hit.v, hit.u, hit.v, colour)) continue;
      values.set(colour, (y * width + x) * 3);
      valid[y * width + x] = 1;
      hits += 1;
    }
  }
  return { width, height, values, valid, coverage: hits / (width * height) };
}

/**
 * Detail that changes both across and up a strip, as windows and balconies do.
 * A smear changes in one direction only and scores low.
 */
function detail({ width, height, values, valid }) {
  let across = 0;
  let upward = 0;
  let count = 0;
  for (let y = 0; y + 1 < height; y += 1) {
    for (let x = 0; x + 1 < width; x += 1) {
      const i = y * width + x;
      if (!valid[i] || !valid[i + 1] || !valid[i + width]) continue;
      across += Math.abs(values[i * 3 + 1] - values[(i + 1) * 3 + 1]);
      upward += Math.abs(values[i * 3 + 1] - values[(i + width) * 3 + 1]);
      count += 1;
    }
  }
  return count ? Math.min(across, upward) / count : 0;
}

/** The best photographed facade of a building, with its small gaps filled. */
function pickDonor(mesh, textures, bvh, faceMap, edges, ppm) {
  let best = null;
  const candidates = edges.filter((e) => e.length >= 5).sort((a, b) => b.length * (b.roof - b.base) - a.length * (a.roof - a.base));
  for (const edge of candidates.slice(0, 14)) {
    const strip = rectify(mesh, textures, bvh, faceMap, edge, ppm);
    if (strip.coverage < 0.55) continue;
    const score = strip.coverage ** 3 * Math.min(edge.length, 12) * (edge.roof - edge.base) * (1 + detail(strip)) ** 2;
    if (!best || score > best.score) best = { ...strip, score, metres: edge.roof - edge.base };
  }
  if (!best) return null;
  pushPull(best.values, best.valid, best.width, best.height, 3);
  return { width: best.width, height: best.height, data: best.values, metres: best.metres };
}

const mirror = (value, period) => {
  const t = ((value % (2 * period)) + 2 * period) % (2 * period);
  return t < period ? t : 2 * period - t;
};

/** Paint one estimated wall run: copied facade, or plain wall with windows. */
export function paintWall(edge, donor, tone, ppm) {
  const width = Math.max(4, Math.ceil(edge.length * ppm));
  const wall = edge.roof - edge.base;
  const height = Math.max(4, Math.ceil(wall * ppm));
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const above = wall - (y + 0.5) / ppm; // metres above the ground
    for (let x = 0; x < width; x += 1) {
      const along = (x + 0.5) / ppm;
      const at = (y * width + x) * 3;
      if (donor) {
        // Skip the donor's roof lip and ground clutter, then repeat storeys.
        const low = Math.min(1, donor.metres * 0.15);
        const high = Math.max(low + 0.5, donor.metres - 0.4);
        const storeyHeight = Math.min(3.2, high - low);
        // Mirrored across (no hard seam), but storeys repeat the right way up.
        const level = above <= high ? Math.max(low, above) : high - storeyHeight + ((above - high) % storeyHeight);
        const sx = Math.min(donor.width - 1, Math.floor(mirror(along, donor.width / ppm - 1e-3) * ppm));
        const sy = Math.max(0, Math.min(donor.height - 1, Math.floor((donor.metres - level) * ppm)));
        for (let c = 0; c < 3; c += 1) data[at + c] = donor.data[(sy * donor.width + sx) * 3 + c];
      } else {
        const bay = (along + 1.5 - ((edge.length % 3) / 2)) % 3;
        const storey = above % 3.2;
        const window = bay > 0.8 && bay < 2.1 && storey > 1 && storey < 2.4 && along > 0.6 && along < edge.length - 0.6 && above < wall - 0.5;
        const grain = ((x * 7 + y * 13) % 5) - 2;
        for (let c = 0; c < 3; c += 1) {
          data[at + c] = Math.max(0, Math.min(255, (window ? tone[c] * 0.45 + 12 : tone[c]) + grain));
        }
      }
    }
  }
  return { width, height, data };
}

/** Typical wall colour across the photographed facades. */
function typicalTone(donors) {
  const channels = [[], [], []];
  for (const donor of donors) {
    for (let i = 0; i < donor.data.length; i += 3 * 17) {
      for (let c = 0; c < 3; c += 1) channels[c].push(donor.data[i + c]);
    }
  }
  return channels.map((v, c) => (v.length ? percentile(v, 0.75) : [182, 168, 146][c]));
}

/**
 * @returns {Array<{edge:object, tiles:number[][], donor:object|null, tone:number[]}>}
 *   one entry per wall run needing estimated surface; each tile is
 *   [from, to, bottom, top] in metres along the run and in world height.
 */
export function planWalls(top, mesh, textures, bvh, faceMap, ppm, onProgress = () => {}) {
  const plans = [];
  let buildings = 0;
  let copied = 0;
  top.footprints.forEach((cells, n) => {
    const edges = buildingEdges(top, cells);
    const needing = [];
    for (const edge of edges) {
      const open = new Map();
      const across = Math.ceil(edge.length / TILE);
      const up = Math.ceil((edge.roof - edge.base) / TILE);
      for (let i = 0; i < across; i += 1) {
        const from = (edge.length * i) / across;
        const to = (edge.length * (i + 1)) / across;
        for (let j = 0; j < up; j += 1) {
          const bottom = edge.base + ((edge.roof - edge.base) * j) / up;
          const upper = edge.base + ((edge.roof - edge.base) * (j + 1)) / up;
          const t = (from + to) / 2;
          const point = [edge.start[0] + edge.along[0] * t, (bottom + upper) / 2, edge.start[1] + edge.along[1] * t];
          if (!covered(bvh, point, edge.out)) open.set(`${i},${j}`, [from, to, bottom, upper]);
        }
      }
      // A single column of tiles on its own reads as a stray post, not a wall.
      const tiles = [...open].filter(([key]) => {
        const [i, j] = key.split(',').map(Number);
        return across === 1 || open.has(`${i - 1},${j}`) || open.has(`${i + 1},${j}`);
      }).map(([, tile]) => tile);
      if (tiles.length) needing.push({ edge, tiles });
    }
    if (needing.length) {
      const donor = pickDonor(mesh, textures, bvh, faceMap, edges, ppm);
      buildings += 1;
      if (donor) copied += 1;
      for (const item of needing) plans.push({ ...item, donor });
    }
    onProgress((n + 1) / top.footprints.length);
  });
  const tone = typicalTone([...new Set(plans.map((p) => p.donor).filter(Boolean))]);
  for (const plan of plans) plan.tone = tone;
  return { plans, buildings, copied };
}
