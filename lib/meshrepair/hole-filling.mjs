/**
 * Local hole filling. Only the loop being closed is touched; no global surface
 * (Poisson or otherwise) is ever fitted, so nothing can be invented elsewhere.
 *
 * Strategies:
 *   boundary  minimum-area / minimum-bend triangulation of the loop (Liepa
 *             2003), working directly in 3D
 *   planar    ear clipping in the best-fit plane
 *   curvature the boundary triangulation, refined and then faired so the patch
 *             continues the curvature of the surrounding surface
 *   auto      planar for flat loops, boundary for small curved ones, curvature
 *             for larger curved ones
 */

import { cross, dot, planeFrame, sub } from './boundaries.mjs';

const DP_LIMIT = 140;
const EAR_LIMIT = 1500;

const triangleArea = (a, b, c) => Math.hypot(...cross(sub(b, a), sub(c, a))) / 2;

/** Dynamic-programming triangulation; `blocked(i,j)` vetoes a diagonal. */
function triangulateMinimal(points, blocked) {
  const n = points.length;
  if (n < 3) return null;
  if (n === 3) return [[0, 1, 2]];
  const normalOf = (a, b, c) => {
    const x = cross(sub(points[b], points[a]), sub(points[c], points[a]));
    const l = Math.hypot(...x) || 1;
    return [x[0] / l, x[1] / l, x[2] / l];
  };
  const cost = new Float64Array(n * n).fill(Infinity);
  const bend = new Float64Array(n * n);
  const choice = new Int32Array(n * n).fill(-1);
  for (let i = 0; i + 1 < n; i += 1) cost[i * n + i + 1] = 0;
  for (let span = 2; span < n; span += 1) {
    for (let i = 0; i + span < n; i += 1) {
      const k = i + span;
      if (!(i === 0 && k === n - 1) && blocked(i, k)) continue;
      for (let m = i + 1; m < k; m += 1) {
        const left = cost[i * n + m];
        const right = cost[m * n + k];
        if (left === Infinity || right === Infinity) continue;
        const normal = normalOf(i, m, k);
        let worst = Math.max(bend[i * n + m], bend[m * n + k]);
        if (choice[i * n + m] >= 0) worst = Math.max(worst, 1 - dot(normal, normalOf(i, choice[i * n + m], m)));
        if (choice[m * n + k] >= 0) worst = Math.max(worst, 1 - dot(normal, normalOf(m, choice[m * n + k], k)));
        const area = left + right + triangleArea(points[i], points[m], points[k]);
        const at = i * n + k;
        // Lexicographic: first avoid folded patches, then prefer less area.
        if (cost[at] === Infinity || worst < bend[at] - 0.02 || (worst <= bend[at] + 0.02 && area < cost[at])) {
          cost[at] = area;
          bend[at] = worst;
          choice[at] = m;
        }
      }
    }
  }
  if (cost[n - 1] === Infinity) return null;
  const faces = [];
  const stack = [[0, n - 1]];
  while (stack.length) {
    const [i, k] = stack.pop();
    if (k - i < 2) continue;
    const m = choice[i * n + k];
    faces.push([i, m, k]);
    stack.push([i, m], [m, k]);
  }
  return faces;
}

function earClip(flat, blocked) {
  const n = flat.length;
  let signed = 0;
  for (let i = 0; i < n; i += 1) {
    const a = flat[i];
    const b = flat[(i + 1) % n];
    signed += a[0] * b[1] - b[0] * a[1];
  }
  const orientation = signed >= 0 ? 1 : -1;
  const turn = (a, b, c) => (flat[b][0] - flat[a][0]) * (flat[c][1] - flat[a][1])
    - (flat[b][1] - flat[a][1]) * (flat[c][0] - flat[a][0]);
  const contains = (a, b, c, p) => turn(a, b, p) * orientation >= 0 && turn(b, c, p) * orientation >= 0
    && turn(c, a, p) * orientation >= 0;
  const ring = Array.from({ length: n }, (_, i) => i);
  const faces = [];
  while (ring.length > 3) {
    let best = -1;
    let bestQuality = -1;
    for (let i = 0; i < ring.length; i += 1) {
      const a = ring[(i + ring.length - 1) % ring.length];
      const b = ring[i];
      const c = ring[(i + 1) % ring.length];
      const t = turn(a, b, c) * orientation;
      if (t <= 0 || blocked(a, c)) continue;
      if (ring.some((p) => p !== a && p !== b && p !== c && contains(a, b, c, p))) continue;
      const edges = [[a, b], [b, c], [c, a]].reduce((s, [x, y]) => s + (flat[x][0] - flat[y][0]) ** 2 + (flat[x][1] - flat[y][1]) ** 2, 0);
      const quality = t / (edges || 1);
      if (quality > bestQuality) { bestQuality = quality; best = i; }
    }
    // A self-intersecting projection has no valid ear: give up on this loop
    // rather than emit overlapping triangles.
    if (best < 0) return null;
    faces.push([ring[(best + ring.length - 1) % ring.length], ring[best], ring[(best + 1) % ring.length]]);
    ring.splice(best, 1);
  }
  faces.push([ring[0], ring[1], ring[2]]);
  return faces;
}

/** Insert centroids until patch triangles are about as large as the surroundings. */
function refine(points, faces, targetArea, boundaryCount, maxPoints = 1500) {
  let current = faces;
  for (let round = 0; round < 5 && points.length < maxPoints; round += 1) {
    const next = [];
    let split = 0;
    for (const [a, b, c] of current) {
      if (points.length < maxPoints && triangleArea(points[a], points[b], points[c]) > targetArea * 2) {
        const m = points.length;
        points.push([0, 1, 2].map((k) => (points[a][k] + points[b][k] + points[c][k]) / 3));
        next.push([a, b, m], [b, c, m], [c, a, m]);
        split += 1;
      } else next.push([a, b, c]);
    }
    current = next;
    // Flip interior edges toward better-shaped triangles.
    current = flipEdges(points, current, boundaryCount);
    if (!split) break;
  }
  return current;
}

function flipEdges(points, faces, boundaryCount) {
  const minAngle = (a, b, c) => {
    const angle = (p, q, r) => {
      const u = sub(points[q], points[p]);
      const v = sub(points[r], points[p]);
      return Math.acos(Math.max(-1, Math.min(1, dot(u, v) / ((Math.hypot(...u) * Math.hypot(...v)) || 1))));
    };
    return Math.min(angle(a, b, c), angle(b, c, a), angle(c, a, b));
  };
  const list = faces.map((f) => [...f]);
  const owners = new Map();
  const register = () => {
    owners.clear();
    list.forEach((f, i) => {
      for (let k = 0; k < 3; k += 1) owners.set(`${f[k]}>${f[(k + 1) % 3]}`, i);
    });
  };
  register();
  for (let pass = 0; pass < 3; pass += 1) {
    let flips = 0;
    for (let i = 0; i < list.length; i += 1) {
      for (let k = 0; k < 3; k += 1) {
        const [a, b, c] = [list[i][k], list[i][(k + 1) % 3], list[i][(k + 2) % 3]];
        const j = owners.get(`${b}>${a}`);
        if (j === undefined || j === i) continue;
        const d = list[j].find((v) => v !== a && v !== b);
        if (d === c || owners.has(`${c}>${d}`) || owners.has(`${d}>${c}`)) continue;
        // Never introduce a diagonal between two boundary vertices.
        if (c < boundaryCount && d < boundaryCount) continue;
        if (Math.min(minAngle(a, d, c), minAngle(d, b, c)) <= Math.min(minAngle(a, b, c), minAngle(b, a, d)) + 1e-6) continue;
        list[i] = [a, d, c];
        list[j] = [d, b, c];
        register();
        flips += 1;
        break;
      }
    }
    if (!flips) break;
  }
  return list;
}

/**
 * Move patch-interior vertices so the bi-Laplacian vanishes (Kobbelt et al.),
 * with the original surface's neighbours of the boundary held fixed, which is
 * what gives tangent continuity.
 */
function fair(points, faces, boundaryCount, rings, iterations = 80) {
  const count = points.length;
  if (count === boundaryCount) return;
  const neighbours = Array.from({ length: count }, () => new Set());
  for (const [a, b, c] of faces) {
    neighbours[a].add(b).add(c);
    neighbours[b].add(a).add(c);
    neighbours[c].add(a).add(b);
  }
  const umbrella = Array.from({ length: count }, () => [0, 0, 0]);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (let v = 0; v < count; v += 1) {
      const sum = [0, 0, 0];
      let n = 0;
      for (const u of neighbours[v]) { for (let k = 0; k < 3; k += 1) sum[k] += points[u][k]; n += 1; }
      if (v < boundaryCount) {
        for (const q of rings[v] || []) { for (let k = 0; k < 3; k += 1) sum[k] += q[k]; n += 1; }
      }
      for (let k = 0; k < 3; k += 1) umbrella[v][k] = n ? sum[k] / n - points[v][k] : 0;
    }
    for (let v = boundaryCount; v < count; v += 1) {
      const sum = [0, 0, 0];
      for (const u of neighbours[v]) for (let k = 0; k < 3; k += 1) sum[k] += umbrella[u][k];
      const n = neighbours[v].size || 1;
      for (let k = 0; k < 3; k += 1) points[v][k] -= 0.4 * (sum[k] / n - umbrella[v][k]);
    }
  }
}

/**
 * Triangulate one hole.
 *
 * @param {number[][]} loop    boundary points in surface half-edge order
 * @param {object} options
 * @param {(i:number,j:number)=>boolean} options.existingEdge  true if the mesh
 *   already joins loop vertices i and j (reusing it would be non-manifold)
 * @returns {{points:number[][], faces:number[][], boundaryCount:number, strategy:string}|null}
 *   the first `boundaryCount` points are the loop, in the order given
 */
export function fillHole(loop, {
  strategy = 'auto', planarity = 0, targetArea = Infinity, existingEdge = () => false, rings = [],
} = {}) {
  const n = loop.length;
  if (n < 3 || n > EAR_LIMIT) return null;
  const adjacent = (i, j) => Math.abs(i - j) === 1 || Math.abs(i - j) === n - 1;
  const blocked = (i, j) => !adjacent(i, j) && existingEdge(i, j);
  let chosen = strategy;
  if (chosen === 'auto') {
    if (planarity < 0.03) chosen = 'planar';
    else chosen = n <= 24 ? 'boundary' : 'curvature';
  }
  let faces = null;
  if (chosen === 'planar') {
    const frame = planeFrame(loop);
    faces = earClip(loop.map(frame.project), blocked);
    if (!faces && n <= DP_LIMIT) { faces = triangulateMinimal(loop, blocked); chosen = 'boundary'; }
  } else if (n <= DP_LIMIT) {
    faces = triangulateMinimal(loop, blocked);
  } else {
    const frame = planeFrame(loop);
    faces = earClip(loop.map(frame.project), blocked);
  }
  if (!faces?.length) return null;
  // The patch must run along the boundary opposite to the surface around it.
  // Both triangulators emit triangles that follow the loop order, so reverse.
  const follows = faces.some((f) => [0, 1, 2].some((k) => (f[k] + 1) % n === f[(k + 1) % 3]));
  if (follows) faces = faces.map(([a, b, c]) => [a, c, b]);
  const points = loop.map((p) => [...p]);
  if (chosen === 'curvature') {
    faces = refine(points, faces, targetArea, n);
    fair(points, faces, n, rings);
  }
  return { points, faces, boundaryCount: n, strategy: chosen };
}
