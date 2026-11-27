/**
 * Triangle bounding-volume hierarchy.
 *
 * Pure JavaScript over typed arrays with no Node or three.js dependency, so the
 * same code accelerates picking in the browser viewer and closest-point queries
 * in the pipeline (texture re-baking, LOD error measurement, attribute
 * transfer, self-intersection sampling).
 */

export class MeshBVH {
  constructor(positions, indices, { leafSize = 6 } = {}) {
    this.positions = positions;
    this.indices = indices;
    const faces = indices.length / 3;
    this.faceCount = faces;
    this.order = new Int32Array(faces);
    const centroids = new Float32Array(faces * 3);
    const bounds = new Float32Array(faces * 6);
    for (let f = 0; f < faces; f += 1) {
      this.order[f] = f;
      for (let k = 0; k < 3; k += 1) {
        const a = positions[indices[f * 3] * 3 + k];
        const b = positions[indices[f * 3 + 1] * 3 + k];
        const c = positions[indices[f * 3 + 2] * 3 + k];
        const lo = Math.min(a, b, c);
        const hi = Math.max(a, b, c);
        bounds[f * 6 + k] = lo;
        bounds[f * 6 + 3 + k] = hi;
        centroids[f * 3 + k] = (a + b + c) / 3;
      }
    }

    const capacity = Math.max(1, Math.ceil((faces / Math.max(1, leafSize >> 1)) * 2) + 2);
    this.nodeBounds = new Float32Array(capacity * 6);
    // For a leaf: start/count into `order`. For an inner node count is 0 and
    // start is the index of the right child (the left child follows directly).
    this.nodeStart = new Int32Array(capacity);
    this.nodeCount = new Int32Array(capacity);
    this.nodes = 0;
    if (faces) this.build(0, faces, leafSize, centroids, bounds);
    // Median splits keep the depth logarithmic, so a fixed stack suffices.
    this.stack = new Int32Array(128);
  }

  build(start, end, leafSize, centroids, bounds) {
    const node = this.nodes;
    this.nodes += 1;
    const b = this.nodeBounds;
    const o = node * 6;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    const clo = [Infinity, Infinity, Infinity];
    const chi = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; i += 1) {
      const f = this.order[i];
      for (let k = 0; k < 3; k += 1) {
        if (bounds[f * 6 + k] < lo[k]) lo[k] = bounds[f * 6 + k];
        if (bounds[f * 6 + 3 + k] > hi[k]) hi[k] = bounds[f * 6 + 3 + k];
        const c = centroids[f * 3 + k];
        if (c < clo[k]) clo[k] = c;
        if (c > chi[k]) chi[k] = c;
      }
    }
    for (let k = 0; k < 3; k += 1) { b[o + k] = lo[k]; b[o + 3 + k] = hi[k]; }

    const count = end - start;
    let axis = 0;
    if (chi[1] - clo[1] > chi[axis] - clo[axis]) axis = 1;
    if (chi[2] - clo[2] > chi[axis] - clo[axis]) axis = 2;
    if (count <= leafSize || chi[axis] - clo[axis] <= 0) {
      this.nodeStart[node] = start;
      this.nodeCount[node] = count;
      return node;
    }
    // Median split by quickselect: balanced depth even on degenerate input.
    const middle = start + (count >> 1);
    this.select(start, end - 1, middle, axis, centroids);
    this.nodeCount[node] = 0;
    this.build(start, middle, leafSize, centroids, bounds);
    this.nodeStart[node] = this.build(middle, end, leafSize, centroids, bounds);
    return node;
  }

  select(left, right, k, axis, centroids) {
    const order = this.order;
    let lo = left;
    let hi = right;
    while (hi > lo) {
      const pivot = centroids[order[(lo + hi) >> 1] * 3 + axis];
      let i = lo;
      let j = hi;
      while (i <= j) {
        while (centroids[order[i] * 3 + axis] < pivot) i += 1;
        while (centroids[order[j] * 3 + axis] > pivot) j -= 1;
        if (i <= j) {
          const t = order[i];
          order[i] = order[j];
          order[j] = t;
          i += 1;
          j -= 1;
        }
      }
      if (k <= j) hi = j;
      else if (k >= i) lo = i;
      else return;
    }
  }

  /** Nearest intersection along a ray, or null. */
  raycast(ox, oy, oz, dx, dy, dz, tMax = Infinity, skipFace = -1) {
    if (!this.nodes) return null;
    const ix = 1 / dx;
    const iy = 1 / dy;
    const iz = 1 / dz;
    const b = this.nodeBounds;
    const p = this.positions;
    const idx = this.indices;
    const stack = [0];
    let best = tMax;
    let hit = null;
    while (stack.length) {
      const node = stack.pop();
      const o = node * 6;
      let t0 = (b[o] - ox) * ix;
      let t1 = (b[o + 3] - ox) * ix;
      let near = Math.min(t0, t1);
      let far = Math.max(t0, t1);
      t0 = (b[o + 1] - oy) * iy;
      t1 = (b[o + 4] - oy) * iy;
      near = Math.max(near, Math.min(t0, t1));
      far = Math.min(far, Math.max(t0, t1));
      t0 = (b[o + 2] - oz) * iz;
      t1 = (b[o + 5] - oz) * iz;
      near = Math.max(near, Math.min(t0, t1));
      far = Math.min(far, Math.max(t0, t1));
      if (far < Math.max(0, near) || near > best) continue;
      const count = this.nodeCount[node];
      if (!count) {
        stack.push(this.nodeStart[node], node + 1);
        continue;
      }
      const start = this.nodeStart[node];
      for (let i = start; i < start + count; i += 1) {
        const f = this.order[i];
        if (f === skipFace) continue;
        const a = idx[f * 3] * 3;
        const bb = idx[f * 3 + 1] * 3;
        const c = idx[f * 3 + 2] * 3;
        const e1x = p[bb] - p[a];
        const e1y = p[bb + 1] - p[a + 1];
        const e1z = p[bb + 2] - p[a + 2];
        const e2x = p[c] - p[a];
        const e2y = p[c + 1] - p[a + 1];
        const e2z = p[c + 2] - p[a + 2];
        const px = dy * e2z - dz * e2y;
        const py = dz * e2x - dx * e2z;
        const pz = dx * e2y - dy * e2x;
        const det = e1x * px + e1y * py + e1z * pz;
        if (Math.abs(det) < 1e-20) continue;
        const inv = 1 / det;
        const tx = ox - p[a];
        const ty = oy - p[a + 1];
        const tz = oz - p[a + 2];
        const u = (tx * px + ty * py + tz * pz) * inv;
        if (u < 0 || u > 1) continue;
        const qx = ty * e1z - tz * e1y;
        const qy = tz * e1x - tx * e1z;
        const qz = tx * e1y - ty * e1x;
        const v = (dx * qx + dy * qy + dz * qz) * inv;
        if (v < 0 || u + v > 1) continue;
        const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
        if (t > 1e-9 && t < best) {
          best = t;
          hit = { t, face: f, u, v };
        }
      }
    }
    return hit;
  }

  /**
   * Closest point on the surface within `maxDistance`.
   *
   * `hint` is a face likely to be close (the previous answer, when queries are
   * coherent); testing it first prunes most of the tree.
   * @returns {{distance:number, face:number, u:number, v:number, point:number[]}|null}
   *   u and v are barycentric weights of the second and third corners.
   */
  closestPoint(x, y, z, maxDistance = Infinity, hint = -1) {
    if (!this.nodes) return null;
    const b = this.nodeBounds;
    let best = maxDistance === Infinity ? Infinity : maxDistance * maxDistance;
    let face = -1;
    let u = 0;
    let v = 0;
    const consider = (f) => {
      const d2 = triangleDistance(this.positions, this.indices, f, x, y, z);
      if (d2 <= best) { best = d2; face = f; u = scratch[0]; v = scratch[1]; }
    };
    if (hint >= 0) consider(hint);
    const boxDistance = (node) => {
      const o = node * 6;
      const dx = x < b[o] ? b[o] - x : x > b[o + 3] ? x - b[o + 3] : 0;
      const dy = y < b[o + 1] ? b[o + 1] - y : y > b[o + 4] ? y - b[o + 4] : 0;
      const dz = z < b[o + 2] ? b[o + 2] - z : z > b[o + 5] ? z - b[o + 5] : 0;
      return dx * dx + dy * dy + dz * dz;
    };
    const stack = this.stack;
    let size = 0;
    stack[size++] = 0;
    while (size) {
      const node = stack[--size];
      if (boxDistance(node) > best) continue;
      const count = this.nodeCount[node];
      if (!count) {
        const left = node + 1;
        const right = this.nodeStart[node];
        // Descend into the nearer child first so `best` shrinks early.
        if (boxDistance(left) < boxDistance(right)) { stack[size++] = right; stack[size++] = left; } else { stack[size++] = left; stack[size++] = right; }
        continue;
      }
      const start = this.nodeStart[node];
      for (let i = start; i < start + count; i += 1) consider(this.order[i]);
    }
    if (face < 0) return null;
    const p = this.positions;
    const ia = this.indices[face * 3] * 3;
    const ib = this.indices[face * 3 + 1] * 3;
    const ic = this.indices[face * 3 + 2] * 3;
    const point = [0, 1, 2].map((k) => p[ia + k] + (p[ib + k] - p[ia + k]) * u + (p[ic + k] - p[ia + k]) * v);
    return { distance: Math.sqrt(best), face, u, v, point };
  }

  /** Visit every face whose leaf box overlaps the query box. */
  forEachInBox(min, max, visit) {
    if (!this.nodes) return;
    const b = this.nodeBounds;
    const stack = [0];
    while (stack.length) {
      const node = stack.pop();
      const o = node * 6;
      if (b[o] > max[0] || b[o + 3] < min[0] || b[o + 1] > max[1] || b[o + 4] < min[1]
        || b[o + 2] > max[2] || b[o + 5] < min[2]) continue;
      const count = this.nodeCount[node];
      if (!count) {
        stack.push(this.nodeStart[node], node + 1);
        continue;
      }
      const start = this.nodeStart[node];
      for (let i = start; i < start + count; i += 1) visit(this.order[i]);
    }
  }
}

const scratch = new Float64Array(2);

/**
 * Squared distance from a point to a triangle, leaving the barycentric weights
 * of the closest point in `scratch`. Ericson, Real-Time Collision Detection
 * 5.1.5: closest point by Voronoi region.
 */
function triangleDistance(positions, indices, face, x, y, z) {
  const a = indices[face * 3] * 3;
  const b = indices[face * 3 + 1] * 3;
  const c = indices[face * 3 + 2] * 3;
  const ax = positions[a];
  const ay = positions[a + 1];
  const az = positions[a + 2];
  const abx = positions[b] - ax;
  const aby = positions[b + 1] - ay;
  const abz = positions[b + 2] - az;
  const acx = positions[c] - ax;
  const acy = positions[c + 1] - ay;
  const acz = positions[c + 2] - az;
  const apx = x - ax;
  const apy = y - ay;
  const apz = z - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  let v = 0;
  let w = 0;
  if (!(d1 <= 0 && d2 <= 0)) {
    const d3 = d1 - (abx * abx + aby * aby + abz * abz);
    const abac = abx * acx + aby * acy + abz * acz;
    const d4 = d2 - abac;
    const d5 = d1 - abac;
    const d6 = d2 - (acx * acx + acy * acy + acz * acz);
    const vc = d1 * d4 - d3 * d2;
    const vb = d5 * d2 - d1 * d6;
    const va = d3 * d6 - d5 * d4;
    if (d3 >= 0 && d4 <= d3) {
      v = 1;
    } else if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      v = d1 / (d1 - d3);
    } else if (d6 >= 0 && d5 <= d6) {
      w = 1;
    } else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
      w = d2 / (d2 - d6);
    } else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
      w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      v = 1 - w;
    } else {
      const sum = va + vb + vc;
      if (Math.abs(sum) > 1e-30) { v = vb / sum; w = vc / sum; }
    }
  }
  scratch[0] = v;
  scratch[1] = w;
  const dx = apx - abx * v - acx * w;
  const dy = apy - aby * v - acy * w;
  const dz = apz - abz * v - acz * w;
  return dx * dx + dy * dy + dz * dz;
}

export function closestOnTriangle(positions, indices, face, x, y, z) {
  const d2 = triangleDistance(positions, indices, face, x, y, z);
  return { d2, u: scratch[0], v: scratch[1] };
}
