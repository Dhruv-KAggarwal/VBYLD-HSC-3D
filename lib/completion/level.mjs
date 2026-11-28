/**
 * The vertical of the scene itself.
 *
 * The exported frame takes "up" from the camera poses, which on a forward
 * looking flight leans with the gimbal's pitch. Roofs and ground are level, so
 * the area-weighted normal of the near-horizontal surface gives the real
 * vertical, and walls built along it stand straight on the photographed ones.
 */

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function findUp(positions, indices, use = () => true) {
  let up = [0, 1, 0];
  // Widest cone first, then tighten around what dominates.
  for (const degrees of [35, 25, 18, 12, 10, 10, 10]) {
    const limit = Math.cos((degrees * Math.PI) / 180);
    const sum = [0, 0, 0];
    for (let f = 0; f < indices.length / 3; f += 1) {
      if (!use(f)) continue;
      const [a, b, c] = [indices[f * 3] * 3, indices[f * 3 + 1] * 3, indices[f * 3 + 2] * 3];
      const u = [0, 1, 2].map((k) => positions[b + k] - positions[a + k]);
      const v = [0, 1, 2].map((k) => positions[c + k] - positions[a + k]);
      const n = cross(u, v);
      const along = dot(n, up);
      if (Math.abs(along) <= Math.hypot(...n) * limit) continue;
      const sign = along < 0 ? -1 : 1;
      for (let k = 0; k < 3; k += 1) sum[k] += sign * n[k];
    }
    const length = Math.hypot(...sum);
    if (!length) break;
    up = sum.map((value) => value / length);
  }
  return up;
}

/** The smallest rotation taking `up` to +Y, as three rows. */
export function levelRotation(up) {
  const [x, y, z] = up;
  const k = 1 / (1 + y);
  return [
    [1 - x * x * k, -x, -x * z * k],
    [x, y, z],
    [-x * z * k, -z, 1 - z * z * k],
  ];
}

export function rotatePositions(positions, rows) {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k += 1) {
      out[i + k] = rows[k][0] * positions[i] + rows[k][1] * positions[i + 1] + rows[k][2] * positions[i + 2];
    }
  }
  return out;
}

/** Inverse of a rotation applied to one point. */
export const unrotate = (p, rows) => [0, 1, 2].map((k) => rows[0][k] * p[0] + rows[1][k] * p[1] + rows[2][k] * p[2]);
