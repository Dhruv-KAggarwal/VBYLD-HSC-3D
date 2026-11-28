/**
 * Which way is up.
 *
 * A gimballed camera holds roll at zero, so every camera's right axis is level.
 * When the drone turns, those axes span the horizontal plane and give gravity
 * exactly. On a pass that holds roughly one heading they all point the same way:
 * they fix the roll but leave the model free to lean forwards or back.
 * `gravityUpFromPoses` settles that with the flight line, assuming it is level,
 * so a climbing or descending pass tips the whole model by its climb angle (15 to
 * 40 degrees on measured flights). In that case the lean is taken from the scene
 * instead: roofs and ground are level, and their normals say where up is.
 */

import { gravityUpFromPoses, symmetricEigen } from './colmap.mjs';

/** Spread of headings above which the right axes need no help: about 23 degrees either way. */
const ENOUGH_TURN = 0.15;
/** Smallest share of surface normals that must agree on the vertical. */
const MIN_SUPPORT = 0.06;
const SAMPLES = 400000;

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => {
  const length = Math.hypot(...a);
  return length > 1e-12 ? a.map((v) => v / length) : null;
};
const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
const radians = (degrees) => (degrees * Math.PI) / 180;

/** RMS camera roll, in degrees, that a candidate up direction would imply. */
function rollUnder(poses, up) {
  const mean = poses.reduce((sum, p) => sum + dot(p.rightAxis, up) ** 2, 0) / poses.length;
  return (Math.asin(Math.min(1, Math.sqrt(mean))) * 180) / Math.PI;
}

/** Median angle the cameras look below the horizon, in degrees. */
const pitchUnder = (poses, up) => median(poses.map((p) => (
  (Math.asin(Math.max(-1, Math.min(1, -dot(unit(p.viewAxis), up)))) * 180) / Math.PI)));

/** Average the normals inside a cone around `up`, narrowing it each time. */
function settle(normals, start, cones) {
  let up = start;
  let support = 0;
  for (const degrees of cones) {
    const limit = Math.cos(radians(degrees));
    const sum = [0, 0, 0];
    let inside = 0;
    for (const n of normals) {
      const along = dot(n, up);
      if (Math.abs(along) < limit) continue;
      const sign = along < 0 ? -1 : 1;
      for (let k = 0; k < 3; k += 1) sum[k] += sign * n[k];
      inside += 1;
    }
    up = unit(sum) || up;
    support = inside / normals.length;
  }
  return { up, support };
}

function sampleNormals(normals) {
  const count = normals.length / 3;
  const step = Math.max(1, Math.floor(count / SAMPLES));
  const out = [];
  for (let i = 0; i < count; i += step) {
    const n = unit([normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]]);
    if (n) out.push(n);
  }
  return out;
}

/**
 * The commonest surface direction among those a level-rolled camera allows,
 * keeping only ones from which the drone would be looking level or downwards,
 * which rules out mistaking a facade for the ground.
 */
function upFromSurfaces(poses, normals, across, hint) {
  const along = dot(hint, across);
  const first = unit(hint.map((v, k) => v - along * across[k]));
  if (!first) return null;
  const second = cross(across, first);
  const bins = new Float64Array(360);
  const limit = Math.sin(radians(10));
  for (const n of normals) {
    if (Math.abs(dot(n, across)) > limit) continue;
    // Normals have no reliable sign; take the end nearer the hint's half.
    const x = dot(n, first);
    const y = dot(n, second);
    const sign = x < 0 ? -1 : 1;
    bins[(Math.round((Math.atan2(sign * y, sign * x) * 180) / Math.PI) + 360) % 360] += 1;
  }
  let best = null;
  let score = 0;
  for (let angle = 0; angle < 360; angle += 1) {
    let sum = 0;
    for (let d = -4; d <= 4; d += 1) sum += bins[(angle + d + 360) % 360] * (5 - Math.abs(d));
    if (sum <= score) continue;
    const candidate = [0, 1, 2].map((k) => Math.cos(radians(angle)) * first[k] + Math.sin(radians(angle)) * second[k]);
    const pitch = pitchUnder(poses, candidate);
    if (pitch < -10 || pitch > 100) continue;
    score = sum;
    best = candidate;
  }
  return best;
}

/**
 * @param {Array} poses                 solved camera poses
 * @param {Float32Array|null} normals   dense-cloud normals in the same frame
 * @returns {{up:number[]|null, method:'colmap-cameras'|'camera-roll'|'level-surfaces'}}
 */
export function estimateUp(poses, normals = null) {
  const fallback = gravityUpFromPoses(poses);
  if (!fallback) return { up: null, method: 'colmap-cameras' };
  const keep = { up: fallback, method: 'colmap-cameras' };

  const scatter = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const { rightAxis } of poses) {
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) scatter[i][j] += (rightAxis[i] * rightAxis[j]) / poses.length;
    }
  }
  const [least, middle, most] = symmetricEigen(scatter);
  // Cameras look down, so their mean "up" axis picks the sign.
  const hint = unit(poses.reduce((sum, p) => sum.map((v, k) => v - p.downAxis[k]), [0, 0, 0])) || fallback;

  if (middle.value > ENOUGH_TURN) {
    const axis = unit(least.vector);
    return { up: dot(axis, hint) < 0 ? axis.map((v) => -v) : axis, method: 'camera-roll' };
  }

  if (!normals?.length) return keep;
  const sampled = sampleNormals(normals);
  if (sampled.length < 500) return keep;
  const start = upFromSurfaces(poses, sampled, unit(most.vector), hint);
  if (!start) return keep;
  const { up, support } = settle(sampled, start, [10, 8, 8, 8]);
  const pitch = pitchUnder(poses, up);
  // It has to agree with the gimbal: no more roll than a level mount allows.
  const roll = rollUnder(poses, up);
  if (support < MIN_SUPPORT || pitch < -10 || pitch > 100 || roll > Math.max(2.5, rollUnder(poses, fallback) + 0.5)) return keep;
  return { up, method: 'level-surfaces' };
}
