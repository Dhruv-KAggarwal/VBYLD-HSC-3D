/**
 * Depth-map consistency, from COLMAP's geometric depth maps.
 *
 * Available only when the dense stage was COLMAP patch-match stereo. OpenMVS is
 * run with `--remove-dmaps`, so for those jobs this evidence is reported as
 * unavailable rather than approximated.
 */

import { existsSync } from 'node:fs';
import { join } from 'node:path';

export function depthMapDirectory(work) {
  const path = join(work, 'dense', 'stereo', 'depth_maps');
  return existsSync(path) ? path : null;
}

/** COLMAP map file: ASCII "width&height&channels&" then little-endian floats. */
export function parseDepthMap(buffer) {
  let separators = 0;
  let offset = 0;
  while (offset < 64 && offset < buffer.length && separators < 3) {
    if (buffer[offset] === 38) separators += 1;
    offset += 1;
  }
  const [width, height, channels] = buffer.toString('ascii', 0, offset).split('&').map(Number);
  if (separators !== 3 || !(width > 0) || !(height > 0) || channels !== 1) return null;
  if (buffer.length - offset < width * height * 4) return null;
  const bytes = buffer.subarray(offset, offset + width * height * 4);
  const data = new Float32Array(width * height);
  new Uint8Array(data.buffer).set(bytes);
  return { width, height, data };
}

/**
 * @param {number} z depth of the surface sample in reconstruction units
 * @returns {boolean|null} agreement, or null where the map holds no estimate
 */
export function compareDepth(map, view, u, v, z, tolerance = 0.02) {
  const x = Math.min(map.width - 1, Math.max(0, Math.floor((u / view.width) * map.width)));
  const y = Math.min(map.height - 1, Math.max(0, Math.floor((v / view.height) * map.height)));
  const measured = map.data[y * map.width + x];
  if (!(measured > 0)) return null;
  return Math.abs(measured - z) <= tolerance * z;
}
