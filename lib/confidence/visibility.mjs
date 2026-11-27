/**
 * Occlusion testing with one software depth buffer per camera.
 *
 * Rasterising the mesh once per camera and then testing every surface sample
 * against it costs O(cameras x faces), whereas casting a ray from each sample to
 * each camera costs the same number of far more expensive tree traversals.
 */

export class DepthRasterizer {
  constructor(positions, indices, maxWidth = 640) {
    this.positions = positions;
    this.indices = indices;
    this.maxWidth = maxWidth;
    const vertices = positions.length / 3;
    this.screen = new Float32Array(vertices * 3);
    this.buffer = null;
  }

  /** Render the depth buffer of one view. */
  render(view) {
    const width = Math.min(this.maxWidth, view.width);
    const scale = width / view.width;
    const height = Math.max(1, Math.round(view.height * scale));
    if (!this.buffer || this.buffer.length !== width * height) this.buffer = new Float32Array(width * height);
    const depth = this.buffer.fill(Infinity);
    const p = this.positions;
    const s = this.screen;
    const [r0, r1, r2] = view.R;
    const [cx, cy, cz] = view.center;
    const fx = view.fx * scale;
    const fy = view.fy * scale;
    const px = view.cx * scale;
    const py = view.cy * scale;
    for (let i = 0; i < p.length; i += 3) {
      const dx = p[i] - cx;
      const dy = p[i + 1] - cy;
      const dz = p[i + 2] - cz;
      const z = r2[0] * dx + r2[1] * dy + r2[2] * dz;
      s[i + 2] = z;
      if (z > 1e-6) {
        s[i] = (fx * (r0[0] * dx + r0[1] * dy + r0[2] * dz)) / z + px;
        s[i + 1] = (fy * (r1[0] * dx + r1[1] * dy + r1[2] * dz)) / z + py;
      }
    }
    const idx = this.indices;
    for (let f = 0; f < idx.length; f += 3) {
      const a = idx[f] * 3;
      const b = idx[f + 1] * 3;
      const c = idx[f + 2] * 3;
      const za = s[a + 2];
      const zb = s[b + 2];
      const zc = s[c + 2];
      // Triangles crossing the camera plane are skipped; in aerial capture they
      // are vanishingly rare.
      if (!(za > 1e-6 && zb > 1e-6 && zc > 1e-6)) continue;
      const ax = s[a];
      const ay = s[a + 1];
      const bx = s[b];
      const by = s[b + 1];
      const qx = s[c];
      const qy = s[c + 1];
      const x0 = Math.max(0, Math.floor(Math.min(ax, bx, qx)));
      const x1 = Math.min(width - 1, Math.ceil(Math.max(ax, bx, qx)));
      const y0 = Math.max(0, Math.floor(Math.min(ay, by, qy)));
      const y1 = Math.min(height - 1, Math.ceil(Math.max(ay, by, qy)));
      if (x0 > x1 || y0 > y1) continue;
      const area = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax);
      if (Math.abs(area) < 1e-9) continue;
      const inv = 1 / area;
      const ia = 1 / za;
      const ib = 1 / zb;
      const ic = 1 / zc;
      for (let y = y0; y <= y1; y += 1) {
        const sy = y + 0.5;
        for (let x = x0; x <= x1; x += 1) {
          const sx = x + 0.5;
          const wa = ((bx - sx) * (qy - sy) - (by - sy) * (qx - sx)) * inv;
          if (wa < -1e-4) continue;
          const wb = ((qx - sx) * (ay - sy) - (qy - sy) * (ax - sx)) * inv;
          if (wb < -1e-4) continue;
          const wc = 1 - wa - wb;
          if (wc < -1e-4) continue;
          const z = 1 / (wa * ia + wb * ib + wc * ic);
          const at = y * width + x;
          if (z < depth[at]) depth[at] = z;
        }
      }
    }
    return { width, height, scale, depth };
  }
}

/**
 * Is a point at pixel (x, y) of the buffer and depth z visible?
 * The farthest of the surrounding texels is used, so a sample on a slanted
 * surface is not reported as occluded by its own neighbouring pixel.
 */
export function visibleAt(frame, x, y, z, tolerance = 0.012) {
  const { width, height, depth } = frame;
  const x0 = Math.min(width - 1, Math.max(0, Math.floor(x - 0.5)));
  const y0 = Math.min(height - 1, Math.max(0, Math.floor(y - 0.5)));
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  let far = -Infinity;
  for (const value of [depth[y0 * width + x0], depth[y0 * width + x1], depth[y1 * width + x0], depth[y1 * width + x1]]) {
    if (value !== Infinity && value > far) far = value;
  }
  // Nothing rasterised here (a sub-pixel triangle): nothing can be in front.
  return far === -Infinity || z <= far * (1 + tolerance);
}
