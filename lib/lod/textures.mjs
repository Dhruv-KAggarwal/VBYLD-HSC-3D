/**
 * Level-of-detail textures.
 *
 * Where simplification could keep the texture seams, a level reuses the original
 * UVs and only needs smaller images. Photogrammetry atlases are often split into
 * tens of thousands of charts, which stops seam-preserving collapse far above
 * the lower targets. Those levels get a new atlas: the simplified surface is
 * unwrapped into charts, and every texel is filled by sampling the ORIGINAL
 * model at the nearest surface point. No texel is synthesised from anything but
 * the original photography.
 */

import { MeshBVH } from '../bvh.mjs';
import { faceGeometry } from '../glbmesh.mjs';
import { buildTopology } from '../meshdata.mjs';
import { decodeImage, encodeImage, imageSize, resizeImage, sampleImage } from '../image.mjs';

const CHART_COS = Math.cos((55 * Math.PI) / 180);
const PADDING = 2;

/** Resize (never enlarge) the images a level refers to. */
export async function textureVariants(images, used, maxEdge, options) {
  const out = images.slice();
  let largest = 0;
  for (const index of used) {
    const source = images[index];
    const size = imageSize(source.bytes);
    const edge = size ? Math.max(size.width, size.height) : Infinity;
    const format = options.format === 'auto' ? null : options.format;
    const sameFormat = !format || source.mime === `image/${format}`;
    if (edge <= maxEdge && sameFormat) {
      largest = Math.max(largest, edge);
      continue;
    }
    const image = resizeImage(decodeImage(source.bytes), maxEdge);
    largest = Math.max(largest, image.width, image.height);
    const encoded = await encodeImage(image, { ...options, format: format || 'jpeg' });
    // Re-encoding a same-sized image that only grows is pointless.
    out[index] = edge <= maxEdge && encoded.bytes.length >= source.bytes.length ? source : encoded;
  }
  return { images: out, maxEdge: largest };
}

/** Group faces into charts of similar orientation and flatten each one. */
function unwrap(mesh) {
  const faces = mesh.indices.length / 3;
  const normals = new Float32Array(faces * 3);
  const areas = new Float32Array(faces);
  for (let f = 0; f < faces; f += 1) {
    const g = faceGeometry(mesh.positions, mesh.indices, f);
    normals.set(g.normal, f * 3);
    areas[f] = g.area;
  }
  const topology = buildTopology(mesh);
  const chartOf = new Int32Array(faces).fill(-1);
  const order = Array.from({ length: faces }, (_, i) => i).sort((a, b) => areas[b] - areas[a]);
  const charts = [];
  const queue = new Int32Array(faces);
  for (const seed of order) {
    if (chartOf[seed] >= 0) continue;
    const id = charts.length;
    const n = [normals[seed * 3], normals[seed * 3 + 1], normals[seed * 3 + 2]];
    let head = 0;
    let tail = 0;
    queue[tail++] = seed;
    chartOf[seed] = id;
    while (head < tail) {
      const f = queue[head++];
      for (let k = 0; k < 3; k += 1) {
        const e = topology.halfEdge[f * 3 + k];
        if (e < 0 || topology.edgeUse[e] !== 2) continue;
        for (let i = topology.edgeStart[e]; i < topology.edgeStart[e + 1]; i += 1) {
          const g = (topology.edgeHalves[i] / 3) | 0;
          if (chartOf[g] >= 0) continue;
          if (n[0] * normals[g * 3] + n[1] * normals[g * 3 + 1] + n[2] * normals[g * 3 + 2] < CHART_COS) continue;
          chartOf[g] = id;
          queue[tail++] = g;
        }
      }
    }
    charts.push({ faces: Array.from(queue.subarray(0, tail)), normal: n });
  }
  // Project each chart onto the plane facing its seed normal.
  for (const chart of charts) {
    const [nx, ny, nz] = chart.normal;
    const helper = Math.abs(nx) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let u = [helper[1] * nz - helper[2] * ny, helper[2] * nx - helper[0] * nz, helper[0] * ny - helper[1] * nx];
    const length = Math.hypot(...u) || 1;
    u = u.map((c) => c / length);
    const v = [ny * u[2] - nz * u[1], nz * u[0] - nx * u[2], nx * u[1] - ny * u[0]];
    const local = new Map();
    let x0 = Infinity; let y0 = Infinity; let x1 = -Infinity; let y1 = -Infinity;
    for (const f of chart.faces) {
      for (let k = 0; k < 3; k += 1) {
        const vertex = mesh.indices[f * 3 + k];
        if (local.has(vertex)) continue;
        const p = vertex * 3;
        const x = mesh.positions[p] * u[0] + mesh.positions[p + 1] * u[1] + mesh.positions[p + 2] * u[2];
        const y = mesh.positions[p] * v[0] + mesh.positions[p + 1] * v[1] + mesh.positions[p + 2] * v[2];
        local.set(vertex, [x, y]);
        x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
      }
    }
    for (const point of local.values()) { point[0] -= x0; point[1] -= y0; }
    chart.points = local;
    chart.width = x1 - x0;
    chart.height = y1 - y0;
    // Lay long charts flat so shelves stay low.
    if (chart.height > chart.width) {
      for (const point of local.values()) { const t = point[0]; point[0] = point[1]; point[1] = chart.width - t; }
      [chart.width, chart.height] = [chart.height, chart.width];
    }
  }
  return charts;
}

/** Shelf packing at `scale` texels per unit; false if the atlas overflows. */
function pack(charts, sorted, size, scale) {
  let x = 0;
  let y = 0;
  let row = 0;
  for (const i of sorted) {
    const chart = charts[i];
    const w = Math.ceil(chart.width * scale) + 1 + PADDING * 2;
    const h = Math.ceil(chart.height * scale) + 1 + PADDING * 2;
    if (w > size) return false;
    if (x + w > size) { x = 0; y += row; row = 0; }
    if (y + h > size) return false;
    chart.x = x + PADDING;
    chart.y = y + PADDING;
    x += w;
    if (h > row) row = h;
  }
  return true;
}

function packCharts(charts, requested, maximum) {
  const sorted = charts.map((_, i) => i).sort((a, b) => charts[b].height - charts[a].height);
  for (let size = requested; size <= maximum; size *= 2) {
    const longest = charts.reduce((m, c) => Math.max(m, c.width), 1e-9);
    let high = (size - PADDING * 2 - 2) / longest;
    let low = 0;
    for (let i = 0; i < 28; i += 1) {
      const middle = (low + high) / 2;
      if (pack(charts, sorted, size, middle)) low = middle; else high = middle;
    }
    if (low > 0 && pack(charts, sorted, size, low)) return { size, scale: low };
  }
  return null;
}

function dilate(atlas, covered, size, passes = 4) {
  for (let pass = 0; pass < passes; pass += 1) {
    const added = [];
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const at = y * size + x;
        if (covered[at]) continue;
        let r = 0; let g = 0; let b = 0; let n = 0;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= size || yy >= size || !covered[yy * size + xx]) continue;
          const o = (yy * size + xx) * 3;
          r += atlas[o]; g += atlas[o + 1]; b += atlas[o + 2]; n += 1;
        }
        if (!n) continue;
        atlas[at * 3] = r / n; atlas[at * 3 + 1] = g / n; atlas[at * 3 + 2] = b / n;
        added.push(at);
      }
    }
    if (!added.length) break;
    for (const at of added) covered[at] = 1;
  }
}

/** Colour of the source model at a point on one of its faces. */
export function createSourceSampler(source) {
  const bvh = new MeshBVH(source.positions, source.indices);
  const decoded = new Map();
  const imageFor = (prim) => {
    if (!prim.hasUv || prim.imageIndex == null) return null;
    if (!decoded.has(prim.imageIndex)) decoded.set(prim.imageIndex, decodeImage(source.images[prim.imageIndex].bytes));
    return decoded.get(prim.imageIndex);
  };
  const color = [0, 0, 0];
  return {
    bvh,
    sample(hit) {
      const f = hit.face;
      const prim = source.prims[source.facePrim[f]];
      const a = source.indices[f * 3];
      const b = source.indices[f * 3 + 1];
      const c = source.indices[f * 3 + 2];
      const w = 1 - hit.u - hit.v;
      const image = imageFor(prim);
      if (image) {
        const uv = source.uvs;
        return sampleImage(image,
          uv[a * 2] * w + uv[b * 2] * hit.u + uv[c * 2] * hit.v,
          uv[a * 2 + 1] * w + uv[b * 2 + 1] * hit.u + uv[c * 2 + 1] * hit.v, color);
      }
      for (let k = 0; k < 3; k += 1) {
        color[k] = source.colors
          ? source.colors[a * 4 + k] * w + source.colors[b * 4 + k] * hit.u + source.colors[c * 4 + k] * hit.v
          : 148;
      }
      return color;
    },
  };
}

/**
 * Give `mesh` (geometry only) new UVs and bake its atlas from the source.
 *
 * @returns {Promise<{mesh:object, atlasSize:number, charts:number}>}
 */
export async function rebakeAtlas(mesh, sampler, { size, maxSize, encode, onProgress = null }) {
  const charts = unwrap(mesh);
  const packed = packCharts(charts, size, Math.max(size, maxSize));
  if (!packed) throw new Error(`${charts.length} charts do not fit in a ${maxSize}px atlas.`);
  const { size: edge, scale } = packed;
  const faces = mesh.indices.length / 3;
  const vertices = charts.reduce((sum, c) => sum + c.points.size, 0);
  const positions = new Float32Array(vertices * 3);
  const uvs = new Float32Array(vertices * 2);
  const indices = new Uint32Array(faces * 3);
  // Which input face each output face is: charts reorder them.
  const order = new Int32Array(faces);
  const atlas = new Uint8Array(edge * edge * 3);
  const covered = new Uint8Array(edge * edge);
  let vertex = 0;
  let face = 0;
  let hint = -1;
  for (let c = 0; c < charts.length; c += 1) {
    const chart = charts[c];
    const remap = new Map();
    const texel = [];
    for (const [source, [x, y]] of chart.points) {
      remap.set(source, vertex);
      positions.set(mesh.positions.subarray(source * 3, source * 3 + 3), vertex * 3);
      const tx = chart.x + x * scale + 0.5;
      const ty = chart.y + y * scale + 0.5;
      texel[vertex] = [tx, ty];
      uvs[vertex * 2] = tx / edge;
      uvs[vertex * 2 + 1] = ty / edge;
      vertex += 1;
    }
    for (const f of chart.faces) {
      const ids = [0, 1, 2].map((k) => remap.get(mesh.indices[f * 3 + k]));
      indices.set(ids, face * 3);
      order[face] = f;
      face += 1;
      const [[ax, ay], [bx, by], [cx, cy]] = ids.map((i) => texel[i]);
      const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      // Include texels a little outside, so thin triangles still get colour.
      const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx) - 1));
      const x1 = Math.min(edge - 1, Math.ceil(Math.max(ax, bx, cx) + 1));
      const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy) - 1));
      const y1 = Math.min(edge - 1, Math.ceil(Math.max(ay, by, cy) + 1));
      const lengths = [Math.hypot(cx - bx, cy - by), Math.hypot(ax - cx, ay - cy), Math.hypot(bx - ax, by - ay)];
      for (let y = y0; y <= y1; y += 1) {
        for (let x = x0; x <= x1; x += 1) {
          const px = x + 0.5;
          const py = y + 0.5;
          let wa = 1 / 3;
          let wb = 1 / 3;
          if (Math.abs(area) > 1e-9) {
            wa = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
            wb = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
          }
          const wc = 1 - wa - wb;
          // Signed distance outside each edge, in texels.
          const outside = Math.max(
            (-wa * Math.abs(area)) / (lengths[0] || 1),
            (-wb * Math.abs(area)) / (lengths[1] || 1),
            (-wc * Math.abs(area)) / (lengths[2] || 1),
          );
          if (outside > 0.75) continue;
          const at = y * edge + x;
          const inside = outside <= 1e-6;
          if (covered[at] === 1 && !inside) continue;
          let a = Math.max(0, wa);
          let b = Math.max(0, wb);
          let g = Math.max(0, wc);
          const sum = a + b + g || 1;
          a /= sum; b /= sum; g /= sum;
          const p = [0, 1, 2].map((k) => positions[ids[0] * 3 + k] * a + positions[ids[1] * 3 + k] * b + positions[ids[2] * 3 + k] * g);
          const hit = sampler.bvh.closestPoint(p[0], p[1], p[2], Infinity, hint);
          if (!hit) continue;
          hint = hit.face;
          const color = sampler.sample(hit);
          atlas[at * 3] = color[0];
          atlas[at * 3 + 1] = color[1];
          atlas[at * 3 + 2] = color[2];
          covered[at] = 1;
        }
      }
    }
    if (c % 64 === 0) {
      onProgress?.(face / faces);
      await new Promise((resolve) => setImmediate(resolve));
    }
  }
  dilate(atlas, covered, edge);
  const image = await encodeImage({ width: edge, height: edge, data: atlas }, encode);
  return {
    mesh: {
      positions,
      uvs,
      colors: null,
      indices,
      facePrim: new Uint16Array(faces),
      prims: [{ material: { name: 'SkyforgeLodAtlas' }, imageIndex: 0, hasUv: true, hasColor: false }],
      images: [image],
      rootMatrix: mesh.rootMatrix,
      unlit: mesh.unlit,
      passthrough: [],
    },
    atlasSize: edge,
    charts: charts.length,
    order,
  };
}
