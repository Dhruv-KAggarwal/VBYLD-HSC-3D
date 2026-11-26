/**
 * Offline multi-angle renderer for a textured GLB.
 * ---------------------------------------------------------------------------
 * The browser console shows the model interactively, but to *analyse* a
 * reconstruction from the terminal — and to compare two runs side by side — we
 * need flat images from fixed viewpoints. This reads a self-contained GLB
 * (positions + UVs + indices + an embedded texture atlas), rasterises it with a
 * z-buffer, samples the texture through real barycentric UVs, and writes one PNG
 * per requested camera angle.
 *
 * Unlike tools/render.mjs (which assumes the DSM's top-down XZ texture), this
 * honours the per-face UV atlas OpenMVS bakes, so it shows the true textured
 * 3-D mesh exactly as the viewer would.
 *
 *   node tools/render-glb.mjs output/latest/model.glb --out=output/analysis --views=all
 */

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { encodePng, decodePng } from '../lib/png.mjs';
import { _internals } from '../lib/glbtransform.mjs';

const COMPONENT_BYTES = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_COUNT = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

/** Read one glTF accessor out of the BIN chunk as a flat JS number array. */
function readAccessor(json, bin, index) {
  if (index == null) return null;
  const accessor = json.accessors[index];
  const view = json.bufferViews[accessor.bufferView];
  const compBytes = COMPONENT_BYTES[accessor.componentType];
  const comps = TYPE_COUNT[accessor.type];
  const base = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const stride = view.byteStride || compBytes * comps;
  const out = new Array(accessor.count * comps);
  for (let i = 0; i < accessor.count; i += 1) {
    const elementStart = base + i * stride;
    for (let c = 0; c < comps; c += 1) {
      const at = elementStart + c * compBytes;
      let value;
      switch (accessor.componentType) {
        case 5126: value = bin.readFloatLE(at); break;
        case 5125: value = bin.readUInt32LE(at); break;
        case 5123: value = bin.readUInt16LE(at); break;
        case 5121: value = bin.readUInt8(at); break;
        case 5122: value = bin.readInt16LE(at); break;
        case 5120: value = bin.readInt8(at); break;
        default: value = 0;
      }
      out[i * comps + c] = value;
    }
  }
  return out;
}

/** Decode the material's base-colour texture into {width,height,data(rgb)}. */
function readTexture(json, bin) {
  const material = json.materials?.[0];
  const texIndex = material?.pbrMetallicRoughness?.baseColorTexture?.index;
  if (texIndex == null) return null;
  const source = json.textures?.[texIndex]?.source;
  const image = json.images?.[source];
  if (!image || image.bufferView == null) return null;
  const view = json.bufferViews[image.bufferView];
  const png = bin.subarray(view.byteOffset || 0, (view.byteOffset || 0) + view.byteLength);
  try { return decodePng(Buffer.from(png)); } catch { return null; }
}

/** Collapse a hierarchy of node matrices into one per-mesh world transform. */
function worldMatrixFor(json) {
  // Our GLBs are shallow: an optional SkyforgeAlignment parent wrapping the mesh
  // node. Multiply the chain so the render matches the oriented viewer model.
  const sceneNodes = json.scenes?.[json.scene || 0]?.nodes || [];
  const mul = (a, b) => {
    const m = new Array(16).fill(0);
    for (let r = 0; r < 4; r += 1) {
      for (let c = 0; c < 4; c += 1) {
        for (let k = 0; k < 4; k += 1) m[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
      }
    }
    return m;
  };
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  let result = identity;
  const walk = (nodeIndex, parent) => {
    const node = json.nodes[nodeIndex];
    const local = node.matrix || identity;
    const world = mul(parent, local);
    if (node.mesh != null) { result = world; return true; }
    for (const child of node.children || []) if (walk(child, world)) return true;
    return false;
  };
  for (const n of sceneNodes) if (walk(n, identity)) break;
  return result;
}

function applyMatrix(m, x, y, z) {
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (v) => { const l = Math.hypot(...v) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };

const VIEWS = {
  iso: { az: 35, el: 28 },
  top: { az: 0, el: 89 },
  front: { az: 0, el: 8 },
  back: { az: 180, el: 8 },
  left: { az: 90, el: 12 },
  right: { az: 270, el: 12 },
};

async function main() {
  const args = process.argv.slice(2);
  const input = resolve(args.find((a) => !a.startsWith('--')) || 'output/latest/model.glb');
  const outDir = resolve((args.find((a) => a.startsWith('--out=')) || '--out=output/analysis').split('=')[1]);
  const width = Number((args.find((a) => a.startsWith('--w=')) || '--w=1000').split('=')[1]);
  const height = Number((args.find((a) => a.startsWith('--h=')) || '--h=750').split('=')[1]);
  const viewsArg = (args.find((a) => a.startsWith('--views=')) || '--views=iso,top,front,left').split('=')[1];
  const viewNames = viewsArg === 'all' ? Object.keys(VIEWS) : viewsArg.split(',').filter((v) => VIEWS[v]);
  const flipV = args.includes('--flipv');
  const flat = args.includes('--flat'); // ignore texture, show pure geometry
  const supersample = args.includes('--ss'); // bilinear-ish 2x2 texture average
  await mkdir(outDir, { recursive: true });

  const { json, bin } = _internals.parseGlb(await readFile(input));
  const prim = json.meshes[0].primitives[0];
  const world = worldMatrixFor(json);
  const rawPos = readAccessor(json, bin, prim.attributes.POSITION);
  const uvs = readAccessor(json, bin, prim.attributes.TEXCOORD_0);
  const indices = readAccessor(json, bin, prim.indices);
  const texture = readTexture(json, bin);
  // Vertex colours (COLOR_0) are used when the mesh has no texture atlas.
  const vColsRaw = readAccessor(json, bin, prim.attributes.COLOR_0);
  const vColComps = prim.attributes.COLOR_0 != null ? TYPE_COUNT[json.accessors[prim.attributes.COLOR_0].type] : 0;
  // uint8 colours are already 0..255; only float (5126) colours are 0..1 and
  // need scaling up for the 8-bit render buffer.
  const vColFloat = prim.attributes.COLOR_0 != null && json.accessors[prim.attributes.COLOR_0].componentType === 5126;
  const vertexColor = (i) => {
    if (!vColsRaw) return null;
    const o = i * vColComps;
    const s = vColFloat ? 255 : 1;
    return [vColsRaw[o] * s, vColsRaw[o + 1] * s, vColsRaw[o + 2] * s];
  };
  const vertexCount = rawPos.length / 3;

  // Bake the world transform into positions once, then every view reuses them.
  const positions = new Float32Array(rawPos.length);
  for (let i = 0; i < vertexCount; i += 1) {
    const w = applyMatrix(world, rawPos[i * 3], rawPos[i * 3 + 1], rawPos[i * 3 + 2]);
    positions[i * 3] = w[0]; positions[i * 3 + 1] = w[1]; positions[i * 3 + 2] = w[2];
  }

  const box = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (let i = 0; i < vertexCount; i += 1) {
    for (let k = 0; k < 3; k += 1) {
      box.min[k] = Math.min(box.min[k], positions[i * 3 + k]);
      box.max[k] = Math.max(box.max[k], positions[i * 3 + k]);
    }
  }
  const center = [0, 1, 2].map((k) => (box.min[k] + box.max[k]) / 2);
  const radius = Math.max(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]) * 0.5 || 1;

  console.log(`Model: ${vertexCount} verts, ${indices.length / 3} faces, texture ${texture ? `${texture.width}x${texture.height}` : 'none'}`);
  console.log(`Extent: X=${(box.max[0] - box.min[0]).toFixed(1)} Y=${(box.max[1] - box.min[1]).toFixed(1)} Z=${(box.max[2] - box.min[2]).toFixed(1)} m`);

  for (const name of viewNames) {
    const { az, el } = VIEWS[name];
    const azr = az * Math.PI / 180;
    const elr = el * Math.PI / 180;
    const dist = radius * 2.6;
    const eye = [
      center[0] + dist * Math.cos(elr) * Math.sin(azr),
      center[1] + dist * Math.sin(elr),
      center[2] + dist * Math.cos(elr) * Math.cos(azr),
    ];
    const forward = norm(sub(center, eye));
    const right = norm(cross(forward, [0, 1, 0]));
    const camUp = cross(right, forward);
    const focal = (height * 0.5) / Math.tan((42 * Math.PI / 180) / 2);

    const projected = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i += 1) {
      const p = sub([positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]], eye);
      const cz = dot(p, forward);
      projected[i * 3] = cz > 1e-4 ? width / 2 + (focal * dot(p, right)) / cz : NaN;
      projected[i * 3 + 1] = cz > 1e-4 ? height / 2 - (focal * dot(p, camUp)) / cz : NaN;
      projected[i * 3 + 2] = cz;
    }

    const sun = norm([0.4, 0.85, 0.35]);
    const depth = new Float32Array(width * height).fill(Infinity);
    const pixels = Buffer.alloc(width * height * 3);
    for (let i = 0; i < pixels.length; i += 3) {
      const t = Math.floor(i / 3 / width) / height;
      pixels[i] = Math.round(125 + 55 * (1 - t));
      pixels[i + 1] = Math.round(150 + 48 * (1 - t));
      pixels[i + 2] = Math.round(178 + 42 * (1 - t));
    }

    let drawn = 0;
    for (let f = 0; f < indices.length; f += 3) {
      const [ia, ib, ic] = [indices[f], indices[f + 1], indices[f + 2]];
      const ax = projected[ia * 3]; const ay = projected[ia * 3 + 1]; const az2 = projected[ia * 3 + 2];
      const bx = projected[ib * 3]; const by = projected[ib * 3 + 1]; const bz = projected[ib * 3 + 2];
      const cx = projected[ic * 3]; const cy = projected[ic * 3 + 1]; const cz = projected[ic * 3 + 2];
      if (!Number.isFinite(ax) || !Number.isFinite(bx) || !Number.isFinite(cx)) continue;
      const minX = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
      const maxX = Math.min(width - 1, Math.ceil(Math.max(ax, bx, cx)));
      const minY = Math.max(0, Math.floor(Math.min(ay, by, cy)));
      const maxY = Math.min(height - 1, Math.ceil(Math.max(ay, by, cy)));
      if (minX > maxX || minY > maxY) continue;
      const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (Math.abs(area) < 1e-9) continue;

      // Flat lambert from the face normal — enough to read relief in the render.
      const e1 = sub([positions[ib * 3], positions[ib * 3 + 1], positions[ib * 3 + 2]], [positions[ia * 3], positions[ia * 3 + 1], positions[ia * 3 + 2]]);
      const e2 = sub([positions[ic * 3], positions[ic * 3 + 1], positions[ic * 3 + 2]], [positions[ia * 3], positions[ia * 3 + 1], positions[ia * 3 + 2]]);
      const n = norm(cross(e1, e2));
      const shade = 0.55 + 0.45 * Math.abs(dot(n, sun));

      for (let y = minY; y <= maxY; y += 1) {
        for (let x = minX; x <= maxX; x += 1) {
          const px = x + 0.5; const py = y + 0.5;
          const w0 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / area;
          const w1 = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / area;
          if (w0 < 0 || w1 < 0 || w0 + w1 > 1) continue;
          const z = az2 + w1 * (bz - az2) + w0 * (cz - az2);
          const key = y * width + x;
          if (z >= depth[key]) continue;
          depth[key] = z;
          const wA = 1 - w0 - w1;
          let cr = 190; let cg = 188; let cb = 182; // neutral stone for flat/geometry mode
          if (!flat && texture && uvs) {
            let u = wA * uvs[ia * 2] + w1 * uvs[ib * 2] + w0 * uvs[ic * 2];
            let v = wA * uvs[ia * 2 + 1] + w1 * uvs[ib * 2 + 1] + w0 * uvs[ic * 2 + 1];
            if (flipV) v = 1 - v;
            u -= Math.floor(u); v -= Math.floor(v);
            // Average a small neighbourhood so the crude nearest-neighbour sample
            // does not fall into the black gutter between tiny atlas patches.
            let rr = 0; let gg = 0; let bb = 0; let hit = 0;
            const span = supersample ? 1 : 0;
            const cxp = u * texture.width; const cyp = v * texture.height;
            for (let oy = -span; oy <= span; oy += 1) {
              for (let ox = -span; ox <= span; ox += 1) {
                const tx = Math.min(texture.width - 1, Math.max(0, Math.round(cxp + ox)));
                const ty = Math.min(texture.height - 1, Math.max(0, Math.round(cyp + oy)));
                const ti = (ty * texture.width + tx) * 3;
                const lum = texture.data[ti] + texture.data[ti + 1] + texture.data[ti + 2];
                if (lum > 12) { rr += texture.data[ti]; gg += texture.data[ti + 1]; bb += texture.data[ti + 2]; hit += 1; }
              }
            }
            if (hit) { cr = rr / hit; cg = gg / hit; cb = bb / hit; }
            else { const ti = (Math.min(texture.height - 1, Math.round(cyp)) * texture.width + Math.min(texture.width - 1, Math.round(cxp))) * 3; cr = texture.data[ti]; cg = texture.data[ti + 1]; cb = texture.data[ti + 2]; }
          } else if (!flat && vColsRaw) {
            const ca = vertexColor(ia); const cb2 = vertexColor(ib); const cc2 = vertexColor(ic);
            cr = wA * ca[0] + w1 * cb2[0] + w0 * cc2[0];
            cg = wA * ca[1] + w1 * cb2[1] + w0 * cc2[1];
            cb = wA * ca[2] + w1 * cb2[2] + w0 * cc2[2];
          }
          pixels[key * 3] = Math.min(255, Math.round(cr * shade));
          pixels[key * 3 + 1] = Math.min(255, Math.round(cg * shade));
          pixels[key * 3 + 2] = Math.min(255, Math.round(cb * shade));
        }
      }
      drawn += 1;
    }
    const outPath = resolve(outDir, `${name}.png`);
    await writeFile(outPath, encodePng(pixels, width, height));
    console.log(`  ${name.padEnd(6)} -> ${outPath} (${drawn} faces)`);
  }
}

main().catch((error) => { console.error(error); process.exit(1); });
