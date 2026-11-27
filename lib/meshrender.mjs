/**
 * Software rasteriser for a self-contained GLB.
 * ---------------------------------------------------------------------------
 * This is the rendering core that `tools/render-glb.mjs` grew: parse a GLB,
 * bake the node hierarchy into world-space positions, then z-buffer the
 * triangles sampling the embedded texture atlas through real barycentric UVs.
 *
 * It was extracted into lib/ because the turntable video (lib/turntable.mjs)
 * needs exactly the same renderer, just driven around a circle instead of
 * through a fixed list of viewpoints. There is deliberately no GPU and no
 * native dependency here: it runs wherever Node runs, which is what makes the
 * turntable safe to attach to the end of every job.
 *
 * Nothing in here allocates per frame except the frame buffers themselves, so
 * rendering 150 frames costs 150 rasterisations and one model parse.
 */

import { readFile } from 'node:fs/promises';
import { encodePng, decodePng } from './png.mjs';
import { _internals } from './glbtransform.mjs';

const COMPONENT_BYTES = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_COUNT = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Read one glTF accessor out of the BIN chunk as a flat JS number array. */
export function readAccessor(json, bin, index) {
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
export function readTexture(json, bin) {
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

const mul = (a, b) => {
  const m = new Array(16).fill(0);
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      for (let k = 0; k < 4; k += 1) m[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
    }
  }
  return m;
};

/** Collapse a hierarchy of node matrices into one per-mesh world transform. */
export function worldMatrixFor(json) {
  // Our GLBs are shallow: an optional SkyforgeAlignment parent wrapping the
  // mesh node. Multiply the chain so the render matches the oriented viewer.
  const sceneNodes = json.scenes?.[json.scene || 0]?.nodes || [];
  let result = IDENTITY;
  const walk = (nodeIndex, parent) => {
    const node = json.nodes[nodeIndex];
    const world = mul(parent, node.matrix || IDENTITY);
    if (node.mesh != null) { result = world; return true; }
    for (const child of node.children || []) if (walk(child, world)) return true;
    return false;
  };
  for (const n of sceneNodes) if (walk(n, IDENTITY)) break;
  return result;
}

export function applyMatrix(m, x, y, z) {
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

/**
 * Parse a GLB into everything the rasteriser needs, once.
 *
 * Positions come back already in world space, so each subsequent frame only
 * has to project and fill. `radius` and `center` describe the bounding box and
 * are what a camera orbit should be sized against.
 */
export async function loadRenderModel(glbPath) {
  const { json, bin } = _internals.parseGlb(await readFile(glbPath));
  const prim = json.meshes?.[0]?.primitives?.[0];
  if (!prim) throw new Error('GLB contains no mesh primitive.');
  const world = worldMatrixFor(json);
  const rawPos = readAccessor(json, bin, prim.attributes.POSITION);
  const uvs = readAccessor(json, bin, prim.attributes.TEXCOORD_0);
  const indices = readAccessor(json, bin, prim.indices);
  if (!rawPos?.length || !indices?.length) throw new Error('GLB has no positions or indices.');
  const texture = readTexture(json, bin);

  // Vertex colours are the fallback when there is no atlas. uint8 colours are
  // already 0..255; only float colours are 0..1 and need scaling.
  const colorIndex = prim.attributes.COLOR_0;
  const colors = readAccessor(json, bin, colorIndex);
  const colorComps = colorIndex != null ? TYPE_COUNT[json.accessors[colorIndex].type] : 0;
  const colorFloat = colorIndex != null && json.accessors[colorIndex].componentType === 5126;

  const vertexCount = rawPos.length / 3;
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
  const radius = Math.max(
    box.max[0] - box.min[0],
    box.max[1] - box.min[1],
    box.max[2] - box.min[2],
  ) * 0.5 || 1;

  return {
    positions,
    uvs,
    indices,
    texture,
    colors,
    colorComps,
    colorFloat,
    vertexCount,
    faceCount: indices.length / 3,
    box,
    center,
    radius,
  };
}

const SUN = norm([0.4, 0.85, 0.35]);

/**
 * How hard to shade, and why it depends on what we are shading.
 *
 * A photographic atlas already contains the real illumination of the scene:
 * the sun was in the sky when the drone flew, so roofs are already bright on
 * one side and walls already fall into shadow. Multiplying a second lambert
 * term on top of that shades everything twice, which is what made the render
 * look dark and muddy rather than lit -- the old single strength took every
 * surface facing away from SUN down to 55% of its photographed colour.
 *
 * So: barely shade a textured model (just enough that relief still reads), and
 * shade a flat or vertex-coloured one properly, because there is no light baked
 * into a point-cloud colour or into flat stone grey.
 */
const SHADE_TEXTURED = { ambient: 0.88, direct: 0.12 };
const SHADE_UNTEXTURED = { ambient: 0.55, direct: 0.45 };

/**
 * sRGB <-> linear-light conversion, as 8-bit lookup tables.
 *
 * Multiplying an sRGB value by a shading factor is wrong: sRGB is a perceptual
 * encoding, not light, so halving the number does not halve the brightness. It
 * crushes midtones and skews hue, because each channel bends by a different
 * amount. Converting to linear light, multiplying there, and encoding back is
 * the correct operation and costs two table lookups per channel.
 */
const SRGB_TO_LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i += 1) {
  const c = i / 255;
  SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}
/** Clamp and round a possibly-fractional channel to a 0..255 table index. */
function level(x) {
  const v = Math.round(x);
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/** Encode a linear 0..1 value back to an 8-bit sRGB level. */
function linearToSrgb8(x) {
  const c = x <= 0 ? 0 : x >= 1 ? 1 : x;
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
  return Math.round(v * 255);
}

/**
 * Rasterise one view of a loaded model into a raw RGB buffer.
 *
 * `az`/`el` are degrees around and above the model centre; `distanceScale`
 * multiplies the auto-framed orbit radius. The caller owns the returned
 * buffer, which is `width * height * 3` bytes.
 */
export function renderView(model, {
  width = 1280,
  height = 720,
  az = 35,
  el = 28,
  distanceScale = 2.6,
  fov = 42,
  flat = false,
  flipV = false,
  supersample = true,
  background = null,
} = {}) {
  const { positions, uvs, indices, texture, vertexCount } = model;
  const azr = az * Math.PI / 180;
  const elr = el * Math.PI / 180;
  const dist = model.radius * distanceScale;
  const center = model.center;
  const eye = [
    center[0] + dist * Math.cos(elr) * Math.sin(azr),
    center[1] + dist * Math.sin(elr),
    center[2] + dist * Math.cos(elr) * Math.cos(azr),
  ];
  const forward = norm(sub(center, eye));
  // A camera looking straight down has no stable "right" from world up; nudge
  // the reference so the cross product does not collapse to zero.
  const upRef = Math.abs(forward[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
  const right = norm(cross(forward, upRef));
  const camUp = cross(right, forward);
  const focal = (height * 0.5) / Math.tan((fov * Math.PI / 180) / 2);

  const projected = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i += 1) {
    const p = sub([positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]], eye);
    const cz = dot(p, forward);
    projected[i * 3] = cz > 1e-4 ? width / 2 + (focal * dot(p, right)) / cz : NaN;
    projected[i * 3 + 1] = cz > 1e-4 ? height / 2 - (focal * dot(p, camUp)) / cz : NaN;
    projected[i * 3 + 2] = cz;
  }

  const depth = new Float32Array(width * height).fill(Infinity);
  const pixels = Buffer.alloc(width * height * 3);
  const sky = background || [[125, 150, 178], [180, 198, 220]];
  for (let i = 0; i < pixels.length; i += 3) {
    const t = Math.floor(i / 3 / width) / height;
    pixels[i] = Math.round(sky[0][0] + (sky[1][0] - sky[0][0]) * (1 - t));
    pixels[i + 1] = Math.round(sky[0][1] + (sky[1][1] - sky[0][1]) * (1 - t));
    pixels[i + 2] = Math.round(sky[0][2] + (sky[1][2] - sky[0][2]) * (1 - t));
  }

  const vertexColor = (i) => {
    if (!model.colors) return null;
    const o = i * model.colorComps;
    const s = model.colorFloat ? 255 : 1;
    return [model.colors[o] * s, model.colors[o + 1] * s, model.colors[o + 2] * s];
  };

  let drawn = 0;
  for (let f = 0; f < indices.length; f += 3) {
    const ia = indices[f]; const ib = indices[f + 1]; const ic = indices[f + 2];
    const ax = projected[ia * 3]; const ay = projected[ia * 3 + 1]; const azz = projected[ia * 3 + 2];
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
    const pa = [positions[ia * 3], positions[ia * 3 + 1], positions[ia * 3 + 2]];
    const e1 = sub([positions[ib * 3], positions[ib * 3 + 1], positions[ib * 3 + 2]], pa);
    const e2 = sub([positions[ic * 3], positions[ic * 3 + 1], positions[ic * 3 + 2]], pa);
    const n = norm(cross(e1, e2));
    const lit = !flat && texture && uvs ? SHADE_TEXTURED : SHADE_UNTEXTURED;
    const shade = lit.ambient + lit.direct * Math.abs(dot(n, SUN));

    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const px = x + 0.5; const py = y + 0.5;
        const w0 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / area;
        const w1 = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / area;
        if (w0 < 0 || w1 < 0 || w0 + w1 > 1) continue;
        const z = azz + w1 * (bz - azz) + w0 * (cz - azz);
        const key = y * width + x;
        if (z >= depth[key]) continue;
        depth[key] = z;
        const wA = 1 - w0 - w1;
        let cr = 190; let cg = 188; let cb = 182; // neutral stone for flat mode
        if (!flat && texture && uvs) {
          let u = wA * uvs[ia * 2] + w1 * uvs[ib * 2] + w0 * uvs[ic * 2];
          let v = wA * uvs[ia * 2 + 1] + w1 * uvs[ib * 2 + 1] + w0 * uvs[ic * 2 + 1];
          if (flipV) v = 1 - v;
          u -= Math.floor(u); v -= Math.floor(v);
          // Average a small neighbourhood so the crude nearest-neighbour
          // sample does not fall into the gutter between tiny atlas patches.
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
          if (hit) { cr = rr / hit; cg = gg / hit; cb = bb / hit; } else {
            const ti = (Math.min(texture.height - 1, Math.round(cyp)) * texture.width
              + Math.min(texture.width - 1, Math.round(cxp))) * 3;
            cr = texture.data[ti]; cg = texture.data[ti + 1]; cb = texture.data[ti + 2];
          }
        } else if (!flat && model.colors) {
          const ca = vertexColor(ia); const cb2 = vertexColor(ib); const cc2 = vertexColor(ic);
          cr = wA * ca[0] + w1 * cb2[0] + w0 * cc2[0];
          cg = wA * ca[1] + w1 * cb2[1] + w0 * cc2[1];
          cb = wA * ca[2] + w1 * cb2[2] + w0 * cc2[2];
        }
        // Shade in linear light, not in sRGB; see SRGB_TO_LINEAR above. The
        // channels are averaged texels, so they are floats and can sit a hair
        // outside 0..255; clamp rather than mask, because a bitwise mask would
        // wrap an overshoot to near-black instead of saturating it to white.
        pixels[key * 3] = linearToSrgb8(SRGB_TO_LINEAR[level(cr)] * shade);
        pixels[key * 3 + 1] = linearToSrgb8(SRGB_TO_LINEAR[level(cg)] * shade);
        pixels[key * 3 + 2] = linearToSrgb8(SRGB_TO_LINEAR[level(cb)] * shade);
      }
    }
    drawn += 1;
  }
  return { pixels, width, height, drawn };
}

/** Convenience: render a view straight to PNG bytes. */
export function renderViewPng(model, options) {
  const { pixels, width, height } = renderView(model, options);
  return encodePng(pixels, width, height);
}

export const VIEWS = {
  iso: { az: 35, el: 28 },
  top: { az: 0, el: 89 },
  front: { az: 0, el: 8 },
  back: { az: 180, el: 8 },
  left: { az: 90, el: 12 },
  right: { az: 270, el: 12 },
};
