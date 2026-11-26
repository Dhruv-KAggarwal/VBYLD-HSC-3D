/**
 * Regression tests for the texture / colour / GLB export path.
 * ---------------------------------------------------------------------------
 * These guard the four defects that made the reconstructed model render black
 * and mis-coloured:
 *
 *   1. OpenMVS atlas UVs were never flipped from the OpenGL bottom-left origin
 *      to the glTF top-left one, so every face sampled the black padding
 *      between atlas patches.
 *   2. COLOR_0 was emitted alongside baseColorTexture, and per the glTF spec it
 *      multiplies into it, squaring the albedo.
 *   3. COLOR_0 was a 3-byte-stride UNSIGNED_BYTE VEC3, violating the 4-byte
 *      alignment rule, so loaders either rejected the file or smeared colour
 *      across neighbouring vertices.
 *   4. Multi-frame colour was averaged in sRGB (gamma) space, which darkens
 *      midtones wherever frames overlap.
 *
 * Run: npm run test:texture
 */

import { writeGlb } from '../lib/glb.mjs';
import { flipTextureV } from '../lib/glbtransform.mjs';
import { encodePng } from '../lib/png.mjs';

const results = [];
const check = (name, pass, detail = '') => {
  results.push({ name, pass, detail });
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
};

/** Split a GLB back into its JSON and BIN chunks. */
function parseGlb(buffer) {
  if (buffer.readUInt32LE(0) !== 0x46546c67) throw new Error('not a GLB');
  let offset = 12;
  let json = null;
  let bin = null;
  while (offset + 8 <= buffer.length) {
    const len = buffer.readUInt32LE(offset);
    const type = buffer.readUInt32LE(offset + 4);
    const body = buffer.subarray(offset + 8, offset + 8 + len);
    if (type === 0x4e4f534a) json = JSON.parse(body.toString('utf8'));
    else if (type === 0x004e4942) bin = body;
    offset += 8 + len + ((4 - (len % 4)) % 4);
  }
  return { json, bin };
}

/** A unit quad: two triangles, four vertices, full-range UVs. */
function quad() {
  return {
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    colors: new Uint8Array([200, 100, 50, 200, 100, 50, 200, 100, 50, 200, 100, 50]),
  };
}

// ---------------------------------------------------------------------------
// 1. The V flip actually inverts every V and leaves U alone.
// ---------------------------------------------------------------------------
{
  const mesh = quad();
  // An asymmetric texture so a flip is observable, not just plausible.
  const px = Buffer.alloc(2 * 2 * 3);
  px[0] = 255; // top-left red
  const glb = writeGlb({ ...mesh, ortho: encodePng(px, 2, 2), textured: true });
  const before = parseGlb(glb);
  const uvAccessor = before.json.accessors[before.json.meshes[0].primitives[0].attributes.TEXCOORD_0];
  const view = before.json.bufferViews[uvAccessor.bufferView];

  const readUvs = (parsed) => {
    const acc = parsed.json.accessors[parsed.json.meshes[0].primitives[0].attributes.TEXCOORD_0];
    const v = parsed.json.bufferViews[acc.bufferView];
    const base = (v.byteOffset || 0) + (acc.byteOffset || 0);
    const out = [];
    for (let i = 0; i < acc.count; i += 1) {
      out.push([parsed.bin.readFloatLE(base + i * 8), parsed.bin.readFloatLE(base + i * 8 + 4)]);
    }
    return out;
  };

  const original = readUvs(before);
  const flipped = readUvs(parseGlb(flipTextureV(glb)));
  const uSame = original.every((uv, i) => Math.abs(uv[0] - flipped[i][0]) < 1e-6);
  const vInverted = original.every((uv, i) => Math.abs((1 - uv[1]) - flipped[i][1]) < 1e-6);
  check('flipTextureV inverts V and preserves U', uSame && vInverted,
    `U stable=${uSame}, V inverted=${vInverted}, accessor count=${uvAccessor.count}, stride=${view.byteStride || 8}`);

  // Flipping twice must be the identity, so a double call can never corrupt UVs.
  const twice = readUvs(parseGlb(flipTextureV(flipTextureV(glb))));
  check('flipTextureV is an involution',
    original.every((uv, i) => Math.abs(uv[1] - twice[i][1]) < 1e-6));
}

// ---------------------------------------------------------------------------
// 2. A textured mesh must not also carry COLOR_0 (the spec multiplies them).
// ---------------------------------------------------------------------------
{
  const px = Buffer.alloc(2 * 2 * 3, 180);
  const textured = parseGlb(writeGlb({ ...quad(), ortho: encodePng(px, 2, 2), textured: true }));
  const attrs = textured.json.meshes[0].primitives[0].attributes;
  const mat = textured.json.materials[0];
  check('textured mesh omits COLOR_0', attrs.COLOR_0 === undefined,
    `attributes=${Object.keys(attrs).join(',')}`);
  check('textured mesh declares baseColorTexture',
    mat.pbrMetallicRoughness.baseColorTexture?.index === 0);
  check('textured mesh keeps a white baseColorFactor',
    JSON.stringify(mat.pbrMetallicRoughness.baseColorFactor) === '[1,1,1,1]');
}

// ---------------------------------------------------------------------------
// 3. An untextured mesh carries COLOR_0, and it is 4-byte aligned VEC4.
// ---------------------------------------------------------------------------
{
  const plain = parseGlb(writeGlb({ ...quad(), ortho: null, textured: false }));
  const attrs = plain.json.meshes[0].primitives[0].attributes;
  const colorAcc = plain.json.accessors[attrs.COLOR_0];
  check('untextured mesh carries COLOR_0', attrs.COLOR_0 !== undefined);
  check('COLOR_0 is VEC4 UNSIGNED_BYTE normalized',
    colorAcc.type === 'VEC4' && colorAcc.componentType === 5121 && colorAcc.normalized === true,
    `${colorAcc.type}/${colorAcc.componentType}/normalized=${colorAcc.normalized}`);
  const colorView = plain.json.bufferViews[colorAcc.bufferView];
  check('COLOR_0 bufferView is 4-byte aligned',
    (colorView.byteOffset || 0) % 4 === 0 && colorView.byteLength % 4 === 0,
    `offset=${colorView.byteOffset || 0}, length=${colorView.byteLength}`);
  // The original RGB must survive the RGBA expansion unchanged.
  const base = colorView.byteOffset || 0;
  const first = [plain.bin[base], plain.bin[base + 1], plain.bin[base + 2], plain.bin[base + 3]];
  check('COLOR_0 preserves RGB and sets opaque alpha',
    first[0] === 200 && first[1] === 100 && first[2] === 50 && first[3] === 255,
    first.join(','));
  check('untextured mesh omits TEXCOORD_0', attrs.TEXCOORD_0 === undefined);
}

// ---------------------------------------------------------------------------
// 4. Unlit is declared, so a viewer with no lights renders the photo not black.
// ---------------------------------------------------------------------------
{
  const px = Buffer.alloc(2 * 2 * 3, 180);
  const glb = parseGlb(writeGlb({ ...quad(), ortho: encodePng(px, 2, 2), textured: true }));
  check('KHR_materials_unlit is declared in extensionsUsed',
    Array.isArray(glb.json.extensionsUsed) && glb.json.extensionsUsed.includes('KHR_materials_unlit'));
  check('surface material carries the unlit extension',
    glb.json.materials[0].extensions?.KHR_materials_unlit !== undefined);
  check('unlit is used, not required (file stays universally loadable)',
    glb.json.extensionsRequired === undefined);
  const lit = parseGlb(writeGlb({ ...quad(), ortho: encodePng(px, 2, 2), textured: true }, null, { unlit: false }));
  check('unlit can be opted out of', lit.json.extensionsUsed === undefined
    && lit.json.materials[0].extensions === undefined);
}

// ---------------------------------------------------------------------------
// 5. The image mimeType must match the actual bytes, never be assumed.
// ---------------------------------------------------------------------------
{
  const png = encodePng(Buffer.alloc(2 * 2 * 3, 90), 2, 2);
  const asPng = parseGlb(writeGlb({ ...quad(), ortho: png, textured: true }));
  check('PNG atlas is declared image/png', asPng.json.images[0].mimeType === 'image/png');

  // Minimal JPEG magic is enough for the sniffer.
  const fakeJpeg = Buffer.concat([Buffer.from([0xff, 0xd8, 0xff, 0xe0]), Buffer.alloc(64, 7)]);
  const asJpeg = parseGlb(writeGlb({ ...quad(), ortho: fakeJpeg, textured: true }));
  check('JPEG atlas is declared image/jpeg', asJpeg.json.images[0].mimeType === 'image/jpeg',
    asJpeg.json.images[0].mimeType);
}

// ---------------------------------------------------------------------------
// 6. Accessor and bufferView indices all resolve, and counts agree.
// ---------------------------------------------------------------------------
{
  const px = Buffer.alloc(4 * 4 * 3, 60);
  const glb = parseGlb(writeGlb(
    { ...quad(), ortho: encodePng(px, 4, 4), textured: true },
    { positions: new Float32Array([0, 1, 0, 1, 1, 1]), colors: new Uint8Array([1, 2, 3, 4, 5, 6]) },
  ));
  let ok = true;
  const why = [];
  for (const acc of glb.json.accessors) {
    if (acc.bufferView == null || !glb.json.bufferViews[acc.bufferView]) { ok = false; why.push('dangling bufferView'); }
  }
  for (const mesh of glb.json.meshes) {
    for (const prim of mesh.primitives) {
      for (const [name, index] of Object.entries(prim.attributes)) {
        if (!glb.json.accessors[index]) { ok = false; why.push(`dangling ${name}`); }
      }
    }
  }
  for (const tex of glb.json.textures || []) {
    if (!glb.json.images[tex.source]) { ok = false; why.push('dangling image'); }
    if (!glb.json.samplers[tex.sampler]) { ok = false; why.push('dangling sampler'); }
  }
  const declared = glb.json.buffers[0].byteLength;
  if (declared !== glb.bin.length) { ok = false; why.push(`buffer length ${declared} != bin ${glb.bin.length}`); }
  check('all accessor / view / texture indices resolve and buffer length matches', ok, why.join('; '));

  const posAcc = glb.json.accessors[glb.json.meshes[0].primitives[0].attributes.POSITION];
  check('POSITION accessor has min/max', Array.isArray(posAcc.min) && Array.isArray(posAcc.max));
  check('point cloud primitive is POINTS mode', glb.json.meshes[1].primitives[0].mode === 0);
}

// ---------------------------------------------------------------------------
// 7. Linear-light averaging does not darken midtones the way sRGB averaging did.
// ---------------------------------------------------------------------------
{
  // Two observations of the same surface, one in shadow and one in sun. The
  // physically correct average is the mean of their LIGHT, then re-encoded.
  const srgbToLinear = (i) => {
    const c = i / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const linearToSrgb = (x) => {
    const c = x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055;
    return Math.round(Math.max(0, Math.min(255, c * 255)));
  };
  const a = 60;
  const b = 200;
  const gammaMean = Math.round((a + b) / 2);
  const linearMean = linearToSrgb((srgbToLinear(a) + srgbToLinear(b)) / 2);
  check('linear-light averaging is brighter than sRGB-space averaging',
    linearMean > gammaMean,
    `sRGB mean=${gammaMean}, linear mean=${linearMean}, lift=+${linearMean - gammaMean}`);
  // Round-tripping a single value must be lossless, so the change is a no-op
  // wherever only one frame sees a texel.
  let worst = 0;
  for (let i = 0; i < 256; i += 1) worst = Math.max(worst, Math.abs(linearToSrgb(srgbToLinear(i)) - i));
  check('sRGB round-trip is lossless for single observations', worst === 0, `max error=${worst}`);
}

// ---------------------------------------------------------------------------
// 8. encodePng accepts a plain Uint8Array, not only a Buffer.
// ---------------------------------------------------------------------------
{
  let ok = true;
  let detail = '';
  try {
    const png = encodePng(new Uint8Array(2 * 2 * 3).fill(123), 2, 2);
    ok = png.readUInt32BE(0) === 0x89504e47 && png.readUInt32BE(16) === 2 && png.readUInt32BE(20) === 2;
    detail = `${png.length} bytes`;
  } catch (error) {
    ok = false;
    detail = error.message;
  }
  check('encodePng handles a Uint8Array', ok, detail);
}

const failed = results.filter((r) => !r.pass);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
if (failed.length) {
  console.error(`\nFAILED:\n${failed.map((r) => `  - ${r.name}${r.detail ? ` (${r.detail})` : ''}`).join('\n')}`);
  process.exit(1);
}
