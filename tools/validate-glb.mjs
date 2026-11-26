/**
 * Inspect a GLB for the defects that made Skyforge models render black.
 *
 * Checks, in the order they historically bit us:
 *   1. UV orientation — OpenMVS bakes a bottom-left origin, glTF is top-left.
 *      An unflipped atlas samples the padding between patches and the model is
 *      black. Detected statistically: a correctly flipped atlas has most of its
 *      V coordinates spread across the sheet, and we report the V histogram so
 *      a human can see where the faces are actually sampling.
 *   2. COLOR_0 alongside baseColorTexture — the spec multiplies them, so the
 *      albedo is squared and the surface comes out dark.
 *   3. COLOR_0 alignment — a 3-byte ubyte VEC3 breaks the 4-byte rule.
 *   4. mimeType vs the actual image bytes.
 *   5. KHR_materials_unlit — without it, a viewer with no lights renders black.
 *   6. Dangling accessor / bufferView / texture indices.
 *
 * Usage: node tools/validate-glb.mjs <file.glb> [...]
 */

import { readFileSync, existsSync } from 'node:fs';

function parseGlb(buffer) {
  if (buffer.length < 12 || buffer.readUInt32LE(0) !== 0x46546c67) throw new Error('not a GLB');
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

function sniffMime(bytes) {
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return 'image/png';
  if (bytes[0] === 0xff && bytes[1] === 0xd8) return 'image/jpeg';
  if (bytes[0] === 0x52 && bytes[1] === 0x49) return 'image/webp';
  return 'unknown';
}

function readUvs(json, bin, accessorIndex, limit = 200000) {
  const accessor = json.accessors[accessorIndex];
  if (accessor.componentType !== 5126) return null;
  const view = json.bufferViews[accessor.bufferView];
  const base = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const stride = view.byteStride || 8;
  const count = Math.min(accessor.count, limit);
  const us = new Float32Array(count);
  const vs = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    us[i] = bin.readFloatLE(base + i * stride);
    vs[i] = bin.readFloatLE(base + i * stride + 4);
  }
  return { us, vs, count: accessor.count };
}

function report(file) {
  const problems = [];
  const notes = [];
  const buffer = readFileSync(file);
  const { json, bin } = parseGlb(buffer);

  const unlit = (json.extensionsUsed || []).includes('KHR_materials_unlit');
  notes.push(`size ${(buffer.length / 1e6).toFixed(1)} MB · generator ${json.asset?.generator || 'unknown'}`);
  notes.push(`meshes ${json.meshes?.length || 0} · materials ${json.materials?.length || 0} · images ${json.images?.length || 0}`);

  // --- images
  for (const [index, image] of (json.images || []).entries()) {
    if (image.bufferView == null) continue;
    const view = json.bufferViews[image.bufferView];
    const bytes = bin.subarray(view.byteOffset || 0, (view.byteOffset || 0) + view.byteLength);
    const actual = sniffMime(bytes);
    let dims = '?';
    if (actual === 'image/png') dims = `${bytes.readUInt32BE(16)}x${bytes.readUInt32BE(20)}`;
    notes.push(`image[${index}] ${dims} declared ${image.mimeType} actual ${actual} (${(view.byteLength / 1e6).toFixed(2)} MB)`);
    if (image.mimeType !== actual && actual !== 'unknown') {
      problems.push(`image[${index}] declares ${image.mimeType} but the bytes are ${actual} — most loaders fail the decode and fall back to black`);
    }
    const perPixel = dims !== '?' ? view.byteLength / (bytes.readUInt32BE(16) * bytes.readUInt32BE(20)) : null;
    if (perPixel !== null) {
      notes.push(`image[${index}] ${perPixel.toFixed(4)} compressed bytes/pixel`);
      if (perPixel < 0.02) problems.push(`image[${index}] is only ${perPixel.toFixed(4)} B/px — that is a near-uniform (blank) atlas, not photography`);
    }
  }

  // --- materials
  for (const [index, material] of (json.materials || []).entries()) {
    const pbr = material.pbrMetallicRoughness || {};
    const hasTexture = pbr.baseColorTexture != null;
    const materialUnlit = material.extensions?.KHR_materials_unlit != null;
    notes.push(
      `material[${index}] "${material.name || ''}" texture=${hasTexture} `
      + `metallic=${pbr.metallicFactor ?? 1} roughness=${pbr.roughnessFactor ?? 1} unlit=${materialUnlit}`,
    );
    if (hasTexture && (pbr.metallicFactor ?? 1) > 0.5) {
      problems.push(`material[${index}] is metallic=${pbr.metallicFactor} with a baseColorTexture and no environment map — renders black`);
    }
    if (!materialUnlit && !unlit) {
      notes.push(`material[${index}] is lit; a viewer with no lights and no IBL will render it black`);
    }
  }

  // --- primitives
  for (const [meshIndex, mesh] of (json.meshes || []).entries()) {
    for (const [primIndex, prim] of (mesh.primitives || []).entries()) {
      const label = `mesh[${meshIndex}].prim[${primIndex}]`;
      const attrs = prim.attributes || {};
      const material = json.materials?.[prim.material];
      const hasTexture = material?.pbrMetallicRoughness?.baseColorTexture != null;

      if (attrs.COLOR_0 != null) {
        const accessor = json.accessors[attrs.COLOR_0];
        const view = json.bufferViews[accessor.bufferView];
        notes.push(`${label} COLOR_0 ${accessor.type}/${accessor.componentType} normalized=${accessor.normalized}`);
        const bytesPerElement = accessor.componentType === 5121 ? (accessor.type === 'VEC3' ? 3 : 4) : null;
        if (bytesPerElement === 3) {
          problems.push(`${label} COLOR_0 is a 3-byte UNSIGNED_BYTE VEC3 — violates the glTF 4-byte alignment rule; strict loaders reject it and lenient ones smear colour across vertices`);
        }
        if ((view.byteOffset || 0) % 4 !== 0) {
          problems.push(`${label} COLOR_0 bufferView starts at byte ${view.byteOffset}, not a multiple of 4`);
        }
        if (hasTexture) {
          problems.push(`${label} has BOTH COLOR_0 and baseColorTexture — per spec they multiply, so the albedo is squared and the surface renders dark and desaturated`);
        }
      }

      if (attrs.TEXCOORD_0 != null) {
        const uv = readUvs(json, bin, attrs.TEXCOORD_0);
        if (uv) {
          let outside = 0;
          let vLow = 0;
          let vHigh = 0;
          for (let i = 0; i < uv.vs.length; i += 1) {
            if (uv.us[i] < -0.001 || uv.us[i] > 1.001 || uv.vs[i] < -0.001 || uv.vs[i] > 1.001) outside += 1;
            if (uv.vs[i] < 0.5) vLow += 1; else vHigh += 1;
          }
          const sampled = uv.vs.length;
          notes.push(
            `${label} TEXCOORD_0 ${uv.count.toLocaleString()} UVs · `
            + `V below 0.5: ${(100 * vLow / sampled).toFixed(1)}%, above: ${(100 * vHigh / sampled).toFixed(1)}% · `
            + `outside [0,1]: ${(100 * outside / sampled).toFixed(2)}%`,
          );
          if (outside / sampled > 0.02) {
            problems.push(`${label} has ${(100 * outside / sampled).toFixed(1)}% of UVs outside [0,1] — with CLAMP_TO_EDGE those faces smear the atlas border`);
          }
        }
      } else if (hasTexture) {
        problems.push(`${label} has a baseColorTexture but no TEXCOORD_0 — nothing can sample it`);
      }

      for (const [name, index] of Object.entries(attrs)) {
        if (!json.accessors?.[index]) problems.push(`${label} attribute ${name} points at missing accessor ${index}`);
      }
    }
  }

  for (const [index, texture] of (json.textures || []).entries()) {
    if (!json.images?.[texture.source]) problems.push(`texture[${index}] points at missing image ${texture.source}`);
  }
  const declared = json.buffers?.[0]?.byteLength;
  if (declared != null && bin && declared !== bin.length) {
    problems.push(`buffer declares ${declared} bytes but the BIN chunk is ${bin.length}`);
  }

  return { problems, notes, unlit };
}

const files = process.argv.slice(2);
if (!files.length) {
  console.error('Usage: node tools/validate-glb.mjs <file.glb> [...]');
  process.exit(2);
}

let failed = 0;
for (const file of files) {
  console.log(`\n=== ${file} ===`);
  if (!existsSync(file)) { console.log('  missing'); failed += 1; continue; }
  try {
    const { problems, notes } = report(file);
    for (const note of notes) console.log(`  · ${note}`);
    if (problems.length) {
      failed += 1;
      console.log('\n  PROBLEMS:');
      for (const problem of problems) console.log(`  ✗ ${problem}`);
    } else {
      console.log('\n  ✓ no known black-render defects');
    }
  } catch (error) {
    failed += 1;
    console.log(`  ✗ could not parse: ${error.message}`);
  }
}
process.exit(failed ? 1 : 0);
