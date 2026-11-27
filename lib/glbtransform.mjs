/**
 * Re-orient and metric-scale a GLB without touching its geometry buffers.
 * ---------------------------------------------------------------------------
 * OpenMVS writes its textured mesh in COLMAP's world frame: an arbitrary
 * rotation (the solver has no idea which way is down) at an arbitrary scale
 * (no GPS, so "one unit" is meaningless). The digital-surface-model exporter in
 * `export3d.mjs` already works out two things from the solved cameras:
 *
 *   • a rotation that puts gravity along +Y (so the model stands upright), and
 *   • a scale that stretches the arbitrary units to the operator's stated
 *     flight altitude in metres.
 *
 * Rather than rewrite every vertex of a multi-million-triangle mesh in JS, we
 * push those corrections into a single parent node's transform matrix. glTF
 * applies node transforms hierarchically, so one 4×4 matrix re-orients and
 * re-scales the whole scene for free, and the heavy BIN chunk is copied through
 * byte-for-byte. The only edit is to the (tiny) JSON chunk.
 *
 * Why a parent node instead of editing existing node matrices? OpenMVS may emit
 * one or several root nodes with their own transforms; re-parenting them under a
 * fresh node composes cleanly with whatever it already wrote.
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK_TYPE = 0x4e4f534a;
const BIN_CHUNK_TYPE = 0x004e4942;

/** Pad a buffer up to a 4-byte boundary with `fill` (0x20 space for JSON, 0 for BIN). */
function pad4(buffer, fill) {
  const extra = (4 - (buffer.length % 4)) % 4;
  return extra ? Buffer.concat([buffer, Buffer.alloc(extra, fill)]) : buffer;
}

/** Split a GLB container into its JSON object and raw BIN chunk. */
function parseGlb(buffer) {
  if (buffer.length < 12 || buffer.readUInt32LE(0) !== GLB_MAGIC) {
    throw new Error('Not a GLB file (bad magic).');
  }
  let offset = 12;
  let json = null;
  let bin = null;
  while (offset + 8 <= buffer.length) {
    const length = buffer.readUInt32LE(offset);
    const type = buffer.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = start + length;
    if (end > buffer.length) break;
    const data = buffer.subarray(start, end);
    if (type === JSON_CHUNK_TYPE) json = JSON.parse(data.toString('utf8'));
    else if (type === BIN_CHUNK_TYPE) bin = data;
    offset = end;
  }
  if (!json) throw new Error('GLB has no JSON chunk.');
  return { json, bin };
}

/** Re-assemble a GLB container from a JSON object and the untouched BIN chunk. */
function serializeGlb(json, bin) {
  const jsonChunk = pad4(Buffer.from(JSON.stringify(json)), 0x20);
  const chunks = [];
  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(jsonChunk.length, 0);
  jsonHeader.writeUInt32LE(JSON_CHUNK_TYPE, 4);
  chunks.push(jsonHeader, jsonChunk);
  if (bin && bin.length) {
    const binChunk = pad4(Buffer.from(bin), 0);
    const binHeader = Buffer.alloc(8);
    binHeader.writeUInt32LE(binChunk.length, 0);
    binHeader.writeUInt32LE(BIN_CHUNK_TYPE, 4);
    chunks.push(binHeader, binChunk);
  }
  const body = Buffer.concat(chunks);
  const header = Buffer.alloc(12);
  header.writeUInt32LE(GLB_MAGIC, 0);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + body.length, 8);
  return Buffer.concat([header, body]);
}

/**
 * Build a column-major glTF node matrix for `p -> s·R·(p − origin)`.
 *
 * @param {number[][]} rotation 3×3 row-major rotation (R·v), COLMAP -> Y-up
 * @param {number}     scale    uniform metric scale
 * @param {number[]}   origin   COLMAP-space point to move to the glTF origin
 */
function affineColumnMajor(rotation, scale, origin = [0, 0, 0]) {
  const R = rotation;
  const s = scale;
  // Translation t = −s·R·origin, so the chosen origin lands at (0,0,0).
  const t = [
    -s * (R[0][0] * origin[0] + R[0][1] * origin[1] + R[0][2] * origin[2]),
    -s * (R[1][0] * origin[0] + R[1][1] * origin[1] + R[1][2] * origin[2]),
    -s * (R[2][0] * origin[0] + R[2][1] * origin[1] + R[2][2] * origin[2]),
  ];
  // glTF matrices are column-major: [col0, col1, col2, col3].
  return [
    s * R[0][0], s * R[1][0], s * R[2][0], 0,
    s * R[0][1], s * R[1][1], s * R[2][1], 0,
    s * R[0][2], s * R[1][2], s * R[2][2], 0,
    t[0], t[1], t[2], 1,
  ];
}

/**
 * Returns a new GLB buffer whose scene is rotated to Y-up and scaled to metres.
 *
 * @param {Buffer} glbBuffer       the OpenMVS-written GLB
 * @param {object} transform
 * @param {number[][]} transform.rotation 3×3 row-major gravity rotation
 * @param {number}     transform.scale    metric scale (defaults to 1)
 * @param {number[]}   [transform.origin] COLMAP-space recentre point
 */
export function reorientGlb(glbBuffer, { rotation, scale = 1, origin = [0, 0, 0] }) {
  const { json, bin } = parseGlb(glbBuffer);
  const sceneIndex = Number.isInteger(json.scene) ? json.scene : 0;
  json.scenes = json.scenes || [{ nodes: [] }];
  json.nodes = json.nodes || [];
  const scene = json.scenes[sceneIndex] || json.scenes[0];
  const roots = Array.isArray(scene.nodes) ? scene.nodes : [];

  const parent = {
    name: 'SkyforgeAlignment',
    matrix: affineColumnMajor(rotation, scale, origin),
    children: roots,
  };
  json.nodes.push(parent);
  scene.nodes = [json.nodes.length - 1];

  if (!json.asset) json.asset = { version: '2.0' };
  json.asset.generator = `${json.asset.generator || 'OpenMVS'} + Skyforge alignment`;
  return serializeGlb(json, bin);
}

/**
 * Fold every extra glTF buffer into the GLB's single BIN chunk.
 *
 * A GLB container may carry only one binary chunk, so a writer that needs a
 * second buffer has to inline it as a `data:` URI. OpenMVS does exactly that
 * when a mesh is split across several texture atlases: primitive 0 reads its
 * geometry from the BIN chunk and primitive 1 reads `buffers[1]`. Collapsing
 * the buffer array to a single entry without moving those bytes leaves every
 * bufferView that named buffer 1 pointing at a buffer that no longer exists,
 * and the whole model fails to load with "Cannot read properties of undefined
 * (reading 'type')" — the mesh is on disk and looks the right size, but no
 * viewer will open it. This copies each extra buffer onto the end of the BIN
 * chunk and re-bases the bufferViews that referenced it.
 *
 * @returns {{parts: Buffer[], length: number, merged: number}}
 */
function consolidateBuffers(json, bin, baseDir) {
  const buffers = Array.isArray(json.buffers) ? json.buffers : [];
  const parts = [bin && bin.length ? Buffer.from(bin) : Buffer.alloc(0)];
  let length = parts[0].length;
  const rebase = new Map();

  for (let index = 1; index < buffers.length; index += 1) {
    const uri = buffers[index]?.uri;
    if (!uri) throw new Error(`glTF buffer ${index} has no uri but is not the GLB chunk.`);
    let data;
    if (uri.startsWith('data:')) {
      const comma = uri.indexOf(',');
      if (comma < 0) throw new Error(`glTF buffer ${index} has a malformed data uri.`);
      data = Buffer.from(uri.slice(comma + 1), 'base64');
    } else {
      const file = join(baseDir || '', decodeURIComponent(uri));
      if (!existsSync(file)) throw new Error(`glTF buffer ${index} points at a missing file: ${uri}`);
      data = readFileSync(file);
    }
    const pad = (4 - (length % 4)) % 4; // bufferViews must stay 4-byte aligned
    if (pad) { parts.push(Buffer.alloc(pad)); length += pad; }
    rebase.set(index, length);
    parts.push(data);
    length += data.length;
  }

  for (const view of json.bufferViews || []) {
    const base = rebase.get(view.buffer);
    if (base === undefined) continue;
    view.byteOffset = base + (view.byteOffset || 0);
    view.buffer = 0;
  }
  return { parts, length, merged: rebase.size };
}

/**
 * Inline any externally-referenced texture images into the GLB's BIN chunk.
 *
 * OpenMVS "GLB" export is really glTF-with-external-textures: the mesh is a .glb
 * container but its material points at a sibling `model_0.png` via `image.uri`.
 * Served over HTTP or handed to someone as a single file, that texture is lost
 * and the mesh renders flat grey. This reads each referenced PNG/JPEG, appends
 * it to the binary buffer, and rewrites the image to a self-contained
 * `bufferView` + `mimeType` — turning the file into a true, portable GLB.
 *
 * @param {Buffer} glbBuffer       the OpenMVS GLB
 * @param {string} [baseDir]       folder the `uri`s are relative to (defaults to
 *                                 the GLB's own directory when a path is known)
 * @returns {{ buffer: Buffer, embedded: number }}
 */
export function embedExternalImages(glbBuffer, baseDir) {
  const { json, bin } = parseGlb(glbBuffer);
  json.bufferViews = json.bufferViews || [];
  // Must run before anything rewrites json.buffers, and regardless of whether
  // there are images to embed: a multi-buffer GLB is unloadable on its own.
  const { parts, length: merged0, merged } = consolidateBuffers(json, bin, baseDir);
  let length = merged0;
  let embedded = 0;
  const mimeFor = (uri) => (/\.jpe?g$/i.test(uri) ? 'image/jpeg' : 'image/png');

  for (const image of json.images) {
    if (!image.uri || image.uri.startsWith('data:')) continue;
    const file = join(baseDir || '', image.uri);
    if (!existsSync(file)) continue;
    const data = readFileSync(file);
    const pad = (4 - (length % 4)) % 4; // bufferViews must be 4-byte aligned
    if (pad) { parts.push(Buffer.alloc(pad)); length += pad; }
    const byteOffset = length;
    parts.push(data);
    length += data.length;
    json.bufferViews.push({ buffer: 0, byteOffset, byteLength: data.length });
    image.bufferView = json.bufferViews.length - 1;
    image.mimeType = mimeFor(image.uri);
    delete image.uri;
    delete image.name;
    embedded += 1;
  }
  if (!embedded && !merged) return { buffer: glbBuffer, embedded: 0 };

  const newBin = Buffer.concat(parts);
  json.buffers = [{ byteLength: newBin.length }];
  return { buffer: serializeGlb(json, newBin), embedded, merged };
}

/**
 * Flip the V (vertical) texture coordinate of every TEXCOORD_0 in the GLB.
 *
 * OpenMVS bakes its atlas UVs with an OpenGL bottom-left origin (V increases
 * upward). glTF — and therefore Three.js and every compliant viewer — defines
 * the origin at the top-left (V increases downward). Loaded as-is, every face
 * samples the black padding *between* atlas patches instead of the photo inside
 * them, and the whole model renders black. Rewriting v -> 1 - v in place makes
 * the UVs spec-compliant, so the texture lands where it belongs in any viewer.
 *
 * Only the float UV values inside the BIN chunk change; geometry, the texture
 * image and the node graph are untouched.
 */
export function flipTextureV(glbBuffer) {
  const { json, bin } = parseGlb(glbBuffer);
  if (!bin || !json.meshes) return glbBuffer;
  const buffer = Buffer.from(bin); // a writable copy of the BIN chunk
  const done = new Set();
  for (const mesh of json.meshes) {
    for (const prim of mesh.primitives || []) {
      const uvIndex = prim.attributes?.TEXCOORD_0;
      if (uvIndex == null || done.has(uvIndex)) continue;
      done.add(uvIndex);
      const accessor = json.accessors[uvIndex];
      if (accessor.componentType !== 5126) continue; // only handle float UVs
      const view = json.bufferViews[accessor.bufferView];
      const base = (view.byteOffset || 0) + (accessor.byteOffset || 0);
      const stride = view.byteStride || 8; // VEC2 of 4-byte floats
      for (let i = 0; i < accessor.count; i += 1) {
        const vOffset = base + i * stride + 4; // second float of the pair
        buffer.writeFloatLE(1 - buffer.readFloatLE(vOffset), vOffset);
      }
    }
  }
  return serializeGlb(json, buffer);
}

/** Exposed for tests: round-trip a GLB through parse + serialize unchanged. */
export const _internals = { parseGlb, serializeGlb, affineColumnMajor, consolidateBuffers, dirname };
