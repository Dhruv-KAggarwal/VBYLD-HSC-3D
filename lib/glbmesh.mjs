/**
 * GLB geometry read/write.
 * ---------------------------------------------------------------------------
 * `glbtransform.mjs` edits a GLB's JSON chunk without ever looking at the
 * geometry, which is exactly right for re-orienting a mesh. The three new
 * enhancement stages need the opposite: mesh repair rewrites topology, LOD
 * generation collapses edges, and confidence attribution attaches a per-vertex
 * attribute. All of them need the vertex and index buffers as typed arrays, and
 * all of them need to write a valid self-contained GLB back out.
 *
 * This module is that layer, and nothing more. It deliberately does not depend
 * on three.js (the pipeline is Node-side) and introduces no new dependency: a
 * GLB is a 12-byte header, a JSON chunk and a BIN chunk, and glTF accessors are
 * a small, well-specified indirection over that BIN chunk.
 *
 * What it handles, because real OpenMVS output contains all of it:
 *   • several primitives, each with its own material and texture image
 *     (a mesh that overflowed one atlas);
 *   • interleaved vertex data (`bufferView.byteStride`);
 *   • unsigned byte / short / int indices;
 *   • an alignment parent node carrying the gravity rotation and metric scale,
 *     which must survive a rewrite byte-for-byte or the model moves.
 *
 * What it does not handle, and says so rather than guessing: sparse accessors,
 * draco/meshopt-compressed primitives, morph targets and skins. None appear in
 * anything this pipeline produces; `readMeshGlb` throws if it meets one.
 */

const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK_TYPE = 0x4e4f534a;
const BIN_CHUNK_TYPE = 0x004e4942;

const COMPONENT = {
  5120: { array: Int8Array, size: 1 },
  5121: { array: Uint8Array, size: 1 },
  5122: { array: Int16Array, size: 2 },
  5123: { array: Uint16Array, size: 2 },
  5125: { array: Uint32Array, size: 4 },
  5126: { array: Float32Array, size: 4 },
};

const TYPE_COMPONENTS = {
  SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16,
};

function pad4(buffer, fill) {
  const extra = (4 - (buffer.length % 4)) % 4;
  return extra ? Buffer.concat([buffer, Buffer.alloc(extra, fill)]) : buffer;
}

export function parseGlbContainer(buffer) {
  const bytes = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer);
  if (bytes.length < 12 || bytes.readUInt32LE(0) !== GLB_MAGIC) throw new Error('Not a GLB file (bad magic).');
  let offset = 12;
  let json = null;
  let bin = null;
  while (offset + 8 <= bytes.length) {
    const length = bytes.readUInt32LE(offset);
    const type = bytes.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = start + length;
    if (end > bytes.length) break;
    const data = bytes.subarray(start, end);
    if (type === JSON_CHUNK_TYPE) json = JSON.parse(data.toString('utf8'));
    else if (type === BIN_CHUNK_TYPE) bin = data;
    offset = end;
  }
  if (!json) throw new Error('GLB has no JSON chunk.');
  return { json, bin: bin || Buffer.alloc(0) };
}

export function serializeGlbContainer(json, bin) {
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
 * Copy one accessor out of the BIN chunk into a tightly packed typed array.
 *
 * Interleaved data has to be de-interleaved here rather than later: every
 * consumer in this project wants `positions[i * 3 + k]`, and a byteStride that
 * does not match the element size silently reads a neighbouring attribute.
 */
function readAccessor(json, bin, index) {
  if (index == null) return null;
  const accessor = json.accessors?.[index];
  if (!accessor) return null;
  if (accessor.sparse) throw new Error('Sparse glTF accessors are not supported.');
  const component = COMPONENT[accessor.componentType];
  if (!component) throw new Error(`Unknown glTF componentType ${accessor.componentType}.`);
  const stride = TYPE_COMPONENTS[accessor.type];
  if (!stride) throw new Error(`Unknown glTF accessor type ${accessor.type}.`);
  const out = new component.array(accessor.count * stride);
  if (accessor.bufferView == null) return out; // spec-legal: all zeroes
  const view = json.bufferViews[accessor.bufferView];
  const base = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const elementBytes = component.size * stride;
  const byteStride = view.byteStride || elementBytes;
  const read = {
    5120: (o) => bin.readInt8(o),
    5121: (o) => bin.readUInt8(o),
    5122: (o) => bin.readInt16LE(o),
    5123: (o) => bin.readUInt16LE(o),
    5125: (o) => bin.readUInt32LE(o),
    5126: (o) => bin.readFloatLE(o),
  }[accessor.componentType];
  for (let i = 0; i < accessor.count; i += 1) {
    const element = base + i * byteStride;
    for (let k = 0; k < stride; k += 1) out[i * stride + k] = read(element + k * component.size);
  }
  return out;
}

/** Decode a normalized integer attribute to floats in [0,1] (or [-1,1]). */
export function dequantize(values, componentType) {
  if (componentType === 5126) return values instanceof Float32Array ? values : Float32Array.from(values);
  const divisor = { 5121: 255, 5123: 65535, 5120: 127, 5122: 32767 }[componentType];
  if (!divisor) return Float32Array.from(values);
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) out[i] = Math.max(-1, values[i] / divisor);
  return out;
}

function imageBytes(json, bin, imageIndex) {
  const image = json.images?.[imageIndex];
  if (!image) return null;
  if (image.bufferView != null) {
    const view = json.bufferViews[image.bufferView];
    const start = view.byteOffset || 0;
    return {
      bytes: Buffer.from(bin.subarray(start, start + view.byteLength)),
      mime: image.mimeType || 'image/png',
    };
  }
  if (typeof image.uri === 'string' && image.uri.startsWith('data:')) {
    const comma = image.uri.indexOf(',');
    const mime = /^data:([^;,]+)/.exec(image.uri)?.[1] || 'image/png';
    return { bytes: Buffer.from(image.uri.slice(comma + 1), 'base64'), mime };
  }
  // An external uri means the GLB is not self-contained. embedExternalImages()
  // in glbtransform.mjs runs before anything reaches here, so this is a bug
  // rather than a case to paper over.
  return null;
}

/**
 * Read a GLB into plain typed-array primitives.
 *
 * @param {Buffer} buffer
 * @returns {{
 *   primitives: Array<{positions: Float32Array, indices: Uint32Array,
 *     uvs: Float32Array|null, normals: Float32Array|null, colors: Uint8Array|null,
 *     imageIndex: number|null, materialIndex: number|null, material: object|null}>,
 *   images: Array<{bytes: Buffer, mime: string}>,
 *   rootMatrix: number[]|null,
 *   unlit: boolean,
 *   json: object,
 * }}
 */
export function readMeshGlb(buffer) {
  const { json, bin } = parseGlbContainer(buffer);
  const primitives = [];
  const images = [];
  const imageMap = new Map();

  const takeImage = (materialIndex) => {
    const material = json.materials?.[materialIndex];
    const textureIndex = material?.pbrMetallicRoughness?.baseColorTexture?.index;
    if (textureIndex == null) return null;
    const texture = json.textures?.[textureIndex];
    const source = texture?.source ?? texture?.extensions?.EXT_texture_webp?.source;
    if (source == null) return null;
    if (imageMap.has(source)) return imageMap.get(source);
    const decoded = imageBytes(json, bin, source);
    if (!decoded) return null;
    images.push(decoded);
    const slot = images.length - 1;
    imageMap.set(source, slot);
    return slot;
  };

  const sceneIndex = Number.isInteger(json.scene) ? json.scene : 0;
  const sceneRoots = json.scenes?.[sceneIndex]?.nodes || [];

  // The alignment matrix. reorientGlb() wraps the whole scene in one node named
  // SkyforgeAlignment; anything that rewrites geometry must put it back or the
  // model loses its gravity rotation and its metric scale. It is stripped off
  // here (so vertex positions stay in the reconstruction's own frame, which is
  // the frame the solved cameras live in) and re-applied by writeMeshGlb().
  let rootMatrix = null;
  let roots = sceneRoots;
  if (sceneRoots.length === 1) {
    const node = json.nodes?.[sceneRoots[0]];
    if (node?.name === 'SkyforgeAlignment' && node.matrix?.length === 16 && node.mesh == null) {
      rootMatrix = [...node.matrix];
      roots = node.children || [];
    }
  }

  const takeMesh = (mesh, matrix) => {
    for (const primitive of mesh.primitives || []) {
      if (primitive.extensions?.KHR_draco_mesh_compression || primitive.extensions?.EXT_meshopt_compression) {
        throw new Error('Compressed glTF primitives cannot be edited in place; use the uncompressed model.');
      }
      // mode 4 is TRIANGLES. Point primitives (the optional photo-point cloud
      // the DSM exporter adds) are reported but never repaired or simplified.
      const mode = primitive.mode ?? 4;
      const attributes = primitive.attributes || {};
      let positions = readAccessor(json, bin, attributes.POSITION);
      if (!positions) continue;
      positions = positions instanceof Float32Array ? positions : Float32Array.from(positions);
      const rawIndices = readAccessor(json, bin, primitive.indices);
      const indices = rawIndices
        ? (rawIndices instanceof Uint32Array ? rawIndices : Uint32Array.from(rawIndices))
        : Uint32Array.from({ length: positions.length / 3 }, (_, i) => i);
      let normals = attributes.NORMAL != null ? Float32Array.from(readAccessor(json, bin, attributes.NORMAL)) : null;
      if (matrix && !isIdentity(matrix)) {
        applyMatrix(positions, matrix, 1);
        // Normals are recomputed by consumers after any edit, so a transformed
        // node simply drops the stored ones rather than carrying a stale set.
        normals = null;
      }
      const colorAccessor = json.accessors?.[attributes.COLOR_0];
      const uvAccessor = json.accessors?.[attributes.TEXCOORD_0];
      primitives.push({
        mode,
        positions,
        indices,
        uvs: uvAccessor ? dequantize(readAccessor(json, bin, attributes.TEXCOORD_0), uvAccessor.componentType) : null,
        normals,
        colors: attributes.COLOR_0 != null
          ? { values: readAccessor(json, bin, attributes.COLOR_0), componentType: colorAccessor.componentType, stride: TYPE_COMPONENTS[colorAccessor.type] }
          : null,
        materialIndex: primitive.material ?? null,
        material: primitive.material != null ? json.materials?.[primitive.material] || null : null,
        imageIndex: takeImage(primitive.material),
        extras: primitive.extras || null,
      });
    }
  };

  // Depth-first in node order, which is the order a viewer's scene traversal
  // visits them. Per-face sidecar files (confidence, provenance) are indexed in
  // exactly this order.
  const visit = (nodeIndex, parent) => {
    const node = json.nodes?.[nodeIndex];
    if (!node) return;
    const matrix = multiplyMatrices(parent, nodeMatrix(node));
    if (node.mesh != null && json.meshes?.[node.mesh]) takeMesh(json.meshes[node.mesh], matrix);
    for (const child of node.children || []) visit(child, matrix);
  };
  if (roots.length) for (const root of roots) visit(root, IDENTITY);
  else for (const mesh of json.meshes || []) takeMesh(mesh, IDENTITY);

  return {
    primitives,
    images,
    rootMatrix,
    unlit: Array.isArray(json.extensionsUsed) && json.extensionsUsed.includes('KHR_materials_unlit'),
    json,
  };
}

const IDENTITY = Object.freeze([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function isIdentity(m) {
  return m.every((value, i) => Math.abs(value - IDENTITY[i]) < 1e-12);
}

/** Column-major 4x4 product a * b. */
export function multiplyMatrices(a, b) {
  const out = new Array(16).fill(0);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      for (let k = 0; k < 4; k += 1) out[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
    }
  }
  return out;
}

/** A node's local transform, from either `matrix` or translation/rotation/scale. */
function nodeMatrix(node) {
  if (node.matrix?.length === 16) return node.matrix;
  if (!node.translation && !node.rotation && !node.scale) return IDENTITY;
  const [tx, ty, tz] = node.translation || [0, 0, 0];
  const [x, y, z, w] = node.rotation || [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale || [1, 1, 1];
  return [
    (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
    2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
    2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

/** Transform packed xyz triples in place. `w` is 1 for points, 0 for vectors. */
export function applyMatrix(values, m, w = 1) {
  for (let i = 0; i < values.length; i += 3) {
    const x = values[i];
    const y = values[i + 1];
    const z = values[i + 2];
    values[i] = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
    values[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
    values[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
  }
  return values;
}

/** Uniform scale carried by a column-major matrix (length of its first column). */
export function matrixScale(m) {
  return m ? Math.hypot(m[0], m[1], m[2]) || 1 : 1;
}

/** Per-component min/max for an accessor, which POSITION is required to carry. */
function minMax(values, stride) {
  const min = Array(stride).fill(Infinity);
  const max = Array(stride).fill(-Infinity);
  for (let i = 0; i < values.length; i += stride) {
    for (let k = 0; k < stride; k += 1) {
      const v = values[i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  if (!values.length) return { min: Array(stride).fill(0), max: Array(stride).fill(0) };
  return { min, max };
}

function sniffMime(bytes) {
  if (!bytes || bytes.length < 4) return 'image/png';
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return 'image/png';
  if (bytes[0] === 0xff && bytes[1] === 0xd8) return 'image/jpeg';
  if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[8] === 0x57) return 'image/webp';
  return 'image/png';
}

/**
 * Write primitives back out as a single self-contained GLB.
 *
 * @param {object} args
 * @param {Array} args.primitives     as returned by readMeshGlb, possibly edited
 * @param {Array} args.images         texture images, referenced by imageIndex
 * @param {number[]|null} args.rootMatrix  alignment matrix to re-apply
 * @param {boolean} args.unlit        emit KHR_materials_unlit (baked photography)
 * @param {string} args.generator
 * @param {object} [args.extras]      written to asset.extras, for provenance
 */
export function writeMeshGlb({
  primitives, images = [], rootMatrix = null, unlit = true, generator = 'Skyforge', extras = null, quantize = false,
}) {
  const chunks = [];
  let length = 0;
  const views = [];
  const accessors = [];

  const pushView = (data, target = 0) => {
    const pad = (4 - (length % 4)) % 4;
    if (pad) { chunks.push(Buffer.alloc(pad)); length += pad; }
    const view = { buffer: 0, byteOffset: length, byteLength: data.byteLength };
    if (target) view.target = target;
    views.push(view);
    chunks.push(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
    length += data.byteLength;
    return views.length - 1;
  };
  const pushAccessor = (data, componentType, type, target, { normalized = false, bounds = false } = {}) => {
    const stride = TYPE_COMPONENTS[type];
    const accessor = {
      bufferView: pushView(data, target),
      componentType,
      count: data.length / stride,
      type,
    };
    if (normalized) accessor.normalized = true;
    if (bounds) Object.assign(accessor, minMax(data, stride));
    accessors.push(accessor);
    return accessors.length - 1;
  };

  // Quantised positions share one grid over the whole model, undone by the mesh
  // node's scale and translation.
  let grid = null;
  if (quantize) {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const primitive of primitives) {
      if (!primitive.positions) continue;
      for (let i = 0; i < primitive.positions.length; i += 1) {
        const k = i % 3;
        if (primitive.positions[i] < lo[k]) lo[k] = primitive.positions[i];
        if (primitive.positions[i] > hi[k]) hi[k] = primitive.positions[i];
      }
    }
    grid = { origin: lo, step: Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-9) / 65535 };
  }
  // Vertex attributes must start on 4-byte boundaries, hence the padded strides.
  const pushPadded = (data, componentType, type, count, strideElements, normalized, bounds) => {
    const view = pushView(data, 34962);
    views[view].byteStride = strideElements * data.BYTES_PER_ELEMENT;
    const accessor = { bufferView: view, componentType, count, type };
    if (normalized) accessor.normalized = true;
    if (bounds) Object.assign(accessor, bounds);
    accessors.push(accessor);
    return accessors.length - 1;
  };
  const quantizedPositions = (positions) => {
    const count = positions.length / 3;
    const out = new Uint16Array(count * 4);
    const min = [65535, 65535, 65535];
    const max = [0, 0, 0];
    for (let i = 0; i < count; i += 1) {
      for (let k = 0; k < 3; k += 1) {
        const q = Math.round((positions[i * 3 + k] - grid.origin[k]) / grid.step);
        out[i * 4 + k] = q;
        if (q < min[k]) min[k] = q;
        if (q > max[k]) max[k] = q;
      }
    }
    return pushPadded(out, 5123, 'VEC3', count, 4, false, { min, max });
  };
  const quantizedNormals = (normals) => {
    const count = normals.length / 3;
    const out = new Int8Array(count * 4);
    for (let i = 0; i < count; i += 1) {
      for (let k = 0; k < 3; k += 1) out[i * 4 + k] = Math.max(-127, Math.min(127, Math.round(normals[i * 3 + k] * 127)));
    }
    return pushPadded(out, 5120, 'VEC3', count, 4, true, null);
  };

  const materials = [];
  const textures = [];
  const jsonImages = [];
  const imageSlot = new Map();
  const glPrimitives = [];
  let faceOffset = 0;
  let usesWebp = false;

  for (const primitive of primitives) {
    if (!primitive.positions?.length || !primitive.indices?.length) continue;
    const attributes = {
      POSITION: grid
        ? quantizedPositions(primitive.positions)
        : pushAccessor(primitive.positions, 5126, 'VEC3', 34962, { bounds: true }),
    };
    if (primitive.normals?.length) {
      attributes.NORMAL = grid ? quantizedNormals(primitive.normals) : pushAccessor(primitive.normals, 5126, 'VEC3', 34962);
    }
    if (primitive.uvs?.length) {
      // 16-bit coordinates only cover [0,1]; anything wrapping stays float.
      const unit = grid && primitive.uvs.every((v) => v >= 0 && v <= 1);
      attributes.TEXCOORD_0 = unit
        ? pushAccessor(Uint16Array.from(primitive.uvs, (v) => Math.round(v * 65535)), 5123, 'VEC2', 34962, { normalized: true })
        : pushAccessor(primitive.uvs, 5126, 'VEC2', 34962);
    }
    if (primitive.colors?.values?.length) {
      const { values, componentType, stride } = primitive.colors;
      attributes.COLOR_0 = pushAccessor(values, componentType, stride === 4 ? 'VEC4' : 'VEC3', 34962, {
        normalized: componentType !== 5126,
      });
    }
    // Index width follows the vertex count: a simplified LOD often fits in
    // 16 bits, which halves the index buffer for free.
    const vertexCount = primitive.positions.length / 3;
    const indices = vertexCount <= 65535
      ? Uint16Array.from(primitive.indices)
      : (primitive.indices instanceof Uint32Array ? primitive.indices : Uint32Array.from(primitive.indices));
    const indexAccessor = pushAccessor(indices, vertexCount <= 65535 ? 5123 : 5125, 'SCALAR', 34963);

    const pbr = { metallicFactor: 0, roughnessFactor: 1, baseColorFactor: [1, 1, 1, 1] };
    if (primitive.imageIndex != null && images[primitive.imageIndex]) {
      if (!imageSlot.has(primitive.imageIndex)) {
        const image = images[primitive.imageIndex];
        const bytes = Buffer.isBuffer(image.bytes) ? image.bytes : Buffer.from(image.bytes);
        const pad = (4 - (length % 4)) % 4;
        if (pad) { chunks.push(Buffer.alloc(pad)); length += pad; }
        views.push({ buffer: 0, byteOffset: length, byteLength: bytes.length });
        chunks.push(bytes);
        length += bytes.length;
        const mimeType = image.mime || sniffMime(bytes);
        jsonImages.push({ bufferView: views.length - 1, mimeType });
        if (mimeType === 'image/webp') {
          usesWebp = true;
          textures.push({ sampler: 0, extensions: { EXT_texture_webp: { source: jsonImages.length - 1 } } });
        } else {
          textures.push({ sampler: 0, source: jsonImages.length - 1 });
        }
        imageSlot.set(primitive.imageIndex, textures.length - 1);
      }
      pbr.baseColorTexture = { index: imageSlot.get(primitive.imageIndex), texCoord: 0 };
    }
    const material = {
      name: primitive.material?.name || `SkyforgeSurface${materials.length}`,
      pbrMetallicRoughness: pbr,
      doubleSided: true,
    };
    if (unlit) material.extensions = { KHR_materials_unlit: {} };
    materials.push(material);
    const entry = {
      attributes,
      indices: indexAccessor,
      material: materials.length - 1,
      mode: primitive.mode ?? 4,
      // The first face of this primitive in the concatenated face order. A
      // viewer reads it from geometry.userData to index per-face sidecar data.
      extras: { ...(primitive.extras || {}), skyforgeFaceOffset: faceOffset },
    };
    faceOffset += (primitive.mode ?? 4) === 4 ? primitive.indices.length / 3 : 0;
    glPrimitives.push(entry);
  }

  if (!glPrimitives.length) throw new Error('No primitives to write.');

  const nodes = [{ name: 'SkyforgeMesh', mesh: 0 }];
  if (grid) {
    nodes[0].translation = grid.origin;
    nodes[0].scale = [grid.step, grid.step, grid.step];
  }
  let sceneRoots = [0];
  if (rootMatrix) {
    nodes.push({ name: 'SkyforgeAlignment', matrix: rootMatrix, children: [0] });
    sceneRoots = [1];
  }

  const json = {
    asset: { version: '2.0', generator },
    scene: 0,
    scenes: [{ nodes: sceneRoots }],
    nodes,
    meshes: [{ name: 'SkyforgeMesh', primitives: glPrimitives }],
    materials,
    accessors,
    bufferViews: views,
    buffers: [{ byteLength: length }],
  };
  if (extras) json.asset.extras = extras;
  const used = [];
  if (unlit) used.push('KHR_materials_unlit');
  if (grid) {
    used.push('KHR_mesh_quantization');
    json.extensionsRequired = ['KHR_mesh_quantization'];
  }
  if (usesWebp) {
    used.push('EXT_texture_webp');
    // No PNG fallback is embedded, so a loader without WebP cannot show it.
    json.extensionsRequired = [...(json.extensionsRequired || []), 'EXT_texture_webp'];
  }
  if (used.length) json.extensionsUsed = used;
  if (jsonImages.length) {
    json.images = jsonImages;
    json.textures = textures;
    json.samplers = [{ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 }];
  }
  return serializeGlbContainer(json, Buffer.concat(chunks));
}

/** Triangle and vertex totals across every triangle primitive. */
export function meshCounts(primitives) {
  let triangles = 0;
  let vertices = 0;
  for (const primitive of primitives) {
    if ((primitive.mode ?? 4) !== 4) continue;
    triangles += primitive.indices.length / 3;
    vertices += primitive.positions.length / 3;
  }
  return { triangles, vertices };
}

/** Recompute per-vertex normals by area-weighted face averaging. */
export function recomputeNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  for (let f = 0; f < indices.length; f += 3) {
    const a = indices[f] * 3;
    const b = indices[f + 1] * 3;
    const c = indices[f + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (const base of [a, b, c]) {
      normals[base] += nx;
      normals[base + 1] += ny;
      normals[base + 2] += nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
    if (len > 1e-12) {
      normals[i] /= len;
      normals[i + 1] /= len;
      normals[i + 2] /= len;
    } else {
      normals[i + 1] = 1;
    }
  }
  return normals;
}

/** Triangle area and centroid, the two quantities every stage here weights by. */
export function faceGeometry(positions, indices, face) {
  const a = indices[face * 3] * 3;
  const b = indices[face * 3 + 1] * 3;
  const c = indices[face * 3 + 2] * 3;
  const ax = positions[a];
  const ay = positions[a + 1];
  const az = positions[a + 2];
  const ux = positions[b] - ax;
  const uy = positions[b + 1] - ay;
  const uz = positions[b + 2] - az;
  const vx = positions[c] - ax;
  const vy = positions[c + 1] - ay;
  const vz = positions[c + 2] - az;
  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;
  const cross = Math.hypot(nx, ny, nz);
  return {
    area: cross * 0.5,
    centroid: [
      (ax + positions[b] + positions[c]) / 3,
      (ay + positions[b + 1] + positions[c + 1]) / 3,
      (az + positions[b + 2] + positions[c + 2]) / 3,
    ],
    normal: cross > 1e-20 ? [nx / cross, ny / cross, nz / cross] : [0, 1, 0],
  };
}
