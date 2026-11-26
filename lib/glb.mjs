import { computeNormals } from './geometry.mjs';

function pad4(buffer, fill = 0x20) {
  const extra = (4 - (buffer.length % 4)) % 4;
  return extra ? Buffer.concat([buffer, Buffer.alloc(extra, fill)]) : buffer;
}

function accessorMinMax(values, stride) {
  const min = Array(stride).fill(Infinity);
  const max = Array(stride).fill(-Infinity);
  for (let i = 0; i < values.length; i += stride) {
    for (let k = 0; k < stride; k += 1) {
      min[k] = Math.min(min[k], values[i + k]);
      max[k] = Math.max(max[k], values[i + k]);
    }
  }
  return { min, max };
}

/**
 * Expand tightly packed RGB bytes to RGBA.
 *
 * glTF 2.0 requires every vertex attribute element to start on a 4-byte
 * boundary. An UNSIGNED_BYTE VEC3 colour is 3 bytes per element, so a tightly
 * packed COLOR_0 buffer violates that: strict loaders reject the file and
 * lenient ones read it with an implicit 4-byte stride, which smears red, green
 * and blue across neighbouring vertices. VEC4 with an opaque alpha is the
 * spec-compliant way to carry byte colour.
 */
function toRgba(rgb, vertexCount) {
  const src = rgb instanceof Uint8Array ? rgb : new Uint8Array(rgb || []);
  const out = new Uint8Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i += 1) {
    out[i * 4] = src[i * 3] ?? 150;
    out[i * 4 + 1] = src[i * 3 + 1] ?? 150;
    out[i * 4 + 2] = src[i * 3 + 2] ?? 150;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** Sniff the container so the GLB never lies about what the image bytes are. */
function sniffImageMime(bytes) {
  if (!bytes || bytes.length < 4) return 'image/png';
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return 'image/png';
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'image/jpeg';
  return 'image/png';
}

export function writeGlb(mesh, pointCloud, options = {}) {
  // Unlit is the honest material for reconstructed photography: the colour in
  // the atlas already contains the real lighting of the scene, so re-lighting it
  // double-shades the model. It also means a viewer with no lights and no
  // environment map renders the photo rather than black.
  const unlit = options.unlit !== false;
  const chunks = [];
  const views = [];
  const accessors = [];
  const pushBuffer = (data, target = 0, byteStride = 0) => {
    const aligned = pad4(Buffer.from(data.buffer, data.byteOffset, data.byteLength), 0);
    const view = { buffer: 0, byteOffset: chunks.reduce((sum, item) => sum + item.length, 0), byteLength: data.byteLength };
    if (target) view.target = target;
    if (byteStride) view.byteStride = byteStride;
    views.push(view);
    chunks.push(aligned);
    return views.length - 1;
  };

  const vertexCount = mesh.positions.length / 3;
  const textured = Boolean(mesh.ortho?.length && mesh.uvs?.length && mesh.textured !== false);
  const meshNormals = computeNormals(mesh.positions, mesh.indices);
  const meshPosStats = accessorMinMax(mesh.positions, 3);

  const attributes = {};
  attributes.POSITION = accessors.length;
  accessors.push({
    bufferView: pushBuffer(mesh.positions, 34962),
    componentType: 5126,
    count: vertexCount,
    type: 'VEC3',
    min: meshPosStats.min,
    max: meshPosStats.max,
  });
  attributes.NORMAL = accessors.length;
  accessors.push({ bufferView: pushBuffer(meshNormals, 34962), componentType: 5126, count: vertexCount, type: 'VEC3' });
  if (textured) {
    attributes.TEXCOORD_0 = accessors.length;
    accessors.push({ bufferView: pushBuffer(mesh.uvs, 34962), componentType: 5126, count: mesh.uvs.length / 2, type: 'VEC2' });
  }
  // COLOR_0 multiplies baseColorTexture per the glTF spec, so emitting both
  // squares the albedo and the surface comes out dark and desaturated. Vertex
  // colour is the alternative to the atlas, never a companion to it.
  if (!textured) {
    attributes.COLOR_0 = accessors.length;
    accessors.push({
      bufferView: pushBuffer(toRgba(mesh.colors, vertexCount), 34962),
      componentType: 5121,
      count: vertexCount,
      type: 'VEC4',
      normalized: true,
    });
  }
  const indicesAccessor = accessors.length;
  accessors.push({ bufferView: pushBuffer(mesh.indices, 34963), componentType: 5125, count: mesh.indices.length, type: 'SCALAR' });

  const primitives = [{ attributes, indices: indicesAccessor, material: 0, mode: 4 }];
  const meshes = [{ name: 'Terrain', primitives }];
  const nodes = [{ name: 'Terrain', mesh: 0 }];

  if (pointCloud?.positions?.length) {
    const cloudCount = pointCloud.positions.length / 3;
    const cloudPos = accessorMinMax(pointCloud.positions, 3);
    const posIndex = accessors.length;
    accessors.push({
      bufferView: pushBuffer(pointCloud.positions, 34962),
      componentType: 5126,
      count: cloudCount,
      type: 'VEC3',
      min: cloudPos.min,
      max: cloudPos.max,
    });
    const colIndex = accessors.length;
    accessors.push({
      bufferView: pushBuffer(toRgba(pointCloud.colors, cloudCount), 34962),
      componentType: 5121,
      count: cloudCount,
      type: 'VEC4',
      normalized: true,
    });
    meshes.push({
      name: 'PhotoPoints',
      primitives: [{ attributes: { POSITION: posIndex, COLOR_0: colIndex }, material: 1, mode: 0 }],
    });
    nodes.push({ name: 'PhotoPoints', mesh: 1 });
  }

  const pbr = { metallicFactor: 0, roughnessFactor: 1, baseColorFactor: [1, 1, 1, 1] };
  const surface = { name: 'TerrainSurface', pbrMetallicRoughness: pbr, doubleSided: true };
  const points = {
    name: 'PhotoPoints',
    pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1 },
  };
  if (unlit) {
    surface.extensions = { KHR_materials_unlit: {} };
    points.extensions = { KHR_materials_unlit: {} };
  }

  const json = {
    asset: { version: '2.0', generator: 'Skyforge single-pass reconstructor' },
    scene: 0,
    scenes: [{ nodes: nodes.map((_, index) => index) }],
    nodes,
    meshes,
    materials: [surface, points],
    accessors,
    bufferViews: views,
    buffers: [{ byteLength: 0 }],
  };
  if (unlit) {
    // Declared as used but not required: a viewer that ignores the extension
    // still gets a correct PBR material, so the file stays universally loadable.
    json.extensionsUsed = ['KHR_materials_unlit'];
  }
  if (textured) {
    const imageView = pushBuffer(mesh.ortho);
    json.images = [{ bufferView: imageView, mimeType: mesh.orthoMime || sniffImageMime(mesh.ortho) }];
    json.samplers = [{ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 }];
    json.textures = [{ sampler: 0, source: 0 }];
    pbr.baseColorTexture = { index: 0, texCoord: 0 };
  }
  json.buffers[0].byteLength = chunks.reduce((sum, item) => sum + item.length, 0);
  const jsonChunk = pad4(Buffer.from(JSON.stringify(json)));
  const binChunk = Buffer.concat(chunks);
  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(jsonChunk.length, 0);
  jsonHeader.writeUInt32LE(0x4e4f534a, 4);
  const binHeader = Buffer.alloc(8);
  binHeader.writeUInt32LE(binChunk.length, 0);
  binHeader.writeUInt32LE(0x004e4942, 4);
  const body = Buffer.concat([jsonHeader, jsonChunk, binHeader, binChunk]);
  const header = Buffer.alloc(12);
  header.writeUInt32LE(0x46546c67, 0);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + body.length, 8);
  return Buffer.concat([header, body]);
}
