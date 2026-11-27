/**
 * Level-of-detail generation for the active model.
 *
 * Outputs products/lod/{lod1..lod4.glb, manifest.json, lodN.confidence.bin,
 * lodN.provenance.bin} and products/reports/lod-report.json. Level 0 is the
 * source model itself and is referenced, not copied.
 */

import { existsSync, statSync } from 'node:fs';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { activeModel, artifactState, originalModelPath, productsDir, recordArtifact, repairedModelPath } from '../artifacts.mjs';
import { loadMesh, meshScale, splitMesh, weldPositions } from '../meshdata.mjs';
import { writeMeshGlb } from '../glbmesh.mjs';
import { imageSize } from '../image.mjs';
import { readJson, writeJson } from '../multivideo/manifest.mjs';
import { encodeProvenance, readProvenance } from '../meshrepair/provenance.mjs';
import { boundingDiagonal, faceAreas } from '../meshrepair/state.mjs';
import { simplifierReady, simplifyGeometry, simplifyPreserving } from './simplify.mjs';
import { createSourceSampler, rebakeAtlas, textureVariants } from './textures.mjs';
import { ERROR_LIMITS, checkGeometry, createErrorMeter } from './validation.mjs';
import { nearestSources, transferConfidence, transferProvenance } from './transfer.mjs';
import { compressionSupport, normalizeLodSettings } from './settings.mjs';

export const lodDir = (work) => join(productsDir(work), 'lod');

/** Below this many faces the original is cheap enough to display directly. */
const SMALL_MESH = 20000;
/** A coarser level than this saves nothing noticeable and destroys structures. */
const MIN_LEVEL_FACES = 30000;
/** A level must remove at least this share of the previous one to be worth keeping. */
const MIN_STEP = 0.15;
/** Seam-preserving collapse "stalled" if it ends this far above its target. */
const STALL = 1.35;

function textureFacts(images, used) {
  let edge = 0;
  let bytes = 0;
  let gpu = 0;
  for (const index of used) {
    const size = imageSize(images[index].bytes);
    bytes += images[index].bytes.length;
    if (!size) continue;
    edge = Math.max(edge, size.width, size.height);
    gpu += size.width * size.height * 4 * (4 / 3); // RGBA with mipmaps
  }
  return { textureResolution: edge || null, textureCount: used.size, textureBytes: bytes, textureGpuBytes: Math.round(gpu) };
}

const usedImages = (mesh) => new Set(
  mesh.prims.filter((p, slot) => p.imageIndex != null && mesh.facePrim.includes(slot)).map((p) => p.imageIndex),
);

/** positions + normals + UV/colour per vertex, 4 bytes per index. */
const geometryBytes = (vertices, faces, quantized) => vertices * (quantized ? 20 : 32) + faces * 12;

/** The same source vertices with a simplified index buffer. */
function withIndices(mesh, indices) {
  const faces = indices.length / 3;
  const primOf = new Uint16Array(mesh.positions.length / 3);
  for (let f = 0; f < mesh.indices.length / 3; f += 1) {
    for (let k = 0; k < 3; k += 1) primOf[mesh.indices[f * 3 + k]] = mesh.facePrim[f];
  }
  const facePrim = new Uint16Array(faces);
  for (let f = 0; f < faces; f += 1) {
    facePrim[f] = primOf[indices[f * 3]];
    // A face spanning two primitives cannot be textured by either.
    if (primOf[indices[f * 3 + 1]] !== facePrim[f] || primOf[indices[f * 3 + 2]] !== facePrim[f]) return null;
  }
  return { ...mesh, indices, facePrim };
}

/** Geometry with one vertex per position, for seam-free simplification. */
function weldedGeometry(mesh) {
  const welded = weldPositions(mesh.positions);
  const indices = new Uint32Array(mesh.indices.length);
  for (let i = 0; i < indices.length; i += 1) indices[i] = welded.representative[welded.map[mesh.indices[i]]];
  return indices;
}

async function sidecarSources(work, which) {
  const dir = join(productsDir(work), 'confidence');
  const confidenceState = await artifactState(work, which === 'repaired' ? 'confidenceRepaired' : 'confidence');
  const confidencePath = join(dir, which === 'repaired' ? 'repaired.bin' : 'confidence.bin');
  const confidence = confidenceState.current && existsSync(confidencePath) ? await readFile(confidencePath) : null;
  const provenance = which === 'repaired'
    ? (await readProvenance(join(productsDir(work), 'repair', 'provenance.bin'))).codes : null;
  return { confidence, provenance };
}

export async function runLod(work, rawSettings = {}, onProgress = () => {}) {
  const started = Date.now();
  const settings = normalizeLodSettings(rawSettings);
  const ffmpeg = process.env.FFMPEG_PATH || resolve('engines/ffmpeg/bin/ffmpeg.exe');
  const support = compressionSupport(ffmpeg);
  const warnings = [];
  if (settings.textureFormat === 'webp' && !support.webp.available) {
    warnings.push(`WebP is unavailable (${support.webp.reason}); JPEG was used.`);
    settings.textureFormat = 'jpeg';
  }
  const active = await activeModel(work);
  if (!active.path) throw new Error('This job has no 3D mesh to build levels of detail from.');
  await simplifierReady();

  onProgress(0.02, 'Loading the full-resolution model');
  const source = await loadMesh(active.path);
  const metres = meshScale(source);
  const faces = source.indices.length / 3;
  const metadata = await readJson(join(productsDir(work), 'metadata.json'));
  const scaleReliable = metadata?.scaleSource === 'flight-altitude';
  const out = lodDir(work);
  await rm(out, { recursive: true, force: true });
  await mkdir(out, { recursive: true });
  await mkdir(join(productsDir(work), 'reports'), { recursive: true });

  const sourceTextures = textureFacts(source.images, usedImages(source));
  const surfaceM2 = faceAreas(source).reduce((sum, a) => sum + a, 0) * metres * metres;
  /** Metres of surface covered by one texel, averaged over the model. */
  const texelSize = (images, used) => {
    let texels = 0;
    for (const index of used) {
      const size = imageSize(images[index].bytes);
      if (size) texels += size.width * size.height;
    }
    return texels ? Math.round(Math.sqrt(surfaceM2 / texels) * 10000) / 10000 : null;
  };
  const levels = [{
    level: 0,
    file: `../${basename(active.path)}`,
    method: 'original',
    triangleCount: faces,
    vertexCount: source.positions.length / 3,
    fileSizeBytes: statSync(active.path).size,
    geometricError: 0,
    retention: 1,
    ...sourceTextures,
    texelSizeM: texelSize(source.images, usedImages(source)),
    geometryGpuBytes: geometryBytes(source.positions.length / 3, faces, false),
    confidence: null,
    provenance: null,
  }];
  const skipped = [];
  const { confidence, provenance } = await sidecarSources(work, active.which);
  if (confidence) levels[0].confidence = active.which === 'repaired' ? '../confidence/repaired.bin' : '../confidence/confidence.bin';
  if (provenance) levels[0].provenance = '../repair/provenance.bin';

  const wanted = settings.levels.filter((l) => l.enabled);
  if (faces < SMALL_MESH) {
    warnings.push(`The mesh has only ${faces.toLocaleString('en-US')} triangles, so no simplified levels were generated.`);
    wanted.length = 0;
  }

  const diagonal = boundingDiagonal(source);
  const meter = wanted.length ? createErrorMeter(source, metres) : null;
  const textured = source.images.length > 0;
  let sampler = null;
  let preserved = source.indices;
  let welded = null;
  let stalled = false;
  let previousFaces = faces;

  for (let i = 0; i < wanted.length; i += 1) {
    const { level, ratio } = wanted[i];
    const base = 0.08 + (0.9 * i) / wanted.length;
    const span = 0.9 / wanted.length;
    onProgress(base, `Level ${level}: simplifying to about ${Math.round(ratio * 100)}%`);
    const skip = (reason) => skipped.push({ level, targetRetention: ratio, reason });
    if (faces * ratio < MIN_LEVEL_FACES) { skip(`The target would be under ${MIN_LEVEL_FACES} triangles.`); continue; }
    const limit = (settings.errorScale * ERROR_LIMITS[level]);

    let accepted = null;
    let reason = 'No acceptable simplification was found.';
    // A rejected attempt is retried keeping more triangles.
    for (const relax of [1, 1.6]) {
      const target = Math.min(previousFaces * (1 - MIN_STEP), faces * ratio * relax);
      let candidate = null;
      if (!stalled) {
        const result = simplifyPreserving(source, preserved, target, limit * 2, { prune: level >= 3 });
        const mesh = withIndices(source, result.indices);
        const tooMany = result.indices.length / 3 > target * STALL;
        if (mesh && (!tooMany || !textured)) candidate = { mesh, indices: result.indices, method: textured ? 'uv-preserving' : 'colour-preserving' };
        else if (textured) stalled = true;
      }
      if (!candidate) {
        if (!textured) { reason = 'Simplification produced mixed primitives.'; break; }
        welded ||= weldedGeometry(source);
        const result = simplifyGeometry(source.positions, welded, target, limit * 2, { prune: level >= 3 });
        candidate = { mesh: { ...source, indices: result.indices }, indices: result.indices, method: 'rebaked-atlas' };
      }
      const count = candidate.indices.length / 3;
      if (count > previousFaces * (1 - MIN_STEP)) {
        reason = `Within the error limit only ${Math.round((count / faces) * 100)}% could be reached, which is redundant with the previous level.`;
        break;
      }
      const invalid = checkGeometry(source.positions, candidate.indices);
      if (invalid) { reason = invalid; continue; }
      const error = meter.measure(source.positions, candidate.indices);
      if (error.p95Units > limit * diagonal) {
        reason = `Rejected: 95th-percentile error ${error.geometricError} m exceeds the ${(limit * diagonal * metres).toFixed(3)} m limit.`;
        continue;
      }
      accepted = { ...candidate, error };
      break;
    }
    if (!accepted) { skip(reason); continue; }

    let mesh = accepted.mesh;
    let order = null;
    const maxEdge = settings.textureSizes[level];
    const encode = { format: settings.textureFormat === 'auto' ? 'jpeg' : settings.textureFormat, quality: settings.textureQuality, ffmpeg };
    let atlas = null;
    if (accepted.method === 'rebaked-atlas') {
      onProgress(base + span * 0.3, `Level ${level}: baking a new texture atlas from the original`);
      sampler ||= createSourceSampler(source);
      const size = Math.min(maxEdge, sourceTextures.textureResolution || maxEdge);
      const baked = await rebakeAtlas(mesh, sampler, {
        size,
        maxSize: Math.max(size, Math.min(8192, sourceTextures.textureResolution || size)),
        encode,
        onProgress: (f) => onProgress(base + span * (0.3 + 0.5 * f)),
      });
      mesh = baked.mesh;
      order = baked.order;
      atlas = { charts: baked.charts, size: baked.atlasSize };
    } else if (textured) {
      const variants = await textureVariants(source.images, usedImages(mesh), maxEdge, {
        format: settings.textureFormat, quality: settings.textureQuality, ffmpeg,
      });
      mesh = { ...mesh, images: variants.images, passthrough: [] };
    } else {
      mesh = { ...mesh, passthrough: [] };
    }

    const split = splitMesh(mesh);
    const glb = writeMeshGlb({
      primitives: split.primitives,
      images: mesh.images,
      rootMatrix: mesh.rootMatrix,
      unlit: mesh.unlit,
      quantize: settings.quantize,
      generator: 'Skyforge LOD',
      extras: { skyforge: { level, derivedFrom: basename(active.path) } },
    });
    const file = `lod${level}.glb`;
    await writeFile(join(out, file), glb);

    // Output face -> face of the geometry that was simplified.
    const outputOrder = order ? Int32Array.from(split.faceOrder, (f) => order[f]) : split.faceOrder;
    const sidecars = { confidence: null, provenance: null };
    if (confidence || provenance) {
      const nearest = nearestSources({ positions: source.positions, indices: accepted.indices }, meter.originalBvh);
      if (confidence) {
        sidecars.confidence = `lod${level}.confidence.bin`;
        await writeFile(join(out, sidecars.confidence), transferConfidence(nearest, outputOrder, confidence));
      }
      if (provenance) {
        const moved = transferProvenance(nearest, outputOrder, provenance);
        sidecars.provenance = `lod${level}.provenance.bin`;
        await writeFile(join(out, sidecars.provenance), encodeProvenance(moved.codes, moved.source));
      }
    }

    const vertexCount = split.primitives.reduce((sum, p) => sum + p.positions.length / 3, 0);
    const triangleCount = split.faceOrder.length;
    const { p95Units, ...error } = accepted.error;
    const textures = textureFacts(mesh.images, usedImages(mesh));
    levels.push({
      level,
      file,
      method: accepted.method,
      targetRetention: ratio,
      retention: Math.round((triangleCount / faces) * 10000) / 10000,
      triangleCount,
      vertexCount,
      fileSizeBytes: glb.length,
      ...error,
      ...textures,
      texelSizeM: texelSize(mesh.images, usedImages(mesh)),
      atlas,
      geometryGpuBytes: geometryBytes(vertexCount, triangleCount, settings.quantize),
      compression: [settings.quantize ? 'KHR_mesh_quantization' : null, textured && settings.textureFormat === 'webp' ? 'EXT_texture_webp' : null].filter(Boolean),
      ...sidecars,
    });
    previousFaces = triangleCount;
    // Later levels continue from this one.
    if (accepted.method === 'rebaked-atlas') welded = accepted.indices;
    else preserved = accepted.indices;
  }

  for (const level of levels) level.compressionRatio = Math.round((levels[0].fileSizeBytes / level.fileSizeBytes) * 100) / 100;

  const manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    sourceModel: basename(active.path),
    model: active.which,
    geometricErrorDefinition: '95th-percentile surface distance, larger of original-to-level and level-to-original, from area-weighted samples',
    geometricErrorUnits: scaleReliable ? 'metres' : 'metres at an unverified scale',
    sceneDiagonalM: Math.round(diagonal * metres * 100) / 100,
    settings,
    compressionSupport: support,
    levels,
    skipped,
    warnings,
  };
  await writeJson(join(out, 'manifest.json'), manifest);
  await writeJson(join(productsDir(work), 'reports', 'lod-report.json'), { ...manifest, processingMs: Date.now() - started });
  await recordArtifact(work, 'lod', {
    sources: [originalModelPath(work), ...(active.which === 'repaired' ? [repairedModelPath(work)] : [])],
    settings,
    model: active.which,
    followsActive: true,
    withConfidence: Boolean(confidence),
  });
  onProgress(1);
  return manifest;
}
