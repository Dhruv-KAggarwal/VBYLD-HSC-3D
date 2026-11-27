/**
 * Mesh repair. Reads the immutable original model and writes, beside it:
 *
 *   products/model-repaired.glb
 *   products/repair/provenance.bin    per-face origin (see provenance.mjs)
 *   products/repair/removed.glb       the geometry that was removed, in red
 *   products/repair/repair-report.json
 *
 * It needs no COLMAP or OpenMVS run: only the model, and the dense cloud and
 * cameras when the workspace still has them.
 */

import { existsSync } from 'node:fs';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { encodeMesh, faceCount, loadMesh, vertexFaces } from '../meshdata.mjs';
import { faceGeometry, writeMeshGlb } from '../glbmesh.mjs';
import { loadContext } from '../enhance/context.mjs';
import { originalModelPath, productsDir, recordArtifact, repairedModelPath, setActiveModel } from '../artifacts.mjs';
import { writeJson } from '../multivideo/manifest.mjs';
import { analyzeMesh } from './analyze.mjs';
import {
  orientFaces, removeDegenerateFaces, removeDuplicateFaces, removeInvalidFaces, repairNonManifoldEdges, weldDuplicateVertices,
} from './cleanup.mjs';
import { removeFloatingComponents } from './components.mjs';
import { analyzeHoles } from './boundaries.mjs';
import { fillHole } from './hole-filling.mjs';
import { smoothingParameters, smoothMesh } from './smoothing.mjs';
import { appearanceForPatches, bakeVertexColors, validateTextures } from './texture-preservation.mjs';
import { PROVENANCE, PROVENANCE_INFO, isGenerated, summarizeProvenance, writeProvenance } from './provenance.mjs';
import { boundingDiagonal, compactVertices, createState, faceAreas, supported } from './state.mjs';

export const REPAIR_PRESETS = Object.freeze({
  conservative: { floating: 'conservative', maxHoleM: 1, maxHolePercent: 0.4, smoothing: 'off' },
  balanced: { floating: 'balanced', maxHoleM: 2.5, maxHolePercent: 1, smoothing: 'light' },
  aggressive: { floating: 'aggressive', maxHoleM: 5, maxHolePercent: 2, smoothing: 'moderate' },
});

const bool = (value, fallback) => (value === undefined || value === null ? fallback : value === true || value === 'true' || value === 1);
const pick = (value, allowed, fallback) => (allowed.includes(value) ? value : fallback);

export function normalizeRepairSettings(raw = {}) {
  const preset = pick(raw.preset, Object.keys(REPAIR_PRESETS), 'conservative');
  const base = REPAIR_PRESETS[preset];
  const unit = pick(raw.maxHoleUnit, ['m', 'percent'], 'm');
  const size = Number(raw.maxHoleSize);
  return {
    preset,
    removeFloating: bool(raw.removeFloating, true),
    floatingAggressiveness: pick(raw.floatingAggressiveness, ['conservative', 'balanced', 'aggressive'], base.floating),
    removeDuplicates: bool(raw.removeDuplicates, true),
    repairNonManifold: bool(raw.repairNonManifold, true),
    fixOrientation: bool(raw.fixOrientation, true),
    fillHoles: bool(raw.fillHoles, true),
    maxHoleUnit: unit,
    maxHoleSize: Number.isFinite(size) && size > 0
      ? Math.min(unit === 'm' ? 200 : 25, size)
      : (unit === 'm' ? base.maxHoleM : base.maxHolePercent),
    holeStrategy: pick(raw.holeStrategy, ['auto', 'planar', 'boundary', 'curvature'], 'auto'),
    preserveOpenings: bool(raw.preserveOpenings, true),
    smoothing: pick(raw.smoothing, ['off', 'light', 'moderate', 'strong', 'custom'], base.smoothing),
    smoothingIterations: Math.min(60, Math.max(1, Number(raw.smoothingIterations) || 5)),
    edgePreservation: Number.isFinite(Number(raw.edgePreservation)) && raw.edgePreservation !== ''
      ? Math.min(1, Math.max(0, Number(raw.edgePreservation))) : 0.7,
    preserveTextures: bool(raw.preserveTextures, true),
  };
}

/** Append hole patches as their own primitive so existing UVs stay untouched. */
function appendPatches(state, patches) {
  const { mesh } = state;
  const oldVertices = mesh.positions.length / 3;
  const oldFaces = faceCount(mesh);
  const addedVertices = patches.reduce((s, p) => s + p.points.length, 0);
  const addedFaces = patches.reduce((s, p) => s + p.faces.length, 0);
  if (!addedFaces) return;
  const grow = (array, Type, length, fill = 0) => {
    const out = new Type(length);
    if (fill) out.fill(fill);
    if (array) out.set(array);
    return out;
  };
  mesh.positions = grow(mesh.positions, Float32Array, (oldVertices + addedVertices) * 3);
  if (mesh.uvs) mesh.uvs = grow(mesh.uvs, Float32Array, (oldVertices + addedVertices) * 2);
  mesh.colors = grow(mesh.colors, Uint8Array, (oldVertices + addedVertices) * 4, 255);
  mesh.indices = grow(mesh.indices, Uint32Array, (oldFaces + addedFaces) * 3);
  mesh.facePrim = grow(mesh.facePrim, Uint16Array, oldFaces + addedFaces);
  state.source = grow(state.source, Int32Array, oldFaces + addedFaces, -1);
  state.codes = grow(state.codes, Uint8Array, oldFaces + addedFaces);
  const prim = mesh.prims.length;
  mesh.prims.push({ material: { name: 'SkyforgeEstimated' }, imageIndex: null, hasUv: false, hasColor: true });
  let v = oldVertices;
  let f = oldFaces;
  for (const patch of patches) {
    patch.points.forEach((p, i) => {
      mesh.positions.set(p, (v + i) * 3);
      mesh.colors.set([...patch.colors[i], 255], (v + i) * 4);
    });
    for (const [a, b, c] of patch.faces) {
      mesh.indices.set([v + a, v + b, v + c], f * 3);
      mesh.facePrim[f] = prim;
      state.source[f] = -1;
      state.codes[f] = patch.code;
      f += 1;
    }
    v += patch.points.length;
  }
}

async function fillHoles(context, state, settings, maxHoleSize, report) {
  const cloud = await context.cloud();
  const { topology, holes } = analyzeHoles(state.mesh, {
    cloud, up: context.up, metresPerUnit: context.metresPerUnit, maxHoleSize, preserveOpenings: settings.preserveOpenings,
  });
  const edges = new Set();
  for (let e = 0; e < topology.edgeCount; e += 1) edges.add(`${topology.edgeA[e]}:${topology.edgeB[e]}`);
  const adjacency = vertexFaces(topology);
  const { mesh } = state;
  const patches = [];
  const skipped = [];
  for (const hole of holes) {
    const record = {
      id: hole.id,
      classification: hole.classification,
      reason: hole.reason,
      boundaryVertices: hole.loop.vertices.length,
      perimeterM: Math.round(hole.perimeter * context.metresPerUnit * 100) / 100,
      diameterM: Math.round(hole.diameter * context.metresPerUnit * 100) / 100,
      planarity: Math.round(hole.planarity * 1000) / 1000,
      curvatureDegrees: Math.round(hole.curvatureDegrees),
      pointSupport: hole.support == null ? null : Math.round(hole.support * 100) / 100,
      visibleBehind: hole.behind == null ? null : Math.round(hole.behind * 100) / 100,
    };
    if (!hole.fillable) { skipped.push(record); continue; }
    const ids = hole.loop.vertices;
    if (new Set(ids).size !== ids.length) {
      skipped.push({ ...record, reason: 'The boundary touches itself; closing it would create non-manifold geometry.' });
      continue;
    }
    const inLoop = new Set(ids);
    const rings = ids.map((id) => {
      const out = new Map();
      for (let i = adjacency.start[id]; i < adjacency.start[id + 1]; i += 1) {
        const f = adjacency.list[i];
        for (let k = 0; k < 3; k += 1) {
          const u = topology.tri[f * 3 + k];
          if (inLoop.has(u) || out.has(u)) continue;
          const r = topology.welded.representative[u] * 3;
          out.set(u, [mesh.positions[r], mesh.positions[r + 1], mesh.positions[r + 2]]);
        }
      }
      return [...out.values()];
    });
    const rimArea = hole.loop.halves.reduce((s, h) => s + faceGeometry(mesh.positions, mesh.indices, (h / 3) | 0).area, 0) / hole.loop.halves.length;
    const patch = fillHole(hole.points, {
      strategy: settings.holeStrategy,
      planarity: hole.planarity,
      targetArea: rimArea,
      rings,
      existingEdge: (i, j) => edges.has(ids[i] < ids[j] ? `${ids[i]}:${ids[j]}` : `${ids[j]}:${ids[i]}`),
    });
    if (!patch) {
      skipped.push({ ...record, reason: 'No valid triangulation exists for this boundary.' });
      continue;
    }
    for (const [a, b, c] of patch.faces) {
      // Register new diagonals so a neighbouring hole cannot reuse them.
      for (const [i, j] of [[a, b], [b, c], [c, a]]) {
        if (i < ids.length && j < ids.length) edges.add(ids[i] < ids[j] ? `${ids[i]}:${ids[j]}` : `${ids[j]}:${ids[i]}`);
      }
    }
    patches.push({
      ...patch,
      normal: hole.surfaceNormal,
      support: hole.support,
      code: hole.classification === 'supported-gap' ? PROVENANCE.INTERPOLATED : PROVENANCE.HOLE_FILLED,
      record: { ...record, strategy: patch.strategy, triangles: patch.faces.length },
    });
  }
  const looks = await appearanceForPatches(context, patches);
  patches.forEach((patch, i) => {
    patch.colors = looks[i].colors;
    patch.record.appearance = looks[i].appearance;
    patch.record.sourceCamera = looks[i].camera;
  });
  appendPatches(state, patches);
  const count = (key) => skipped.filter((s) => s.classification === key).length;
  report.holes = {
    detected: holes.length,
    filled: patches.length,
    filledDetails: patches.map((p) => p.record).slice(0, 500),
    leftOpen: skipped.length,
    leftOpenByClass: {
      outerBoundary: count('outer-boundary'),
      suspectedOpening: count('suspected-opening'),
      largeMissing: count('large-missing'),
      largeGap: count('large-gap'),
      untriangulable: count('unsupported-gap') + count('supported-gap'),
    },
    leftOpenDetails: skipped.filter((s) => s.classification !== 'outer-boundary').sort((a, b) => b.diameterM - a.diameterM).slice(0, 500),
  };
}

/** Original geometry that has no dense-point evidence at all. */
async function markUnsupported(context, state) {
  const cloud = await context.cloud();
  if (!cloud) return null;
  const { mesh } = state;
  let marked = 0;
  for (let f = 0; f < faceCount(mesh); f += 1) {
    if (isGenerated(state.codes[f])) continue;
    const g = faceGeometry(mesh.positions, mesh.indices, f);
    const radius = Math.max(4 * cloud.spacing, Math.sqrt(g.area) * 1.5);
    if (supported(cloud, ...g.centroid, radius)) continue;
    let near = false;
    for (let k = 0; k < 3 && !near; k += 1) {
      const v = mesh.indices[f * 3 + k] * 3;
      near = supported(cloud, mesh.positions[v], mesh.positions[v + 1], mesh.positions[v + 2], radius);
    }
    if (!near) { state.codes[f] = PROVENANCE.UNSUPPORTED; marked += 1; }
  }
  return marked;
}

async function writeRemoved(path, original, removedIds) {
  if (!removedIds.length) { await rm(path, { force: true }); return false; }
  const positions = new Float32Array(removedIds.length * 9);
  removedIds.forEach((f, i) => {
    for (let k = 0; k < 3; k += 1) {
      const v = original.indices[f * 3 + k];
      for (let a = 0; a < 3; a += 1) {
        const value = v * 3 + a < original.positions.length ? original.positions[v * 3 + a] : 0;
        positions[i * 9 + k * 3 + a] = Number.isFinite(value) ? value : 0;
      }
    }
  });
  const count = removedIds.length * 3;
  const colors = new Uint8Array(count * 4);
  for (let i = 0; i < count; i += 1) colors.set([239, 68, 68, 255], i * 4);
  await writeFile(path, writeMeshGlb({
    primitives: [{
      mode: 4,
      positions,
      indices: Uint32Array.from({ length: count }, (_, i) => i),
      colors: { values: colors, componentType: 5121, stride: 4 },
      imageIndex: null,
    }],
    rootMatrix: original.rootMatrix,
    generator: 'Skyforge repair (removed geometry)',
  }));
  return true;
}

export async function runRepair(work, rawSettings = {}, onProgress = () => {}) {
  const started = Date.now();
  const settings = normalizeRepairSettings(rawSettings);
  const originalPath = originalModelPath(work);
  const context = await loadContext(work, { modelPath: originalPath });
  const original = context.mesh;
  const outDir = join(productsDir(work), 'repair');
  await mkdir(outDir, { recursive: true });

  onProgress(0.02, 'Analysing the original mesh');
  const before = await analyzeMesh(context, original, { onProgress: (f) => onProgress(0.02 + f * 0.2) });

  // Edit a second copy; `original` stays as read for comparison.
  const state = createState(await loadMesh(originalPath));
  const report = { settings, warnings: [], steps: {} };
  const removed = [];
  const reasons = {};
  const step = (name, ids) => {
    reasons[name] = ids.length;
    for (const id of ids) removed.push(id);
  };

  onProgress(0.24, 'Removing invalid geometry');
  step('invalid', removeInvalidFaces(state));
  if (!settings.preserveTextures && state.mesh.images.length) {
    bakeVertexColors(state.mesh);
    report.warnings.push('Textures were replaced by per-vertex colour because "preserve textures" is off.');
  }
  if (settings.removeDuplicates) {
    report.steps.welding = weldDuplicateVertices(state);
  }
  step('degenerate', removeDegenerateFaces(state));
  if (settings.removeDuplicates) step('duplicate', removeDuplicateFaces(state));
  onProgress(0.34, 'Repairing topology');
  if (settings.repairNonManifold) {
    const outcome = repairNonManifoldEdges(state);
    step('nonManifoldFins', outcome.removed);
    report.steps.nonManifold = { removedFins: outcome.removed.length, edgesLeftUnchanged: outcome.remaining };
  }
  if (settings.fixOrientation) report.steps.orientation = { flippedFaces: orientFaces(state) };

  if (settings.removeFloating) {
    onProgress(0.44, 'Removing floating geometry');
    const outcome = removeFloatingComponents(state, {
      cloud: await context.cloud(), aggressiveness: settings.floatingAggressiveness, metresPerUnit: context.metresPerUnit,
    });
    step('floating', outcome.removed);
    report.steps.floating = {
      aggressiveness: settings.floatingAggressiveness,
      rules: outcome.rules,
      componentsExamined: outcome.totalComponents,
      componentsRemoved: outcome.removedComponents.length,
      removed: outcome.removedComponents.slice(0, 300),
    };
    if (!(await context.cloud())) report.warnings.push('No dense cloud is available, so floating geometry was judged on size and distance only.');
  }
  compactVertices(state);

  let maxHoleSize = 0;
  if (settings.fillHoles) {
    onProgress(0.55, 'Detecting and filling holes');
    const diagonal = boundingDiagonal(state.mesh);
    if (settings.maxHoleUnit === 'm' && context.scaleReliable) {
      maxHoleSize = settings.maxHoleSize / context.metresPerUnit;
    } else if (settings.maxHoleUnit === 'percent') {
      maxHoleSize = (settings.maxHoleSize / 100) * diagonal;
    } else {
      const percent = REPAIR_PRESETS[settings.preset].maxHolePercent;
      maxHoleSize = (percent / 100) * diagonal;
      report.warnings.push(`The scene has no reliable metric scale, so the hole limit of ${settings.maxHoleSize} m was replaced by ${percent}% of the scene diagonal.`);
    }
    report.maxHoleSizeM = Math.round(maxHoleSize * context.metresPerUnit * 100) / 100;
    await fillHoles(context, state, settings, maxHoleSize, report);
  }

  const smoothing = smoothingParameters({
    strength: settings.smoothing, iterations: settings.smoothingIterations, edgePreservation: settings.edgePreservation,
  });
  if (smoothing) {
    onProgress(0.72, 'Edge-preserving smoothing');
    report.steps.smoothing = smoothMesh(state, smoothing);
    report.steps.smoothing.maxDisplacementM = report.steps.smoothing.maxDisplacement * context.metresPerUnit;
  }

  onProgress(0.78, 'Recording geometry provenance');
  report.steps.unsupportedFaces = await markUnsupported(context, state);
  compactVertices(state);
  if (!faceCount(state.mesh)) throw new Error('Repair would remove every face; nothing was written.');

  const { glb, faceOrder } = encodeMesh(state.mesh, {
    generator: 'Skyforge mesh repair',
    extras: { skyforge: { derivedFrom: 'model.glb', provenance: 'repair/provenance.bin', settings } },
  });
  const codes = new Uint8Array(faceOrder.length);
  const source = new Int32Array(faceOrder.length);
  for (let i = 0; i < faceOrder.length; i += 1) {
    codes[i] = state.codes[faceOrder[i]];
    source[i] = state.source[faceOrder[i]];
  }
  await writeFile(repairedModelPath(work), glb);
  await writeProvenance(join(outDir, 'provenance.bin'), codes, source);
  await writeJson(join(outDir, 'provenance.json'), {
    schema: 'SFPV v1: "SFPV", u16 version, u16 reserved, u32 faceCount, u8 code[faceCount] (padded to 4), i32 sourceFace[faceCount]',
    faceOrder: 'primitive order of model-repaired.glb; primitive.extras.skyforgeFaceOffset gives each primitive\'s first face',
    categories: PROVENANCE_INFO,
  });
  const wroteRemoved = await writeRemoved(join(outDir, 'removed.glb'), original, removed);

  if (settings.preserveTextures) report.textures = validateTextures(original, glb);
  if (report.textures && !report.textures.valid) report.warnings.push(...report.textures.issues);

  onProgress(0.84, 'Analysing the repaired mesh');
  const repaired = await loadMesh(repairedModelPath(work));
  const after = await analyzeMesh(context, repaired, { onProgress: (f) => onProgress(0.84 + f * 0.14) });

  const areas = faceAreas(repaired);
  const provenance = summarizeProvenance(codes, areas);
  const originalVertices = original.positions;
  // Vertices whose position no longer exists in the original were moved or made.
  const seen = new Set();
  for (let i = 0; i < originalVertices.length; i += 3) seen.add(`${originalVertices[i]},${originalVertices[i + 1]},${originalVertices[i + 2]}`);
  let modifiedVertices = 0;
  for (let i = 0; i < repaired.positions.length; i += 3) {
    if (!seen.has(`${repaired.positions[i]},${repaired.positions[i + 1]},${repaired.positions[i + 2]}`)) modifiedVertices += 1;
  }

  Object.assign(report, {
    originalTriangles: before.triangles,
    repairedTriangles: after.triangles,
    removedComponents: report.steps.floating?.componentsRemoved || 0,
    removedFaces: removed.length,
    removedByReason: reasons,
    removedGeometryFile: wroteRemoved ? 'repair/removed.glb' : null,
    filledHoles: report.holes?.filled || 0,
    surfaceAreaBeforeM2: before.surfaceAreaM2,
    surfaceAreaAfterM2: after.surfaceAreaM2,
    surfaceAreaChangeM2: Math.round((after.surfaceAreaM2 - before.surfaceAreaM2) * 100) / 100,
    scaleReliable: context.scaleReliable,
    modifiedVertices,
    // Generated surface only: interpolated and hole-filled.
    estimatedGeometryPercent: Math.round(provenance.filter((p) => isGenerated(p.code)).reduce((s, p) => s + p.areaPercent, 0) * 100) / 100,
    provenance,
    before,
    after,
    processingMs: Date.now() - started,
    generatedAt: new Date().toISOString(),
  });
  await writeJson(join(outDir, 'repair-report.json'), report);
  await writeJson(join(outDir, 'mesh-quality.json'), { original: before, repaired: after });

  const sources = [originalPath, repairedModelPath(work)].filter(existsSync);
  await recordArtifact(work, 'repair', { sources, settings });
  await setActiveModel(work, 'repaired');
  onProgress(1);
  return report;
}
