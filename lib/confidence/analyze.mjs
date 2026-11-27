/**
 * Confidence analysis of the canonical (original) mesh, and its transfer to the
 * repaired one.
 *
 * Outputs in products/confidence/:
 *   confidence.bin / confidence.json       per-face records and summary
 *   coverage.json                          camera and video coverage
 *   low-confidence-regions.json            actionable regions
 *   confidence-report.md, confidence.glb, confidence.ply
 *   repaired.bin / repaired.json / repaired-regions.json   (after repair)
 */

import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { loadContext } from '../enhance/context.mjs';
import { artifactState, originalModelPath, productsDir, recordArtifact, repairedModelPath } from '../artifacts.mjs';
import { framesPath, readJson, registrationPath, writeJson } from '../multivideo/manifest.mjs';
import { capture } from '../multivideo/video-input.mjs';
import { laplacianVariance } from '../multivideo/imaging.mjs';
import { loadMesh } from '../meshdata.mjs';
import { readProvenance } from '../meshrepair/provenance.mjs';
import { accumulateCoverage, faceSamples } from './camera-coverage.mjs';
import { reprojectionPerFace, reprojectionScore } from './reprojection.mjs';
import { supportPerFace } from './point-support.mjs';
import { diversityScore, spreadDegrees } from './viewing-angles.mjs';
import {
  COMPONENTS, FULL_COVERAGE_CAMERAS, HEADER_SIZE, MIN_AVAILABILITY, PROVENANCE_CAPS, RECORD_FIELDS, RECORD_SIZE,
  applyProvenance, combine, createRecords, normalizeWeights, readRecord, writeRecord,
} from './scoring.mjs';
import { findLowRegions } from './regions.mjs';
import { confidencePly, heatmapGlb, summarize, writeReport } from './export.mjs';

export const confidenceDir = (work) => join(productsDir(work), 'confidence');

const clamp = (value, min, max, fallback) => {
  const n = Number(value);
  return value !== '' && value != null && Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
};

export function normalizeConfidenceSettings(raw = {}) {
  return {
    weights: normalizeWeights(raw.weights),
    lowThreshold: clamp(raw.lowThreshold, 5, 80, 40),
    minAvailability: clamp(raw.minAvailability, 0.1, 0.9, MIN_AVAILABILITY),
    resolution: Math.round(clamp(raw.resolution, 240, 1280, 640)),
    exportPly: raw.exportPly !== false,
  };
}

/**
 * Sharpness of every registered image, relative to the sharpest tenth of this
 * job. All images are decoded by one FFmpeg process.
 */
async function imageQuality(work, views) {
  const ffmpeg = process.env.FFMPEG_PATH || resolve('engines/ffmpeg/bin/ffmpeg.exe');
  const width = 320;
  const height = 180;
  if (!views.length || !existsSync(ffmpeg)) return null;
  const list = join(confidenceDir(work), 'images.ffconcat');
  const escape = (path) => resolve(path).replace(/\\/g, '/').replace(/'/g, "'\\''");
  await writeFile(list, `ffconcat version 1.0\n${views.map((v) => `file '${escape(v.path)}'`).join('\n')}\n`);
  let raw;
  try {
    raw = await capture(ffmpeg, [
      '-v', 'error', '-f', 'concat', '-safe', '0', '-i', list,
      '-vf', `scale=${width}:${height},format=gray`, '-fps_mode', 'passthrough', '-f', 'rawvideo', '-',
    ], { timeoutMs: 15 * 60 * 1000, maxBytes: width * height * (views.length + 4) });
  } catch {
    return null;
  }
  const size = width * height;
  if (Math.floor(raw.length / size) !== views.length) return null;
  const values = views.map((_, i) => laplacianVariance(raw.subarray(i * size, (i + 1) * size), width, height));
  const reference = [...values].sort((a, b) => a - b)[Math.floor(values.length * 0.9)] || 1;
  return new Map(views.map((v, i) => [v.name, Math.min(1, values[i] / reference)]));
}

async function videoStatistics(context, videoIds) {
  const views = await context.views();
  const frames = await readJson(framesPath(context.work));
  const registration = await readJson(registrationPath(context.work));
  return videoIds.map((id) => {
    const own = views.filter((v) => v.video === id);
    const registered = registration?.videos?.find((v) => v.id === id);
    return {
      id,
      label: own[0]?.videoLabel || registered?.label || id,
      registeredImages: own.length,
      selectedFrames: registered?.selectedFrames ?? frames?.perVideo?.find((v) => v.id === id)?.selected ?? null,
      registrationPercent: registered?.registrationPercent ?? null,
    };
  });
}

function describeMetrics(weights, evidence) {
  const rows = {
    coverage: ['Camera coverage', evidence.views, 'measured', 'COLMAP poses and intrinsics with depth-buffer occlusion tests'],
    diversity: ['Viewpoint diversity', evidence.views, 'measured', 'angular spread of unoccluded viewing rays'],
    reprojection: ['Reprojection consistency', evidence.sparse, 'estimated', 'median error of sparse landmarks near each face'],
    support: ['Dense point support', evidence.cloud, 'measured', 'distance to and density of dense cloud points'],
    quality: ['Image quality', evidence.quality, 'estimated', 'Laplacian sharpness of observing frames'],
    depth: ['Depth consistency', evidence.depth, 'measured', 'COLMAP geometric depth maps'],
  };
  return COMPONENTS.map((key) => {
    const [label, available, status, source] = rows[key];
    const missing = key === 'depth'
      ? 'no depth maps (OpenMVS removes them after fusion)'
      : 'the required reconstruction data is not in this workspace';
    return { key, label, weight: weights[key], status: available ? status : 'unavailable', source: available ? source : missing };
  });
}

export async function runConfidence(work, rawSettings = {}, onProgress = () => {}) {
  const started = Date.now();
  const settings = normalizeConfidenceSettings(rawSettings);
  const modelPath = originalModelPath(work);
  const context = await loadContext(work, { modelPath });
  const { mesh } = context;
  const out = confidenceDir(work);
  await mkdir(out, { recursive: true });

  onProgress(0.02, 'Loading cameras and reconstruction evidence');
  const views = await context.views();
  const sparse = await context.sparse();
  const cloud = await context.cloud();
  if (!views.length && !cloud) {
    throw new Error('Confidence is unavailable: this job has neither registered cameras nor a dense cloud.');
  }
  const samples = faceSamples(mesh);
  const quality = await imageQuality(work, views);

  onProgress(0.1, `Testing visibility from ${views.length} cameras`);
  const coverage = await accumulateCoverage(context, samples, {
    resolution: settings.resolution,
    quality,
    onProgress: (f) => onProgress(0.1 + f * 0.55),
  });
  onProgress(0.66, 'Evaluating reprojection error');
  const reprojection = reprojectionPerFace(sparse, samples);
  onProgress(0.72, 'Evaluating dense point support');
  const support = cloud ? supportPerFace(cloud, mesh, samples) : null;

  onProgress(0.82, 'Scoring surfaces');
  const records = createRecords(samples.faces);
  const hasViews = views.length > 0;
  for (let f = 0; f < samples.faces; f += 1) {
    const cameras = hasViews ? coverage.cameras[f] : null;
    const spread = cameras
      ? spreadDegrees(coverage.direction[f * 3], coverage.direction[f * 3 + 1], coverage.direction[f * 3 + 2], cameras) : null;
    const values = {
      coverage: hasViews ? Math.min(1, coverage.effective[f] / FULL_COVERAGE_CAMERAS) : null,
      diversity: cameras ? diversityScore(spread, coverage.bestCos[f], cameras) : null,
      reprojection: reprojection && reprojection.median[f] >= 0 ? reprojectionScore(reprojection.median[f]) : null,
      support: support && support.support[f] >= 0 ? support.support[f] : null,
      quality: coverage.qualityWeight[f] > 0 ? coverage.qualitySum[f] / coverage.qualityWeight[f] : null,
      depth: coverage.depthTested[f] > 0 ? coverage.depthAgree[f] / coverage.depthTested[f] : null,
    };
    const { score, availability } = combine(values, settings.weights, settings.minAvailability);
    let videos = 0;
    for (let m = coverage.videoMask[f]; m; m >>>= 1) videos += m & 1;
    writeRecord(records, f, {
      ...values,
      score,
      availability,
      cameras,
      reprojectionError: reprojection ? reprojection.median[f] : null,
      videos,
      provenance: 0,
      incidence: cameras ? (Math.acos(Math.min(1, coverage.cosSum[f] / cameras)) * 180) / Math.PI : null,
      spread,
      videoMask: coverage.videoMask[f],
    });
  }

  const summary = summarize(records, samples, { videoIds: coverage.videoIds });
  const videos = await videoStatistics(context, coverage.videoIds);
  onProgress(0.9, 'Grouping low-confidence regions');
  const regions = findLowRegions({
    context, mesh, records, samples, occluded: coverage.occluded, grazing: coverage.grazing,
    views, videoIds: coverage.videoIds, threshold: settings.lowThreshold,
  });
  const report = {
    version: 1,
    generatedAt: new Date().toISOString(),
    model: modelPath.split(/[/\\]/).pop(),
    disclaimer: 'Relative reconstruction confidence. Not a probability and not a certification of dimensional accuracy.',
    settings,
    formula: 'score = 100 x sum(weight x component) / sum(weights of available components); unknown (255) when available weight < minAvailability',
    provenanceCaps: PROVENANCE_CAPS,
    metrics: describeMetrics(settings.weights, {
      views: hasViews, sparse: Boolean(reprojection), cloud: Boolean(cloud), quality: Boolean(quality), depth: coverage.depthMapsUsed > 0,
    }),
    cameras: views.length,
    depthMapsUsed: coverage.depthMapsUsed,
    meshWinding: coverage.winding === 1 ? 'outward' : 'inverted (compensated)',
    videos,
    scaleReliable: context.scaleReliable,
    metresPerUnit: context.metresPerUnit,
    summary,
    binary: {
      file: 'confidence.bin',
      layout: `"SFCF", u16 version, u16 recordSize=${RECORD_SIZE}, u32 faceCount, u32 flags; ${HEADER_SIZE}-byte header, then records, then u32 videoMask[faceCount]`,
      fields: RECORD_FIELDS,
      faceOrder: 'face order of the model file: primitives in node order, faces in index order',
      videoMaskBits: coverage.videoIds,
    },
    processingMs: Date.now() - started,
  };

  onProgress(0.94, 'Writing confidence products');
  await writeFile(join(out, 'confidence.bin'), records);
  await writeJson(join(out, 'confidence.json'), report);
  await writeJson(join(out, 'coverage.json'), {
    cameraDistribution: summary.cameraDistribution,
    videos: videos.map((v) => ({ ...v, surfacePercent: summary.videoSurfacePercent.find((s) => s.id === v.id)?.areaPercent ?? 0 })),
    perCamera: coverage.visibleFacesPerCamera,
  });
  await writeJson(join(out, 'low-confidence-regions.json'), regions);
  await writeReport(join(out, 'confidence-report.md'), report, regions);
  await writeFile(join(out, 'confidence.glb'), heatmapGlb(mesh, records));
  if (settings.exportPly) await writeFile(join(out, 'confidence.ply'), confidencePly(mesh, records, samples));
  await recordArtifact(work, 'confidence', { sources: [modelPath], settings });

  const repair = await artifactState(work, 'repair');
  if (repair.current) await attributeRepaired(work);
  onProgress(1);
  return { ...report, regionCount: regions.totalRegions };
}

/**
 * Carry the original mesh's confidence onto the repaired mesh through the
 * per-face source mapping. Scores are never recomputed: surviving faces keep
 * their observed score (limited by what repair did to them) and generated faces
 * have no photographic observation of their own, so they are unknown.
 */
export async function attributeRepaired(work) {
  const out = confidenceDir(work);
  const base = await artifactState(work, 'confidence');
  const repair = await artifactState(work, 'repair');
  if (!base.current || !repair.current) return null;
  const original = await readFile(join(out, 'confidence.bin'));
  const originalReport = await readJson(join(out, 'confidence.json'));
  const { codes, source, count } = await readProvenance(join(productsDir(work), 'repair', 'provenance.bin'));
  const mesh = await loadMesh(repairedModelPath(work));
  const context = await loadContext(work, { modelPath: repairedModelPath(work), mesh });
  const samples = faceSamples(mesh);
  if (samples.faces !== count) throw new Error('Provenance does not match the repaired mesh.');
  const generated = [];
  for (let f = 0; f < count; f += 1) if (source[f] < 0) generated.push(f);
  const cloud = await context.cloud();
  const support = cloud && generated.length ? supportPerFace(cloud, mesh, samples, generated) : null;
  const weights = originalReport.settings.weights;
  const records = createRecords(count);
  for (let f = 0; f < count; f += 1) {
    const code = codes[f];
    if (source[f] >= 0) {
      const r = readRecord(original, source[f]);
      writeRecord(records, f, {
        ...r,
        score: applyProvenance(r.score, code),
        provenance: code,
        coverage: r.coverage == null ? null : r.coverage / 100,
        diversity: r.diversity == null ? null : r.diversity / 100,
        reprojection: r.reprojection == null ? null : r.reprojection / 100,
        support: r.support == null ? null : r.support / 100,
        quality: r.quality == null ? null : r.quality / 100,
        depth: r.depth == null ? null : r.depth / 100,
      });
    } else {
      const value = support && support.support[f] >= 0 ? support.support[f] : null;
      const { score, availability } = combine({ support: value }, weights, originalReport.settings.minAvailability);
      writeRecord(records, f, {
        score: applyProvenance(score, code), availability, support: value, cameras: null, provenance: code,
      });
    }
  }
  const videoIds = originalReport.binary.videoMaskBits;
  const summary = summarize(records, samples, { videoIds });
  const regions = findLowRegions({
    context, mesh, records, samples, views: await context.views(), videoIds, threshold: originalReport.settings.lowThreshold,
  });
  const report = {
    ...originalReport,
    generatedAt: new Date().toISOString(),
    model: 'model-repaired.glb',
    derivation: 'transferred from the original mesh through repair/provenance.bin; generated faces are not scored from neighbours',
    summary,
    binary: { ...originalReport.binary, file: 'repaired.bin' },
  };
  await writeFile(join(out, 'repaired.bin'), records);
  await writeJson(join(out, 'repaired.json'), report);
  await writeJson(join(out, 'repaired-regions.json'), regions);
  await recordArtifact(work, 'confidenceRepaired', {
    sources: [originalModelPath(work), repairedModelPath(work), join(out, 'confidence.bin')],
  });
  return report;
}
