/** Summaries, heatmap exports and the human-readable confidence report. */

import { writeFile } from 'node:fs/promises';
import { writeMeshGlb, applyMatrix } from '../glbmesh.mjs';
import { BANDS, UNKNOWN, readRecord } from './scoring.mjs';

/** Default and colour-vision-safe (viridis-derived) ramps, very low to high. */
export const PALETTES = Object.freeze({
  standard: { veryLow: [220, 38, 38], low: [249, 115, 22], moderate: [250, 204, 21], good: [163, 230, 53], high: [34, 197, 94], unknown: [128, 128, 128] },
  accessible: { veryLow: [68, 1, 84], low: [59, 82, 139], moderate: [33, 145, 140], good: [94, 201, 98], high: [253, 231, 37], unknown: [150, 150, 150] },
});

export function bandOf(score) {
  if (score === UNKNOWN) return 'unknown';
  return BANDS.find((b) => score >= b.min).key;
}

const round = (value, digits = 2) => (value == null ? null : Math.round(value * 10 ** digits) / 10 ** digits);

/** Area-weighted statistics over one set of records. */
export function summarize(records, samples, { videoIds = [] } = {}) {
  const n = samples.faces;
  const bands = Object.fromEntries([...BANDS.map((b) => [b.key, 0]), ['unknown', 0]]);
  const histogram = new Array(10).fill(0);
  const scores = new Float64Array(101);
  const cameraBins = [
    { label: '0', min: 0, max: 0 }, { label: '1', min: 1, max: 1 }, { label: '2', min: 2, max: 2 },
    { label: '3-5', min: 3, max: 5 }, { label: '6-10', min: 6, max: 10 }, { label: '11+', min: 11, max: Infinity },
  ].map((b) => ({ ...b, area: 0 }));
  const perVideo = new Float64Array(videoIds.length);
  const componentArea = { coverage: 0, diversity: 0, reprojection: 0, support: 0, quality: 0, depth: 0 };
  let total = 0;
  let known = 0;
  let weighted = 0;
  let cameraKnown = 0;
  let estimated = 0;
  let repaired = 0;
  let unsupported = 0;
  let availability = 0;
  for (let f = 0; f < n; f += 1) {
    const r = readRecord(records, f);
    const a = samples.area[f];
    total += a;
    availability += r.availability * a;
    bands[bandOf(r.score)] += a;
    if (r.score !== UNKNOWN) {
      known += a;
      weighted += r.score * a;
      scores[r.score] += a;
      histogram[Math.min(9, Math.floor(r.score / 10))] += a;
    }
    if (r.cameras != null) {
      cameraKnown += a;
      cameraBins.find((b) => r.cameras >= b.min && r.cameras <= b.max).area += a;
    }
    for (let i = 0; i < videoIds.length; i += 1) if (r.videoMask & (1 << i)) perVideo[i] += a;
    for (const key of Object.keys(componentArea)) if (r[key] != null) componentArea[key] += a;
    if (r.provenance === 3 || r.provenance === 4) estimated += a;
    if (r.provenance >= 1 && r.provenance <= 4) repaired += a;
    if (r.provenance === 5) unsupported += a;
  }
  const percent = (area, base = total) => (base ? round((area / base) * 100) : 0);
  let median = null;
  let running = 0;
  for (let s = 0; s <= 100; s += 1) {
    running += scores[s];
    if (known && running >= known / 2) { median = s; break; }
  }
  return {
    weighting: 'surface area',
    faceCount: n,
    meanScore: known ? round(weighted / known, 1) : null,
    medianScore: median,
    meanEvidenceAvailabilityPercent: total ? round((availability / total) * 100, 1) : 0,
    evidenceCoveragePercent: percent(known),
    bandsPercent: Object.fromEntries(Object.entries(bands).map(([k, v]) => [k, percent(v)])),
    highPercent: percent(bands.high + bands.good),
    moderatePercent: percent(bands.moderate),
    lowPercent: percent(bands.low + bands.veryLow),
    unknownPercent: percent(bands.unknown),
    estimatedGeometryPercent: percent(estimated),
    repairedGeometryPercent: percent(repaired),
    unsupportedGeometryPercent: percent(unsupported),
    histogram: histogram.map((area, i) => ({ from: i * 10, to: i === 9 ? 100 : i * 10 + 9, areaPercent: percent(area, known) })),
    cameraDistribution: cameraBins.map((b) => ({ cameras: b.label, areaPercent: percent(b.area, cameraKnown) })),
    videoSurfacePercent: videoIds.map((id, i) => ({ id, areaPercent: percent(perVideo[i]) })),
    metricAvailabilityPercent: Object.fromEntries(Object.entries(componentArea).map(([k, v]) => [k, percent(v)])),
  };
}

export function heatColor(score, palette = PALETTES.standard) {
  return palette[bandOf(score)];
}

/** A standalone vertex-coloured heatmap model (one colour per face). */
export function heatmapGlb(mesh, records) {
  const faces = mesh.indices.length / 3;
  const positions = new Float32Array(faces * 9);
  const colors = new Uint8Array(faces * 12);
  for (let f = 0; f < faces; f += 1) {
    const color = heatColor(readRecord(records, f).score);
    for (let k = 0; k < 3; k += 1) {
      const v = mesh.indices[f * 3 + k] * 3;
      positions.set([mesh.positions[v], mesh.positions[v + 1], mesh.positions[v + 2]], f * 9 + k * 3);
      colors.set([color[0], color[1], color[2], 255], f * 12 + k * 4);
    }
  }
  return writeMeshGlb({
    primitives: [{
      mode: 4,
      positions,
      indices: Uint32Array.from({ length: faces * 3 }, (_, i) => i),
      colors: { values: colors, componentType: 5121, stride: 4 },
      imageIndex: null,
    }],
    rootMatrix: mesh.rootMatrix,
    generator: 'Skyforge confidence heatmap',
  });
}

/**
 * Binary PLY in world coordinates with per-vertex confidence (area-weighted
 * mean of the adjacent scored faces) and mean supporting camera count.
 */
export function confidencePly(mesh, records, samples) {
  const vertices = mesh.positions.length / 3;
  const faces = mesh.indices.length / 3;
  const score = new Float64Array(vertices);
  const weight = new Float64Array(vertices);
  const cameras = new Float64Array(vertices);
  const cameraWeight = new Float64Array(vertices);
  for (let f = 0; f < faces; f += 1) {
    const r = readRecord(records, f);
    const a = samples.area[f] || 1e-12;
    for (let k = 0; k < 3; k += 1) {
      const v = mesh.indices[f * 3 + k];
      if (r.score !== UNKNOWN) { score[v] += r.score * a; weight[v] += a; }
      if (r.cameras != null) { cameras[v] += r.cameras * a; cameraWeight[v] += a; }
    }
  }
  const world = Float32Array.from(mesh.positions);
  if (mesh.rootMatrix) applyMatrix(world, mesh.rootMatrix, 1);
  const header = Buffer.from([
    'ply', 'format binary_little_endian 1.0',
    'comment confidence: 0-100 relative reconstruction confidence, 255 unknown',
    'comment cameras: mean unoccluded observers of adjacent faces, 65535 unavailable',
    `element vertex ${vertices}`,
    'property float x', 'property float y', 'property float z',
    'property uchar red', 'property uchar green', 'property uchar blue',
    'property uchar confidence', 'property ushort cameras',
    `element face ${faces}`, 'property list uchar uint vertex_indices', 'end_header\n',
  ].join('\n'));
  const body = Buffer.alloc(vertices * 18 + faces * 13);
  let o = 0;
  for (let v = 0; v < vertices; v += 1) {
    const value = weight[v] ? Math.round(score[v] / weight[v]) : UNKNOWN;
    const color = heatColor(value);
    body.writeFloatLE(world[v * 3], o);
    body.writeFloatLE(world[v * 3 + 1], o + 4);
    body.writeFloatLE(world[v * 3 + 2], o + 8);
    body.set(color, o + 12);
    body[o + 15] = value;
    body.writeUInt16LE(cameraWeight[v] ? Math.min(65534, Math.round(cameras[v] / cameraWeight[v])) : 65535, o + 16);
    o += 18;
  }
  for (let f = 0; f < faces; f += 1) {
    body[o] = 3;
    body.writeUInt32LE(mesh.indices[f * 3], o + 1);
    body.writeUInt32LE(mesh.indices[f * 3 + 1], o + 5);
    body.writeUInt32LE(mesh.indices[f * 3 + 2], o + 9);
    o += 13;
  }
  return Buffer.concat([header, body]);
}

export function reportMarkdown(report, regions) {
  const s = report.summary;
  const show = (v, unit = '') => (v == null ? 'unavailable' : `${v}${unit}`);
  const lines = [
    '# Reconstruction confidence report',
    '',
    `Model: ${report.model} · ${s.faceCount.toLocaleString('en-US')} faces · generated ${report.generatedAt}`,
    '',
    '> Confidence is a relative measure of the photographic and geometric evidence behind each surface.',
    '> It is not a probability and does not certify dimensional accuracy.',
    '',
    '## Summary (weighted by surface area)',
    '',
    '| Measure | Value |',
    '|---|---|',
    `| Mean confidence | ${show(s.meanScore)} |`,
    `| Median confidence | ${show(s.medianScore)} |`,
    `| Surface with enough evidence to score | ${s.evidenceCoveragePercent}% |`,
    `| High or good (60-100) | ${s.highPercent}% |`,
    `| Moderate (40-59) | ${s.moderatePercent}% |`,
    `| Low (0-39) | ${s.lowPercent}% |`,
    `| Unknown | ${s.unknownPercent}% |`,
    `| Estimated (generated) geometry | ${s.estimatedGeometryPercent}% |`,
    `| Repaired geometry | ${s.repairedGeometryPercent}% |`,
    `| Low-confidence regions | ${regions.totalRegions} |`,
    '',
    '## Evidence used',
    '',
    '| Metric | Weight | Status | Source |',
    '|---|---|---|---|',
    ...report.metrics.map((m) => `| ${m.label} | ${Math.round(m.weight * 100)}% | ${m.status} | ${m.source} |`),
    '',
    '## Largest low-confidence regions',
    '',
  ];
  if (!regions.regions.length) lines.push('None above the size threshold.');
  for (const region of regions.regions.slice(0, 15)) {
    lines.push(`### ${region.id}: ${region.areaM2} m² (${region.orientation})`);
    lines.push(`Average confidence ${show(region.averageConfidence)}, ${region.unknownPercent}% unknown, ${region.meanSupportingCameras} supporting cameras on average.`);
    for (const cause of region.causes) lines.push(`- Cause: ${cause.label} (${Math.round(cause.areaShare * 100)}% of the region)`);
    for (const advice of region.recommendations) lines.push(`- Recommendation: ${advice}`);
    lines.push('');
  }
  return `${lines.join('\n')}\n`;
}

export async function writeReport(path, report, regions) {
  await writeFile(path, reportMarkdown(report, regions));
}
