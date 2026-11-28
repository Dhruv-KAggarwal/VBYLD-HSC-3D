/**
 * The enhancement operations that can run during a reconstruction or on their
 * own against a finished job. None of them re-runs COLMAP or OpenMVS.
 */

import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { artifactState, originalModelPath, productsDir, recordArtifact, repairedModelPath } from '../artifacts.mjs';
import { writeJson } from '../multivideo/manifest.mjs';
import { loadContext } from './context.mjs';
import { analyzeMesh } from '../meshrepair/analyze.mjs';
import { runRepair } from '../meshrepair/repair.mjs';
import { attributeRepaired, runConfidence } from '../confidence/analyze.mjs';
import { runLod } from '../lod/generate.mjs';
import { runCompletion } from '../completion/complete.mjs';

export const OPERATIONS = Object.freeze(['analyze', 'confidence', 'repair', 'lod', 'completion']);

async function runAnalysis(work, _settings, onProgress) {
  const original = originalModelPath(work);
  const context = await loadContext(work, { modelPath: original });
  onProgress(0.05, 'Analysing mesh topology');
  const result = { original: await analyzeMesh(context, context.mesh, { onProgress: (f) => onProgress(0.05 + f * 0.9) }) };
  const repair = await artifactState(work, 'repair');
  if (repair.current) {
    const repaired = await loadContext(work, { modelPath: repairedModelPath(work) });
    result.repaired = await analyzeMesh(repaired, repaired.mesh);
  }
  const dir = join(productsDir(work), 'repair');
  await mkdir(dir, { recursive: true });
  await writeJson(join(dir, 'mesh-quality.json'), result);
  await recordArtifact(work, 'analysis', { sources: [original, ...(result.repaired ? [repairedModelPath(work)] : [])] });
  const a = result.original;
  return {
    summary: `Mesh analysis: ${a.triangles.toLocaleString('en-US')} triangles, ${a.connectedComponents} component(s), `
      + `${a.boundaryLoops} boundary loop(s), ${a.nonManifoldEdges.toLocaleString('en-US')} non-manifold edge(s)`,
  };
}

export async function runOperation(operation, work, settings = {}, onProgress = () => {}) {
  if (operation === 'analyze') return runAnalysis(work, settings, onProgress);
  if (operation === 'confidence') {
    const report = await runConfidence(work, settings, onProgress);
    const s = report.summary;
    return {
      summary: `Confidence: mean ${s.meanScore ?? 'unavailable'} over ${s.evidenceCoveragePercent}% of the surface, `
        + `${s.unknownPercent}% unknown, ${report.regionCount} low-confidence region(s)`,
    };
  }
  if (operation === 'repair') {
    const report = await runRepair(work, settings, (f, m) => onProgress(f * 0.95, m));
    await attributeRepaired(work);
    onProgress(1);
    return {
      summary: `Repair: ${report.originalTriangles.toLocaleString('en-US')} to ${report.repairedTriangles.toLocaleString('en-US')} triangles, `
        + `${report.removedComponents} floating component(s) removed, ${report.filledHoles} hole(s) filled, `
        + `${report.estimatedGeometryPercent}% estimated surface`,
    };
  }
  if (operation === 'lod') {
    const manifest = await runLod(work, settings, onProgress);
    return {
      summary: `LOD: ${manifest.levels.length} level(s) [${manifest.levels.map((l) => l.triangleCount.toLocaleString('en-US')).join(' / ')} triangles]`
        + (manifest.skipped.length ? `, ${manifest.skipped.length} skipped` : ''),
    };
  }
  if (operation === 'completion') {
    const report = await runCompletion(work, settings, onProgress);
    return {
      summary: `Unseen sides: removed ${report.removedTriangles.toLocaleString('en-US')} broken triangles, `
        + `estimated ${report.estimatedWallM2.toLocaleString('en-US')} m2 of wall on ${report.buildingsFilled} structure(s) `
        + `and ${report.estimatedGroundM2.toLocaleString('en-US')} m2 of ground`,
    };
  }
  throw new Error(`Unknown operation: ${operation}`);
}
