/**
 * Stand the model of a finished job upright again, without re-running any engine.
 *
 *   node tools/relevel.mjs <job-id> [<job-id> ...]
 *   node tools/relevel.mjs --all
 *   node tools/relevel.mjs --all --check     report the tilt, change nothing
 *
 * The vertical and the metric scale are measured again, the model and point
 * cloud are moved into the new frame, and every product derived from them
 * (confidence, repair, filled sides, levels of detail, turntable) is rebuilt.
 * The replaced files are copied to products/before-level first. To undo, copy
 * them back (job.json goes in the job folder) and run the tools from the app.
 */

import { existsSync, statSync } from 'node:fs';
import { copyFile, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { originalModelPath, readRegistry, setActiveModel } from '../lib/artifacts.mjs';
import { denseCloudPath } from '../lib/enhance/context.mjs';
import { runOperation } from '../lib/enhance/index.mjs';
import { computeSceneTransform, writeCloudProducts } from '../lib/export3d.mjs';
import { applyMatrix, readMeshGlb, writeMeshGlb } from '../lib/glbmesh.mjs';
import { _internals } from '../lib/glbtransform.mjs';
import { inputsPath, readJson, writeJson } from '../lib/multivideo/manifest.mjs';
import { renderTurntable, turntableSettings } from '../lib/turntable.mjs';
import { jobViews } from '../lib/views.mjs';

const ROOT = resolve('runtime');
const BACKUP = ['model.glb', 'pointcloud.ply', 'transform.json', 'metadata.json', 'views.json', 'turning.mp4', 'turning.png'];

const degreesBetween = (a, b) => (Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180) / Math.PI;

/** Column-major matrix taking a point of the old exported frame to the new one. */
function frameChange(from, to) {
  // new = s1 R1 (R0' old / s0 + o0 - o1)
  const turn = [0, 1, 2].map((r) => [0, 1, 2].map((c) => (
    (to.scale / from.scale) * [0, 1, 2].reduce((sum, k) => sum + to.rotation[r][k] * from.rotation[c][k], 0))));
  const shift = [0, 1, 2].map((r) => to.scale * [0, 1, 2].reduce((sum, k) => sum + to.rotation[r][k] * (from.origin[k] - to.origin[k]), 0));
  return [
    turn[0][0], turn[1][0], turn[2][0], 0,
    turn[0][1], turn[1][1], turn[2][1], 0,
    turn[0][2], turn[1][2], turn[2][2], 0,
    shift[0], shift[1], shift[2], 1,
  ];
}

async function moveModel(path, from, to) {
  const read = readMeshGlb(await readFile(path));
  if (read.rootMatrix) {
    // Vertices are in the reconstruction frame; only the alignment matrix changes.
    const { json, bin } = _internals.parseGlb(await readFile(path));
    const node = json.nodes.find((n) => n.name === 'SkyforgeAlignment');
    node.matrix = _internals.affineColumnMajor(to.rotation, to.scale, to.origin);
    await writeFile(path, _internals.serializeGlb(json, bin));
    return;
  }
  const matrix = frameChange(from, to);
  for (const primitive of read.primitives) {
    applyMatrix(primitive.positions, matrix, 1);
    primitive.normals = null;
  }
  await writeFile(path, writeMeshGlb({
    primitives: read.primitives, images: read.images, unlit: read.unlit, generator: 'Skyforge levelled',
  }));
}

async function relevel(id, checkOnly) {
  const work = join(ROOT, id);
  const products = join(work, 'products');
  const old = await readJson(join(products, 'transform.json'));
  const model = originalModelPath(work);
  const cloudPath = denseCloudPath(work);
  if (!old?.rotation || !model?.endsWith('model.glb') || !cloudPath) {
    console.log(`${id}: skipped (no 3D model, dense cloud or stored alignment)`);
    return;
  }
  const metadata = await readJson(join(products, 'metadata.json'));
  const inputs = await readJson(inputsPath(work));
  const videos = inputs?.videos?.length
    ? inputs.videos.filter((v) => v.settings?.enabled !== false)
      .map((v) => ({ id: v.id, label: v.settings?.label || v.originalName || v.id, altitudeM: v.settings?.altitudeM ?? null }))
    : null;
  const next = await computeSceneTransform({
    cloudPath, meshPath: cloudPath, workDir: work, altitude: metadata?.altitudeM || 82, videos,
  });
  const tilt = degreesBetween(old.rotation[1], next.rotation[1]);
  const resize = next.scale / old.scale;
  console.log(`${id}: leans ${tilt.toFixed(1)} degrees; lengths change by ${((resize - 1) * 100).toFixed(1)}% (up from ${next.orientation})`);
  if (checkOnly) return;
  if (tilt < 0.5 && Math.abs(resize - 1) < 0.005) { console.log('  already level'); return; }

  const backup = join(products, 'before-level');
  if (existsSync(backup)) throw new Error(`${backup} already exists; move it away before levelling this job again.`);
  await mkdir(backup);
  for (const name of BACKUP) {
    if (existsSync(join(products, name))) await copyFile(join(products, name), join(backup, name));
  }
  if (existsSync(join(work, 'job.json'))) await copyFile(join(work, 'job.json'), join(backup, 'job.json'));

  await moveModel(model, old, next);
  await writeJson(join(products, 'transform.json'), {
    description: 'world = scale * rotation * (reconstruction - origin); Y up, metres', ...next,
  });
  const cloud = await writeCloudProducts({
    cloudPath,
    outDir: products,
    transform: next,
    altitude: metadata?.altitudeM || 82,
    sourceName: metadata?.source,
    maxPoints: metadata?.exportedPoints || 400000,
  });
  await rm(join(products, 'views.json'), { force: true });
  const record = await readJson(join(work, 'job.json'));
  if (record?.metrics) {
    for (const key of Object.keys(cloud.metadata)) {
      if (key in record.metrics && key !== 'generatedAt') record.metrics[key] = cloud.metadata[key];
    }
    if (next.altitudeReport) record.altitudeReport = next.altitudeReport;
    await writeJson(join(work, 'job.json'), record);
  }

  // Rebuild what the job had, with the settings it was built with.
  const registry = await readRegistry(work);
  for (const [kind, operation] of [['analysis', 'analyze'], ['confidence', 'confidence'], ['repair', 'repair'], ['completion', 'completion'], ['lod', 'lod']]) {
    const entry = registry.entries[kind];
    if (!entry) continue;
    // Filled sides and levels follow whichever model the operator had active.
    if (kind === 'completion') await setActiveModel(work, registry.activeModel);
    process.stdout.write(`  rebuilding ${kind}... `);
    try {
      const result = await runOperation(operation, work, entry.settings || {});
      console.log(result.summary);
    } catch (error) {
      console.log(`failed: ${error.message}`);
    }
  }
  await setActiveModel(work, registry.activeModel);

  const filled = join(products, 'model-completed.glb');
  if (existsSync(join(backup, 'turning.mp4'))) {
    process.stdout.write('  rendering the turntable... ');
    const views = await jobViews(work).catch(() => null);
    const outcome = await renderTurntable({
      focus: views?.focus,
      glbPath: existsSync(filled) && statSync(filled).size ? filled : model,
      outPath: join(products, 'turning.mp4'),
      ffmpeg: resolve('engines/ffmpeg/bin/ffmpeg.exe'),
      elevation: 30,
      ...turntableSettings(record?.quality || 'balanced'),
    });
    console.log(outcome.ok ? 'done' : `failed: ${outcome.reason}`);
  }
}

const args = process.argv.slice(2);
const ids = args.includes('--all')
  ? (await readdir(ROOT)).filter((name) => existsSync(join(ROOT, name, 'products')))
  : args.filter((a) => !a.startsWith('--'));
if (!ids.length) {
  console.error('Usage: node tools/relevel.mjs <job-id> ... | --all [--check]');
  process.exit(1);
}
for (const id of ids) await relevel(id, args.includes('--check'));
