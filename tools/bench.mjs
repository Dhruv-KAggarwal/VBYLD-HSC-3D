/**
 * Stage-timing benchmark.
 * ---------------------------------------------------------------------------
 * Two modes, because the useful comparison is usually against a run that has
 * already happened:
 *
 *   node tools/bench.mjs --from-logs
 *       Read every pipeline.log under runtime/ and print one row per run with
 *       its per-stage wall clock. This is how to get a "before" column without
 *       re-running anything, and it works on logs from earlier builds.
 *
 *   node tools/bench.mjs <video.mp4> --profile=balanced
 *       Run an actual reconstruction and print its stage table. Takes as long
 *       as the pipeline takes; nothing here is a shortcut.
 *
 *   node tools/bench.mjs --from-logs --calibrate
 *       Also print a PIPELINE_STAGES referenceSeconds block derived from the
 *       median of the runs found, ready to paste into lib/reconstruct.mjs.
 *       referenceSeconds only drives the initial ETA — the tracker corrects
 *       itself from elapsed time once a run is underway — but a bad starting
 *       estimate is why a progress bar can sit at 90% for twenty minutes.
 *
 * Quality is checked separately: run tools/validate-glb.mjs on the resulting
 * model.glb. A faster run that produced a worse mesh is not an improvement,
 * and timing alone cannot tell you which you got.
 */

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { PIPELINE_STAGES, reconstruct } from '../lib/reconstruct.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback = null) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.split('=').slice(1).join('=');
};
const has = (name) => args.includes(`--${name}`);

const STAGE_KEYS = PIPELINE_STAGES.map((stage) => stage.key);
const STAGE_LABEL = new Map(PIPELINE_STAGES.map((stage) => [stage.key, stage.label]));

/** Strip the ANSI colour the logger writes into its file as well as the tty. */
const stripAnsi = (text) => text.replace(/\u001b\[[0-9;]*m/g, '');

/**
 * Pull stage durations out of a pipeline.log.
 *
 * The logger writes a line per stage transition carrying an elapsed offset:
 *
 *   [21:46:28 +3m51s] STAGE [2/9] Detecting GPU features — ALIKED keypoints
 *
 * Differencing those offsets gives each stage's wall clock, which is the only
 * timing available for a run that happened before this script existed. The
 * offset is right-padded (`+   3s`) and the whole line is ANSI-coloured, hence
 * the strip and the tolerant whitespace.
 */
function timingsFromLog(logPath) {
  let text;
  try { text = stripAnsi(readFileSync(logPath, 'utf8')); } catch { return null; }
  const marks = [];
  const line = /\+\s*(?:(\d+)h)?(?:(\d+)m)?(\d+)s\]\s+STAGE\s+\[(\d+)\/(\d+)\]\s+(.*)/g;
  let match = line.exec(text);
  while (match) {
    const seconds = Number(match[1] || 0) * 3600 + Number(match[2] || 0) * 60 + Number(match[3] || 0);
    marks.push({ at: seconds, index: Number(match[4]) - 1, label: match[6].trim() });
    match = line.exec(text);
  }
  if (marks.length < 2) return null;

  // The final elapsed offset anywhere in the file closes the last stage.
  let last = marks[marks.length - 1].at;
  const tail = /\+\s*(?:(\d+)h)?(?:(\d+)m)?(\d+)s\]/g;
  let t = tail.exec(text);
  while (t) {
    last = Math.max(last, Number(t[1] || 0) * 3600 + Number(t[2] || 0) * 60 + Number(t[3] || 0));
    t = tail.exec(text);
  }

  // The run header states both, e.g. `profile: detailed (168 frames @ 2.500 fps`.
  const header = /profile:\s*([a-z]+)\s*\((\d+)\s+frames/i.exec(text);

  const stages = {};
  for (let i = 0; i < marks.length; i += 1) {
    const end = i + 1 < marks.length ? marks[i + 1].at : last;
    const key = STAGE_KEYS[marks[i].index] || `stage${marks[i].index}`;
    // A stage can be entered more than once (a fallback path re-enters the
    // mapper); charge it the sum.
    stages[key] = (stages[key] || 0) + Math.max(0, end - marks[i].at);
  }
  return {
    total: last,
    stages,
    profile: header?.[1] || null,
    frames: Number(header?.[2] || 0),
  };
}

/** Everything useful that can be scraped from one job directory. */
function readRun(dir) {
  const log = [join(dir, 'pipeline.log'), join(dir, 'logs', 'pipeline.log')].find((p) => existsSync(p));
  if (!log) return null;
  const timings = timingsFromLog(log);
  if (!timings) return null;
  let metrics = {};
  for (const candidate of [join(dir, 'products', 'metadata.json'), join(dir, 'metadata.json')]) {
    if (!existsSync(candidate)) continue;
    try { metrics = JSON.parse(readFileSync(candidate, 'utf8')); } catch { /* keep going */ }
    break;
  }
  return {
    id: dir.split(/[/\\]/).pop(),
    at: statSync(log).mtime,
    // metadata.json is written by the products stage, so a run that failed
    // earlier has none. Both of these are also stated in the log, which every
    // run has, so fall back to scraping it rather than printing "?".
    profile: metrics.profile || timings.profile || '?',
    frames: metrics.framesUsed || metrics.frameCount || timings.frames || 0,
    faces: metrics.meshFaces || 0,
    points: metrics.points || 0,
    ...timings,
  };
}

/** Thousands separators, independent of the machine's locale. */
const count = (value) => (value ? Number(value).toLocaleString('en-US') : '-');

const fmt = (seconds) => {
  if (!seconds) return '-';
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m${String(seconds % 60).padStart(2, '0')}s`;
};

/** Print the stage table, one column per run. */
function printTable(runs) {
  if (!runs.length) {
    console.log('No runs with parseable stage timings were found.');
    return;
  }
  const head = ['stage', ...runs.map((r) => `${r.profile}/${r.frames}f`)];
  const widths = head.map((h) => h.length);
  const rows = [];
  for (const key of STAGE_KEYS) {
    const cells = runs.map((r) => {
      const seconds = r.stages[key] || 0;
      const share = r.total ? Math.round((100 * seconds) / r.total) : 0;
      return seconds ? `${fmt(seconds)} (${share}%)` : '-';
    });
    if (cells.every((c) => c === '-')) continue;
    rows.push([STAGE_LABEL.get(key) || key, ...cells]);
  }
  rows.push(['TOTAL', ...runs.map((r) => fmt(r.total))]);
  rows.push(['mesh faces', ...runs.map((r) => count(r.faces))]);
  rows.push(['dense points', ...runs.map((r) => count(r.points))]);

  for (const row of [head, ...rows]) {
    row.forEach((cell, i) => { widths[i] = Math.max(widths[i] || 0, String(cell).length); });
  }
  const line = (row) => row.map((cell, i) => String(cell).padEnd(widths[i])).join('  ');
  console.log(line(head));
  console.log(widths.map((w) => '-'.repeat(w)).join('  '));
  for (const row of rows) console.log(line(row));
}

/** Median per-stage seconds, normalised to the reference frame count. */
function calibrate(runs) {
  const REFERENCE_FRAMES = 90;
  console.log('\nSuggested PIPELINE_STAGES referenceSeconds (median, scaled to 90 frames):\n');
  for (const key of STAGE_KEYS) {
    const samples = runs
      .filter((r) => r.stages[key] && r.frames)
      .map((r) => (key === 'extract' || key === 'turntable'
        // These two do not scale with frame count; see buildStagePlan.
        ? r.stages[key]
        : (r.stages[key] * REFERENCE_FRAMES) / r.frames));
    if (!samples.length) continue;
    samples.sort((a, b) => a - b);
    const median = samples[Math.floor(samples.length / 2)];
    const current = PIPELINE_STAGES.find((s) => s.key === key)?.referenceSeconds;
    const delta = current ? ` (currently ${current})` : '';
    console.log(`  ${key.padEnd(10)} referenceSeconds: ${Math.round(median)},${delta}  [n=${samples.length}]`);
  }
  console.log('\nThese only set the OPENING estimate; the tracker re-derives ETA from');
  console.log('elapsed time as a run proceeds. Paste them into lib/reconstruct.mjs.');
}

if (has('from-logs') || !args.some((a) => !a.startsWith('--'))) {
  const runtimeDir = resolve(flag('runtime', 'runtime'));
  if (!existsSync(runtimeDir)) {
    console.error(`No runtime directory at ${runtimeDir}. Pass --runtime=<dir> or a video to benchmark.`);
    process.exit(1);
  }
  const limit = Number(flag('limit', 8));
  const only = flag('profile', null);
  const runs = readdirSync(runtimeDir)
    .map((name) => join(runtimeDir, name))
    .filter((dir) => { try { return statSync(dir).isDirectory(); } catch { return false; } })
    .map(readRun)
    .filter(Boolean)
    .filter((run) => !only || run.profile === only)
    .sort((a, b) => b.at - a.at)
    .slice(0, limit)
    .reverse();
  console.log(`Stage timings from ${runs.length} run(s) under ${runtimeDir}, oldest first\n`);
  printTable(runs);
  if (has('calibrate')) calibrate(runs);
} else {
  const video = resolve(args.find((a) => !a.startsWith('--')));
  if (!existsSync(video)) {
    console.error(`No such video: ${video}`);
    process.exit(1);
  }
  const profile = flag('profile', 'balanced');
  console.log(`Benchmarking ${profile} on ${video}\nThis runs the real pipeline; expect it to take as long as a normal job.\n`);
  const job = {
    id: `bench-${Date.now()}`,
    status: 'queued',
    quality: profile,
    altitude: Number(flag('altitude', 80)),
    frames: flag('frames', null),
    fileName: video.split(/[/\\]/).pop(),
    wantRefine: has('refine'),
    wantSplat: false,
    wantDsm: false,
  };
  const started = Date.now();
  await reconstruct(job, video);
  const seconds = Math.round((Date.now() - started) / 1000);
  const stages = {};
  for (const stage of job.stages || []) {
    if (stage.durationMs != null) stages[stage.key] = Math.round(stage.durationMs / 1000);
  }
  printTable([{
    id: job.id,
    profile,
    frames: job.metrics?.framesUsed || 0,
    faces: job.metrics?.meshFaces || 0,
    points: job.metrics?.points || 0,
    total: seconds,
    stages,
    at: new Date(),
  }]);
  console.log(`\nValidate the output before trusting the speed-up:\n  node tools/validate-glb.mjs ${join('runtime', job.id, 'products', 'model.glb')}`);
}
