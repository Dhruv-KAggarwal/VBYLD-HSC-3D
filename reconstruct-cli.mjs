import { mkdir, copyFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { basename, resolve, join } from 'node:path';
import { randomUUID } from 'node:crypto';
import {
  commands, reconstruct, exportExistingJob, normalizeFrames, normalizeQuality, normalizeThreads,
} from './lib/reconstruct.mjs';
import { assessQuality, hashFile, probeMedia, safeExtension } from './lib/multivideo/video-input.mjs';
import { normalizeVideoSettings, videoId, writeInputs } from './lib/multivideo/manifest.mjs';
import { normalizeEnhance } from './lib/api.mjs';
import { OPERATIONS, runOperation } from './lib/enhance/index.mjs';

const args = process.argv.slice(2);
const option = (name) => args.find((arg) => arg.startsWith(`--${name}=`))?.slice(name.length + 3);
const quality = normalizeQuality(
  args.includes('--detailed') || args.includes('--studio') ? 'detailed'
    : args.includes('--balanced') ? 'balanced'
      : 'fast',
);
const altitude = Number(option('altitude') || 82);
const frames = normalizeFrames(option('frames'));
const threads = option('threads') ? normalizeThreads(option('threads')) : null;
const reuse = option('reuse');
const full = args.includes('--full');
const inputs = args.filter((arg) => !arg.startsWith('--'));
const tool = option('tool');

if (!inputs.length && !reuse) {
  console.error('Usage: node reconstruct-cli.mjs <video.mp4> [more videos…] [--fast|--balanced|--detailed]');
  console.error('         [--frames=64] [--threads=8] [--altitude=82] [--repair] [--confidence] [--lod] [--fill]');
  console.error('   or: node reconstruct-cli.mjs --reuse=runtime/<job-id> [--view|--full] [--altitude=82]');
  console.error(`   or: node reconstruct-cli.mjs --tool=${OPERATIONS.join('|')} --reuse=runtime/<job-id> [--settings=<json>]`);
  process.exit(1);
}

// An enhancement against a finished job; no engine is re-run.
if (tool) {
  if (!OPERATIONS.includes(tool) || !reuse) {
    console.error(`--tool must be one of ${OPERATIONS.join(', ')} and needs --reuse=runtime/<job-id>.`);
    process.exit(1);
  }
  try {
    const settings = option('settings') ? JSON.parse(option('settings')) : {};
    const result = await runOperation(tool, resolve(reuse), settings, (fraction, message) => {
      if (message) console.log(`${String(Math.round(fraction * 100)).padStart(3)}%  ${message}`);
    });
    console.log(result.summary);
    process.exit(0);
  } catch (error) {
    console.error(`Failed: ${error.message}`);
    process.exit(1);
  }
}

if (reuse) {
  const result = await exportExistingJob(resolve(reuse), altitude, basename(inputs[0] || reuse), full ? 'full' : 'view');
  console.log(JSON.stringify(result.metadata, null, 2));
  console.log(`Wrote products to ${result.files.glb}`);
  process.exit(0);
}

for (const input of inputs) {
  if (!existsSync(input)) {
    console.error(`Video not found: ${input}`);
    process.exit(1);
  }
}

const id = randomUUID();
const work = resolve('runtime', id);
await mkdir(work, { recursive: true });
const multi = inputs.length > 1;
const job = {
  id,
  work,
  fileName: multi ? `${inputs.length} videos` : basename(inputs[0]),
  quality,
  frames,
  threads,
  altitude,
  enhance: normalizeEnhance({
    analysis: args.includes('--repair') || args.includes('--analyze'),
    repair: args.includes('--repair'),
    completion: args.includes('--fill'),
    confidence: args.includes('--confidence'),
    lod: args.includes('--lod'),
  }),
  progress: 0,
  stage: 'Queued',
  status: 'processing',
  cancelHooks: new Set(),
};

if (multi) {
  await mkdir(join(work, 'inputs'), { recursive: true });
  const videos = [];
  const seen = new Set();
  for (const input of inputs) {
    const vid = videoId(videos.length + 1);
    const file = `inputs/${vid}${safeExtension(input)}`;
    const probe = await probeMedia(existsSync(commands.ffprobe) ? commands.ffprobe : 'ffprobe', input);
    if (!probe.ok) {
      console.error(`${input}: ${probe.error}`);
      process.exit(1);
    }
    const sha256 = await hashFile(input);
    if (seen.has(sha256)) {
      console.warn(`Skipping ${input}: identical to a video already listed.`);
      continue;
    }
    seen.add(sha256);
    await copyFile(input, join(work, file));
    videos.push({
      id: vid,
      order: videos.length,
      originalName: basename(input),
      file,
      bytes: (await stat(input)).size,
      sha256,
      status: 'ready',
      probe,
      quality: await assessQuality(commands.ffmpeg, input, probe),
      settings: normalizeVideoSettings(),
    });
  }
  await writeInputs(work, videos);
  job.mode = 'multi';
  job.maxFrames = Number(option('frames')) || null;
  job.matching = args.includes('--exhaustive') ? 'exhaustive' : 'auto';
} else {
  const dest = join(work, `input${safeExtension(inputs[0])}`);
  await copyFile(inputs[0], dest);
  job.input = dest;
}

const timer = setInterval(() => {
  process.stdout.write(`\r${String(job.progress).padStart(3, ' ')}%  ${job.stage || ''}`.padEnd(80, ' '));
}, 800);
try {
  await reconstruct(job);
  clearInterval(timer);
  console.log(`\nComplete: ${job.result.model}`);
  console.log(JSON.stringify(job.metrics, null, 2));
  process.exit(0);
} catch (error) {
  clearInterval(timer);
  console.error(`\nFailed: ${error.message}`);
  process.exit(1);
}
