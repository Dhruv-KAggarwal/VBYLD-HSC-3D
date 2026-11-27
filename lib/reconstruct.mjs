import { spawn, spawnSync } from 'node:child_process';
import { closeSync, existsSync, openSync, readSync, readdirSync, statSync } from 'node:fs';
import { mkdir, readdir, writeFile, readFile, copyFile, rm } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import { join, resolve } from 'node:path';
import { buildProducts, computeSceneTransform, writeCloudProducts } from './export3d.mjs';
import { classifyHardware, hardwareFromTelemetry } from './hardware.mjs';
import { sampleTelemetry } from './telemetry.mjs';
import { openmvsCommands, openmvsAvailable, buildTexturedMesh, isCancellation } from './openmvs.mjs';
import { reorientGlb, embedExternalImages } from './glbtransform.mjs';
import { buildColoredMeshGlb } from './meshcolor.mjs';
import { parsePly } from './ply.mjs';
import { createLogger } from './logger.mjs';
import { buildReport, reportToMarkdown } from './report.mjs';
import { detectGlomap, detectHloc, detectMasking, detectSplat, detectAll } from './external.mjs';
import { selectSharpFrames } from './preprocess.mjs';
import { renderTurntable, turntablePath, turntableSettings } from './turntable.mjs';

export const commands = {
  ffmpeg: process.env.FFMPEG_PATH || resolve('engines/ffmpeg/bin/ffmpeg.exe'),
  ffprobe: process.env.FFPROBE_PATH || resolve('engines/ffmpeg/bin/ffprobe.exe'),
  colmap: process.env.COLMAP_PATH || resolve('engines/colmap/bin/colmap.exe'),
  openmvs: process.env.OPENMVS_PATH || resolve('engines/openmvs'),
};

/**
 * cuDNN 9 runtime for COLMAP's learned features.
 * ---------------------------------------------------------------------------
 * COLMAP 4.2 can run ALIKED detection and LightGlue matching through ONNX
 * Runtime on the GPU, which is markedly more robust than SIFT on the repetitive
 * facades and large flat roofs a drone pass is full of. But
 * `onnxruntime_providers_cuda.dll` links against `cudnn64_9.dll`, and the CUDA
 * Toolkit ships cuDNN 8 — so without this the provider fails to load, ONNX
 * falls back to CPU, and the learned path is slower than SIFT rather than
 * better.
 *
 * `npm run setup:cudnn` drops the v9 runtime in engines/cudnn. We prepend it to
 * PATH for child processes only, so nothing leaks into the parent environment
 * or shadows a system CUDA install for anything else on the machine.
 */
const CUDNN_ROOT = process.env.CUDNN_PATH || resolve('engines/cudnn/nvidia');
const CUDNN_DIRS = ['cudnn/bin', 'cublas/bin', 'cuda_nvrtc/bin'].map((part) => join(CUDNN_ROOT, part));

/** True when the v9 runtime COLMAP needs is actually present. */
export function cudnnAvailable() {
  return existsSync(join(CUDNN_ROOT, 'cudnn/bin/cudnn64_9.dll'));
}

/** Child-process environment with the cuDNN runtime prepended to PATH. */
export function engineEnv() {
  if (!cudnnAvailable()) return process.env;
  const existing = process.env.PATH || process.env.Path || '';
  return { ...process.env, PATH: `${CUDNN_DIRS.join(';')};${existing}` };
}

/** Every engine process currently running, so none can outlive this server. */
const activeChildren = new Set();

/**
 * Terminates an engine and everything it spawned.
 *
 * `child.kill()` is not enough on Windows: it signals only the direct child, so
 * a COLMAP run detaches and keeps holding the GPU after its owner has gone.
 * Orphans like that starve the next job and silently corrupt a shared dense
 * workspace, so the whole tree has to go.
 */
export function killTree(child) {
  if (!child || child.killed || child.exitCode !== null) return;
  try {
    if (process.platform === 'win32') {
      spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore', timeout: 5000 });
    } else {
      process.kill(-child.pid, 'SIGKILL');
    }
  } catch {
    try { child.kill('SIGKILL'); } catch { /* already gone */ }
  }
}

export function terminateAllEngines() {
  for (const child of activeChildren) killTree(child);
  activeChildren.clear();
}

/**
 * Recognises the sub-progress each engine prints.
 *
 * Without this the bar sits frozen for the eight minutes that dense stereo
 * takes. Formats are taken from real `engine.log` output rather than guessed:
 *
 *   ffmpeg              frame=   90 fps=0.0 q=2.0 ... speed=27.4x
 *   feature_extractor   feature_extraction.cc:270] Processed file [77/90]
 *   sequential_matcher  pairing.cc:560] Processing image [35/90]
 *   mapper              incremental_pipeline.cc:620] Registering image #56 (num_reg_frames=55)
 *   patch_match_stereo  patch_match.cc:419] === Processing view 89 / 90 for frame_000089.jpg ===
 *   stereo_fusion       fusion.cc:283] Fusing image [43/90] with index 45
 */
export function parseEngineLine(line) {
  const text = String(line);
  let match = text.match(/=== Processing view (\d+) \/ (\d+)/);
  if (match) return { kind: 'view', done: Number(match[1]), total: Number(match[2]) };

  match = text.match(/Fusing image \[(\d+)\/(\d+)\]/);
  if (match) return { kind: 'fuse', done: Number(match[1]), total: Number(match[2]) };

  match = text.match(/Processed file \[(\d+)\/(\d+)\]/);
  if (match) return { kind: 'file', done: Number(match[1]), total: Number(match[2]) };

  match = text.match(/Processing image \[(\d+)\/(\d+)\]/);
  if (match) return { kind: 'image', done: Number(match[1]), total: Number(match[2]) };

  match = text.match(/num_reg_frames=(\d+)/);
  if (match) return { kind: 'register', done: Number(match[1]) + 1, total: null };

  match = text.match(/Registering image #(\d+)/);
  if (match) return { kind: 'register', done: Number(match[1]), total: null };

  match = text.match(/^frame=\s*(\d+)/m) || text.match(/\bframe=\s*(\d+)/);
  if (match) return { kind: 'frame', done: Number(match[1]), total: null };

  // OpenMVS depth-map / fusion / texturing progress, e.g.
  //   "Estimated depth-map 12 (of 90)"  /  "image 45 of 90"  /  "...processed 30/90"
  match = text.match(/\b(?:image|view|depth-?maps?)\D*(\d+)\s*(?:of|\/)\s*(\d+)/i);
  if (match) return { kind: 'view', done: Number(match[1]), total: Number(match[2]) };

  // OpenMVS prints a percentage for the long reconstruction / texturing passes.
  match = text.match(/(?:^|[^\d])(\d{1,3})%/);
  if (match) {
    const pct = Math.min(100, Number(match[1]));
    return { kind: 'percent', done: pct, total: 100 };
  }

  return null;
}

/**
 * Turns a stream of engine lines into a 0..1 fraction for one stage.
 *
 * `passes` matters: with `geom_consistency 1` patch_match_stereo walks every
 * view twice, so counting a single 1..N sweep as "done" would park the bar at
 * 100% for half the stage. A view index that goes backwards means a new pass.
 */
export function createLineProgress({ total = null, passes = 1 } = {}) {
  let knownTotal = total && total > 0 ? total : null;
  let pass = 0;
  let previous = 0;
  let best = 0;
  return {
    get fraction() { return best; },
    feed(line) {
      const parsed = parseEngineLine(line);
      if (!parsed) return null;
      if (parsed.total && parsed.total > 0) knownTotal = parsed.total;
      if (!knownTotal) return null;
      const done = Math.min(parsed.done, knownTotal);
      if (done < previous) pass = Math.min(pass + 1, passes - 1);
      previous = done;
      const fraction = (pass + done / knownTotal) / passes;
      best = Math.min(1, Math.max(best, fraction));
      return { ...parsed, total: knownTotal, pass: pass + 1, passes, fraction: best };
    },
  };
}

export function run(command, args, cwd, job, label, options = {}) {
  return new Promise((resolveRun, rejectRun) => {
    const tracker = job.tracker || null;
    // The tracker owns `job.stage` when there is one; the CLI has no tracker.
    if (tracker) tracker.start(options.stage, label);
    else job.stage = label;
    const lineProgress = createLineProgress({ total: options.total, passes: options.passes || 1 });
    let pending = '';
    const child = spawn(command, args, {
      cwd,
      windowsHide: true,
      detached: process.platform !== 'win32',
      // Engines inherit the cuDNN 9 runtime so ONNX can use the GPU; the
      // parent environment is left untouched.
      env: engineEnv(),
    });
    activeChildren.add(child);
    let stderr = '';
    let settled = false;
    // Heartbeat state: some engine steps (COLMAP global bundle adjustment, OpenMVS
    // depth-map fusion) run many minutes without printing a single line, which
    // reads as a hang. We track the last output time + a little context and emit a
    // periodic "still working" line so the terminal shows the job is alive.
    let heartbeat = null;
    let lastLineAt = Date.now();
    let heartbeatContext = '';
    let regFrames = 0;
    const clearHeartbeat = () => { if (heartbeat) { clearInterval(heartbeat); heartbeat = null; } };
    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearHeartbeat();
      activeChildren.delete(child);
      if (error) rejectRun(error);
      else resolveRun();
    };

    // A cancelled or timed-out job must take its engine down with it rather
    // than leaving it to compete with whatever runs next.
    //
    // The two are NOT equivalent downstream, though: see `isCancellation` below.
    const onCancel = () => {
      killTree(child);
      job.cancelHooks?.delete(onCancel);
      finish(new Error(`${label} was cancelled.`));
    };
    job.cancelHooks?.add(onCancel);

    // Dense depth-map estimation and mesh refinement legitimately run for many
    // minutes, so the caller can lift the ceiling per stage; everything else
    // keeps the original COLMAP/other defaults.
    const timeoutMs = options.timeoutMs
      || (command.toLowerCase().includes('colmap') ? 50 * 60 * 1000 : 30 * 60 * 1000);
    const timeout = setTimeout(() => {
      killTree(child);
      job.cancelHooks?.delete(onCancel);
      finish(new Error(`${label} exceeded the processing timeout.`));
    }, timeoutMs);

    const logger = job.logger || null;
    const collect = (chunk) => {
      const text = chunk.toString();
      stderr += text;
      if (stderr.length > 1_500_000) stderr = stderr.slice(-800_000);
      // FFmpeg rewrites its status line with \r and never emits \n, so both
      // terminators have to end a line or the bar would never move during
      // frame extraction.
      pending += text;
      const parts = pending.split(/[\r\n]+/);
      pending = parts.pop() ?? '';
      if (pending.length > 4096) pending = pending.slice(-2048);
      for (const part of parts) {
        // Mirror every engine line into the per-job log file (terminal stays
        // clean; the file keeps the full forensic trail for post-mortems).
        if (logger && part.trim()) logger.engine(`${options.stage || label}| ${part.trim()}`);
        // Feed the heartbeat: remember the last meaningful context and the live
        // registered-frame count so a silent BA pass still reports progress.
        const trimmed = part.trim();
        if (trimmed) {
          lastLineAt = Date.now();
          const reg = trimmed.match(/num_reg_frames=(\d+)/);
          if (reg) regFrames = Number(reg[1]);
          if (/Global bundle adjustment|Retriangulation/i.test(trimmed)) heartbeatContext = 'global bundle adjustment';
          else if (/Bundle adjustment/i.test(trimmed)) heartbeatContext = 'bundle adjustment';
          else if (/Processing view|Estimating depth|Fusing|Filtering/i.test(trimmed)) heartbeatContext = trimmed.slice(0, 48).toLowerCase();
        }
        const update = lineProgress.feed(part);
        if (update && tracker) tracker.fraction(update.fraction, update);
      }
    };
    child.stderr?.on('data', collect);
    child.stdout?.on('data', collect);
    // Watchdog: if the engine has printed nothing for a while during a long step,
    // reassure the terminal that it is still alive (and roughly where it is).
    if (logger) {
      heartbeat = setInterval(() => {
        const idleMs = Date.now() - lastLineAt;
        if (idleMs < 25000) return;
        const idleS = Math.round(idleMs / 1000);
        const ctx = heartbeatContext ? ` — ${heartbeatContext}` : '';
        const frames = regFrames ? `, ${regFrames} frames solved` : '';
        logger.info(`${label} still running${ctx} (${idleS}s since last engine output${frames}) …`);
      }, 20000);
      heartbeat.unref?.();
    }
    child.on('error', (error) => {
      clearTimeout(timeout);
      job.cancelHooks?.delete(onCancel);
      finish(new Error(`${command} is unavailable: ${error.message}`));
    });
    child.on('close', async (code) => {
      clearTimeout(timeout);
      job.cancelHooks?.delete(onCancel);
      try {
        await writeFile(join(cwd, 'engine.log'), `\n--- ${label} (${new Date().toISOString()}) ---\n${stderr.slice(-8000)}\n`, { flag: 'a' });
      } catch {
        // logging must never mask the real result
      }
      finish(code === 0 ? null : new Error(`${label} failed with code ${code}: ${stderr.slice(-1200)}`));
    });
  });
}

let gpuChain = Promise.resolve();

/**
 * Serialises GPU work. Two dense runs sharing one card thrash its memory and,
 * worse, interleave writes into the same depth-map directory, so a second job
 * can quietly poison the first one's output.
 */
export function queueExclusive(task) {
  const result = gpuChain.then(task, task);
  gpuChain = result.then(() => undefined, () => undefined);
  return result;
}

export async function engineAvailable(command) {
  if (command.includes('/') || command.includes('\\')) return existsSync(command);
  return new Promise((resolveAvailable) => {
    let settled = false;
    const done = (value) => {
      if (settled) return;
      settled = true;
      resolveAvailable(value);
    };
    const probe = spawn(command, ['-h'], { windowsHide: true, stdio: 'ignore' });
    const guard = setTimeout(() => {
      killTree(probe);
      done(false);
    }, 10_000);
    probe.on('error', () => {
      clearTimeout(guard);
      done(false);
    });
    probe.on('close', (code) => {
      clearTimeout(guard);
      done(code === 0 || code === 1);
    });
  });
}

const PROBE_FALLBACK = { duration: 10, frames: 240, width: 1280, height: 720, reliable: false };

/**
 * Reads the clip's real duration.
 *
 * The frame sampling rate is derived from this, so guessing here is dangerous:
 * silently assuming ten seconds for a ten-minute flight samples one frame per
 * minute and produces a reconstruction that cannot possibly solve. Failures are
 * reported through `reliable` so the caller can warn instead of pretending.
 */
function probeVideo(input) {
  return new Promise((resolveProbe) => {
    const probeBin = existsSync(commands.ffprobe) ? commands.ffprobe : 'ffprobe';
    const child = spawn(probeBin, ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames,avg_frame_rate,width,height', '-show_entries', 'format=duration', '-of', 'json', input], { windowsHide: true });
    let out = '';
    const guard = setTimeout(() => {
      killTree(child);
      resolveProbe({ ...PROBE_FALLBACK, error: 'ffprobe timed out' });
    }, 60_000);
    child.stdout.on('data', (chunk) => { out += chunk.toString(); });
    child.on('error', (error) => {
      clearTimeout(guard);
      resolveProbe({ ...PROBE_FALLBACK, error: error.message });
    });
    child.on('close', () => {
      clearTimeout(guard);
      try {
        const json = JSON.parse(out);
        const stream = json.streams?.[0] || {};
        const duration = Number(json.format?.duration);
        const rate = String(stream.avg_frame_rate || '24/1').split('/');
        const fps = Number(rate[0] || 24) / Number(rate[1] || 1);
        const frames = Number(stream.nb_frames) || Math.round(duration * fps);
        if (!Number.isFinite(duration) || duration <= 0) {
          resolveProbe({ ...PROBE_FALLBACK, error: 'no duration reported' });
          return;
        }
        resolveProbe({
          duration,
          frames: Number.isFinite(frames) && frames > 0 ? frames : Math.round(duration * 24),
          width: Number(stream.width) || 1280,
          height: Number(stream.height) || 720,
          reliable: true,
        });
      } catch (error) {
        resolveProbe({ ...PROBE_FALLBACK, error: error.message });
      }
    });
  });
}

/**
 * The mapper can split a flight into several disconnected models when tracking
 * breaks. Model `0` is not necessarily the good one, so pick whichever
 * registered the most images and report how much of the flight it covers.
 */
function registeredImageCount(dir) {
  const binary = join(dir, 'images.bin');
  if (!existsSync(binary)) return 0;
  try {
    const handle = openSync(binary, 'r');
    const header = Buffer.alloc(8);
    readSync(handle, header, 0, 8, 0);
    closeSync(handle);
    const count = Number(header.readBigUInt64LE(0));
    return Number.isFinite(count) && count >= 0 && count < 1e6 ? count : 0;
  } catch {
    return 0;
  }
}

function chooseSparseModel(sparse) {
  if (!existsSync(sparse)) return { path: sparse, images: 0, models: 0 };
  const candidates = [];
  for (const entry of readdirSync(sparse, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const dir = join(sparse, entry.name);
    if (!existsSync(join(dir, 'images.bin')) && !existsSync(join(dir, 'images.txt'))) continue;
    candidates.push({ path: dir, images: registeredImageCount(dir) });
  }
  if (!candidates.length) return { path: sparse, images: 0, models: 0 };
  candidates.sort((a, b) => b.images - a.images);
  return { path: candidates[0].path, images: candidates[0].images, models: candidates.length };
}

/**
 * The pipeline as a declared list of stages.
 *
 * `referenceSeconds` are measured, not guessed: they come from the timestamps
 * in `runtime/0c7f4613-.../engine.log`, a 10 s / 90-frame clip that ran the
 * current settings end to end (mapper 442 s, dense stereo 522 s, everything
 * else under a minute). Weights are derived from those numbers so the bar
 * spends its time where the pipeline actually spends its time.
 */
/**
 * Stage weights for the progress bar and ETA.
 *
 * `referenceSeconds` is a RELATIVE cost, not a promise: the tracker normalises
 * these into a weighted bar, so only their ratios matter. They were badly wrong
 * before — `texture: 80` against a measured 1301s meant the bar reached ~90%
 * and then sat there for twenty minutes, which is the single biggest reason the
 * UI felt hung.
 *
 * Current figures are anchored on a measured 96-frame `detailed` run
 * (RTX 4060 8GB / 15.7GB RAM): extract 3s, features 11s, match 37s, mapper
 * 179s, undistort 7s, densify 155s, mesh 36s, texture 1301s, products 149s.
 * They are then adjusted for the changes made since that run: NVDEC halves
 * extract, the source-width cap and the 4096 atlas ceiling cut texture hard,
 * global SfM replaces the incremental mapper, and the DSM is no longer built
 * on every job. Re-measure and update these whenever the stage costs move.
 */
export const PIPELINE_STAGES = [
  { key: 'extract', label: 'Extracting frames', detail: 'FFmpeg samples and sharpens the flight (NVDEC when available)', referenceSeconds: 15 },
  { key: 'features', label: 'Detecting GPU features', detail: 'ALIKED keypoints per frame on the GPU', referenceSeconds: 20 },
  { key: 'match', label: 'Matching sequential views', detail: 'LightGlue sequential matching with loop closure', referenceSeconds: 60 },
  { key: 'mapper', label: 'Solving camera poses', detail: 'Global SfM, with incremental as fallback', referenceSeconds: 120 },
  { key: 'undistort', label: 'Undistorting images', detail: 'Preparing the dense workspace', referenceSeconds: 10 },
  { key: 'densify', label: 'Dense depth maps on GPU', detail: 'OpenMVS CUDA depth-map estimation and fusion', referenceSeconds: 180 },
  { key: 'mesh', label: 'Reconstructing 3D mesh', detail: 'Delaunay + graph-cut surface, optional photo refinement', referenceSeconds: 90 },
  { key: 'texture', label: 'Baking texture atlas', detail: 'Per-face photographic texture baked into a GLB', referenceSeconds: 420 },
  { key: 'products', label: 'Building exports', detail: 'Oriented, metric GLB plus point cloud and report', referenceSeconds: 40 },
  { key: 'turntable', label: 'Rendering turntable video', detail: '360-degree MP4 flyaround of the finished model', referenceSeconds: 70 },
];

const REFERENCE_FRAMES = 90;

/**
 * Scales the reference timings to this clip and normalises them into weights.
 *
 * Frame extraction is the one stage that tracks clip length rather than frame
 * count: FFmpeg decodes the whole video no matter how few frames it keeps.
 * Everything after it is paid per frame.
 */
export function buildStagePlan({ durationSec = 10, frameCount = REFERENCE_FRAMES } = {}) {
  const frameScale = Math.max(0.2, (Number(frameCount) || REFERENCE_FRAMES) / REFERENCE_FRAMES);
  const seconds = PIPELINE_STAGES.map((stage) => {
    // Frame extraction tracks clip length, not frame count: FFmpeg decodes the
    // whole video no matter how few frames it keeps.
    if (stage.key === 'extract') return 2.5 + Math.max(0, Number(durationSec) || 0) * 0.12;
    // The turntable rasterises the finished mesh a fixed number of times. Its
    // cost tracks face count, which is only loosely tied to frame count, so
    // scaling it with the others would make the estimate worse, not better.
    if (stage.key === 'turntable') return stage.referenceSeconds;
    return stage.referenceSeconds * frameScale;
  });
  const total = seconds.reduce((sum, value) => sum + value, 0) || 1;
  return PIPELINE_STAGES.map((stage, index) => ({
    key: stage.key,
    label: stage.label,
    detail: stage.detail,
    weight: seconds[index] / total,
    expectedSeconds: seconds[index],
  }));
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/**
 * Tracks which stage is running and how far into it we are, and writes the
 * result onto the job so the API, the SSE stream and the CLI all see the same
 * numbers.
 *
 * ETA is pure measurement: elapsed time divided by the fraction of total weight
 * completed, minus the elapsed time. That self-corrects when the declared
 * weights are wrong for a particular clip, which they will sometimes be.
 */
export function createProgressTracker(job, plan = buildStagePlan(), now = () => Date.now(), logger = null) {
  const stages = plan.map((stage) => ({
    ...stage,
    status: 'pending',
    startedAt: null,
    endedAt: null,
    fraction: 0,
  }));
  const totalWeight = stages.reduce((sum, stage) => sum + stage.weight, 0) || 1;
  const startedAt = now();
  let index = -1;
  let highWater = 0;

  const overallFraction = () => {
    let done = 0;
    for (let i = 0; i < stages.length; i += 1) {
      if (i < index) done += stages[i].weight;
      else if (i === index) done += stages[i].weight * stages[i].fraction;
    }
    return clamp01(done / totalWeight);
  };

  const publish = (detail = null) => {
    const fraction = Math.max(highWater, overallFraction());
    highWater = fraction;
    const elapsedMs = now() - startedAt;
    // Below ~2% the sample is too short to extrapolate from and the estimate
    // swings by tens of minutes, so say nothing rather than something silly.
    const etaMs = fraction > 0.02 && fraction < 1 ? Math.round((elapsedMs * (1 - fraction)) / fraction) : null;
    const current = stages[index] || null;
    job.progress = Math.round(fraction * 1000) / 10;
    job.stage = current ? current.label : 'Queued';
    job.stageKey = current?.key || null;
    job.stageIndex = index;
    job.stageCount = stages.length;
    job.stageProgress = current ? Math.round(current.fraction * 1000) / 10 : 0;
    job.stageDetail = detail;
    job.startedAt = new Date(startedAt).toISOString();
    job.elapsedMs = elapsedMs;
    job.etaMs = etaMs;
    job.stages = stages.map((stage) => ({
      key: stage.key,
      label: stage.label,
      detail: stage.detail,
      weight: Math.round(stage.weight * 10000) / 10000,
      status: stage.status,
      progress: Math.round(stage.fraction * 1000) / 10,
      startedAt: stage.startedAt ? new Date(stage.startedAt).toISOString() : null,
      endedAt: stage.endedAt ? new Date(stage.endedAt).toISOString() : null,
      durationMs: stage.startedAt ? (stage.endedAt || now()) - stage.startedAt : null,
    }));
    return job;
  };

  const describe = (update) => {
    if (!update) return null;
    const passPart = update.passes > 1 ? ` · pass ${update.pass}/${update.passes}` : '';
    const nouns = { view: 'view', fuse: 'image', file: 'frame', image: 'frame', register: 'camera', frame: 'frame' };
    const noun = nouns[update.kind] || 'step';
    return `${noun} ${Math.min(update.done, update.total)}/${update.total}${passPart}`;
  };

  return {
    stages,
    /** Move to `key`, closing every stage before it. A null key only relabels. */
    start(key, label) {
      if (!key) {
        if (index >= 0 && label) stages[index].label = label;
        return publish();
      }
      const next = stages.findIndex((stage) => stage.key === key);
      if (next < 0) return publish();
      for (let i = 0; i <= next - 1; i += 1) {
        if (stages[i].status !== 'complete') {
          stages[i].status = 'complete';
          stages[i].fraction = 1;
          stages[i].startedAt = stages[i].startedAt || now();
          stages[i].endedAt = stages[i].endedAt || now();
        }
      }
      if (next !== index) {
        index = next;
        stages[next].startedAt = stages[next].startedAt || now();
        stages[next].fraction = 0;
        // A genuine stage transition is worth one timestamped line in the log.
        logger?.stage(next + 1, stages.length, label || stages[next].label, stages[next].detail);
      }
      stages[next].status = 'active';
      if (label) stages[next].label = label;
      return publish();
    },
    fraction(value, update = null) {
      if (index < 0) return job;
      stages[index].fraction = Math.max(stages[index].fraction, clamp01(value));
      return publish(describe(update));
    },
    complete(key) {
      const target = key ? stages.findIndex((stage) => stage.key === key) : index;
      if (target < 0) return publish();
      stages[target].fraction = 1;
      stages[target].status = 'complete';
      stages[target].startedAt = stages[target].startedAt || now();
      stages[target].endedAt = now();
      logger?.ok(`${stages[target].label} complete`, stages[target].endedAt - stages[target].startedAt);
      if (target === index && index < stages.length - 1) index = target;
      return publish();
    },
    /** Marks the run finished: every stage complete, 100%, no ETA. */
    finish() {
      for (const stage of stages) {
        if (stage.status !== 'complete') {
          stage.status = 'complete';
          stage.fraction = 1;
          stage.startedAt = stage.startedAt || now();
          stage.endedAt = stage.endedAt || now();
        }
      }
      index = stages.length - 1;
      highWater = 1;
      publish();
      job.progress = 100;
      job.stage = 'Complete';
      job.stageProgress = 100;
      job.etaMs = 0;
      logger?.ok(`Reconstruction complete in ${Math.round((now() - startedAt) / 1000)}s`);
      return job;
    },
    /** Skipped stages still have to stop blocking the bar. */
    skip(key, reason = null) {
      const target = stages.findIndex((stage) => stage.key === key);
      if (target < 0 || stages[target].status === 'complete') return publish();
      stages[target].fraction = 1;
      stages[target].status = 'skipped';
      stages[target].startedAt = stages[target].startedAt || now();
      stages[target].endedAt = now();
      logger?.warn(`${stages[target].label} skipped${reason ? `: ${reason}` : ''}`);
      if (index < target) index = target;
      return publish(reason);
    },
    snapshot: () => publish(job.stageDetail),
  };
}

export function cpuThreads() {
  try { return Math.max(2, availableParallelism()); } catch { return 8; }
}

/** Blank or invalid means "use every core". */
export function normalizeThreads(value) {
  const cores = cpuThreads();
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return String(cores);
  return String(Math.min(cores, Math.max(1, Math.round(n))));
}

/** Blank means the profile picks the frame count. Hard cap keeps a 10 min pass inside the time budget. */
export function normalizeFrames(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  return Math.min(240, Math.max(12, Math.round(n)));
}

export function normalizeQuality(value) {
  const mode = String(value || 'fast').toLowerCase();
  if (mode === 'detailed' || mode === 'studio' || mode === 'survey' || mode === 'full') return 'detailed';
  if (mode === 'balanced' || mode === 'standard') return 'balanced';
  return 'fast';
}

/**
 * Stereo time scales with frames × pixels × samples × iterations × 2
 * (the second factor is the geometric-consistency pass). A measured 80-frame
 * 1280 px run spent ~13 min in stereo and ~8 min in CPU bundle adjustment.
 * Rapid stays lighter than Detailed by using a 1024 px depth map, fewer
 * PatchMatch iterations, and GPU bundle adjustment. It still samples fast
 * flights densely enough that neighbouring frames overlap — a 67 s FPV pass
 * at 0.6 fps registered only a fragment and the mesh covered almost nothing.
 * Geometric consistency stays on in every profile — dropping it is what
 * turns the cloud into spray.
 */
function surfaceFor(quality, tier) {
  const shared = {
    cpu: { resolution: 360, smoothing: 3, slope: 2.5, ortho: 1024, points: 120000 },
    low: { resolution: 480, smoothing: 2, slope: 3.2, ortho: 1280, points: 160000 },
  };
  // `points` caps the coloured cloud kept for the viewer GLB; `ortho` is the
  // top-down atlas resolution; `resolution` is the DSM grid. Detailed profiles
  // carry far more points and a 4K ortho so fine ground detail survives.
  const ranked = {
    mid: {
      fast: { resolution: 760, smoothing: 2, slope: 4.6, ortho: 1792, points: 300000 },
      balanced: { resolution: 900, smoothing: 1, slope: 5.2, ortho: 2560, points: 500000 },
      detailed: { resolution: 1100, smoothing: 1, slope: 7, ortho: 4096, points: 900000 },
    },
    high: {
      fast: { resolution: 840, smoothing: 1, slope: 5.5, ortho: 1920, points: 340000 },
      balanced: { resolution: 1024, smoothing: 1, slope: 6.5, ortho: 2560, points: 600000 },
      detailed: { resolution: 1400, smoothing: 1, slope: 7.5, ortho: 4096, points: 1300000 },
    },
  };
  if (ranked[tier]) return ranked[tier][quality] || ranked[tier].fast;
  const base = shared[tier] || shared.cpu;
  if (quality === 'detailed') {
    return { ...base, resolution: Math.round(base.resolution * 1.35), smoothing: 1, slope: base.slope + 1.5, ortho: 1536 };
  }
  if (quality === 'balanced') return { ...base, resolution: Math.round(base.resolution * 1.15), slope: base.slope + 0.6 };
  return base;
}

/**
 * Minimum frames so a moving camera still shares scene between neighbours.
 * A fixed preset (40 frames on a one-minute clip) is too sparse for FPV:
 * the mapper discards most of the flight and the DSM is a sliver.
 */
function framesForMotion(quality, durationSec, presetCount) {
  const policy = {
    fast: { rate: durationSec <= 120 ? 2 : durationSec <= 300 ? 1.5 : 1, floor: 72, cap: 180 },
    balanced: { rate: durationSec <= 120 ? 2.2 : durationSec <= 300 ? 1.6 : 1.1, floor: 80, cap: 200 },
    detailed: { rate: durationSec <= 120 ? 2.5 : durationSec <= 300 ? 1.8 : 1.2, floor: 96, cap: 220 },
  }[quality] || { rate: 2, floor: 72, cap: 180 };
  const fromRate = Math.ceil(Math.max(1, durationSec) * policy.rate);
  return Math.min(policy.cap, Math.max(policy.floor, presetCount, fromRate));
}

/**
 * OpenMVS tuning for the high-detail textured-mesh path.
 *
 * These knobs trade VRAM and wall-clock for geometric and texture detail, so
 * they are chosen against the *measured* budget of this machine: an 8 GB RTX
 * 4060 and 28 i7 threads. Densify and texture run on the GPU, so the thread
 * count below (0 = every core) only really loads the CPU during mesh
 * reconstruction and refinement — exactly where the user asked for "full
 * computational resources".
 *
 *   resolutionLevel   how many times to halve the image before depth-map
 *                     estimation. 1 = half-res (detail), 2 = quarter (speed).
 *   maxResolution     hard pixel ceiling so a 4K frame cannot blow past VRAM.
 *   minPointDistance  Delaunay triangulation spacing in pixels; smaller packs
 *                     more triangles into the mesh (finer surface).
 *   refine            photo-consistent vertex optimisation — the single most
 *                     expensive stage, enabled only for the "detailed" profile.
 *   maxTextureSize    atlas resolution; 8192 keeps signage/texture legible.
 */
function mvsFor(quality, hw, mvsWidth = 0) {
  // High tiers (>=10 GB VRAM) can afford full-resolution densify and GPU refine;
  // the 8 GB mid tier stays one level down so depth-map estimation fits in VRAM.
  const roofed = hw.tier === 'high';
  // Photo-consistent mesh refinement (OpenMVS RefineMesh) optimises vertex
  // positions against the source photographs, which is what turns a blobby
  // graph-cut surface into one with crisp roof edges and flat, upright
  // facades. Facade texture distortion is driven by mesh geometry error, so
  // on a job whose whole point is inspecting building fronts this is not a
  // luxury — it is the stage that makes the fronts legible.
  //
  // It is now ON for `detailed` and off for the faster profiles. It is the
  // most VRAM-hungry stage, so openmvs.mjs degrades to the unrefined mesh on
  // OOM rather than failing. SKYFORGE_MESH_REFINE=0 forces it off, =1 forces
  // it on for every profile.
  const refineEnv = process.env.SKYFORGE_MESH_REFINE;
  const refineForced = refineEnv === '1';
  const refineDisabled = refineEnv === '0';
  // Photographic texture atlas. This is what stops the model looking like a washed-out
  // blob: vertex colours can only carry as much colour as the mesh has VERTICES, so
  // every triangle we drop for speed also threw away texture. An atlas decouples the
  // two — 4K-frame detail on a light, fast mesh. openmvs.mjs validates the bake and
  // silently falls back to vertex colours if it comes out blank, so this is safe to
  // default on; SKYFORGE_NO_ATLAS=1 forces the old vertex-colour-only path.
  const atlas = process.env.SKYFORGE_NO_ATLAS !== '1';
  const base = {
    // 0 = use every available core. OpenMVS manages its own memory with
    // reduce-memory, so this is the knob that honours "use full resources".
    threads: 0,
    cudaDevice: -1,
    minResolution: 640,
    subResolutionLevels: 2,
    numberViewsFuse: 2,
    densifyIters: 4,
    geometricIters: 2,
    meshSmooth: 2,
    removeSpurious: 20,
    closeHoles: 30,
    meshDecimate: 1,
    refine: false,
    refineResolutionLevel: 2,
    refineMaxViews: 8,
    refineScales: 1,
    // Refinement defaults to the GPU but falls back to the unrefined mesh if the
    // 8 GB card runs out of room (handled in openmvs.mjs).
    refineCudaDevice: -1,
  };
  const byQuality = {
    fast: {
      resolutionLevel: 2,
      maxResolution: 1600,
      numberViews: 6,
      numberViewsFuse: 2,
      minPointDistance: 2.2,
      removeSpurious: 25,
      textureResolutionLevel: 1,
      maxTextureSize: 4096,
      refine: false,
    },
    balanced: {
      resolutionLevel: roofed ? 1 : 1,
      maxResolution: 2048,
      numberViews: 7,
      numberViewsFuse: 2,
      minPointDistance: 1.5,
      textureResolutionLevel: roofed ? 0 : 1,
      maxTextureSize: 8192,
      textureAtlas: atlas,
      refine: false,
    },
    detailed: {
      // resolution-level 1 on 2560px undistorted frames => ~1280px depth maps,
      // which keeps far more of the 4K source detail while still fitting the 8 GB
      // card (the reference run used ~1.3 GB at 960px; 1280px stays well inside).
      resolutionLevel: roofed ? 0 : 1,
      maxResolution: roofed ? 3840 : 2560,
      minResolution: 960,
      numberViews: roofed ? 10 : 9,
      // Fuse points agreed by 2 views (not 3): keep more of the fine geometry the
      // user said was being lost, rather than discarding thin/edge detail.
      numberViewsFuse: 2,
      densifyIters: 5,
      // Delaunay spacing (pixels) is the dominant cost driver of ReconstructMesh:
      // halving it ~quadruples the tetrahedra and the (super-linear) graph-cut time.
      // At 0.7 a 156-frame clip built a 6.4 M-point cloud → 38.9 M cells → a 28-min
      // graph-cut and a 44-min mesh that overran the old timeout. ~1.1 keeps plenty
      // of triangles for crisp roofs/facades (the extruded building blocks carry the
      // sharp vertical geometry anyway) while roughly halving the point count, the
      // cell count and the mesh wall-clock — faster AND far less likely to stall/OOM
      // on the 16 GB-RAM reference machine. The high tier (>=10 GB VRAM, more RAM)
      // can still afford a denser 0.8.
      minPointDistance: roofed ? 0.8 : 1.1,
      meshSmooth: 1,
      removeSpurious: 15,
      // Close only genuine small gaps. A high value (was 40) bridges the voids
      // BETWEEN buildings — streets, courtyards, river banks the nadir camera never
      // saw into — welding the whole city into one smooth "draped" sheet, which is
      // exactly why the result reads as a landscape photo instead of distinct 3D
      // buildings. Keep it low so separate structures stay separate.
      closeHoles: 18,
      // Full-resolution texture atlas straight from the sharpest frames — this is
      // where visible photographic detail (roof tiles, markings) actually lives.
      textureResolutionLevel: 0,
      maxTextureSize: 8192,
      textureAtlas: atlas,
      refine: true,
      refineResolutionLevel: 1,
      refineScales: 2,
    },
  };
  const tuned = { ...base, ...(byQuality[quality] || byQuality.fast) };

  // ---- Texture-stage cost control -----------------------------------------
  // TextureMesh was 68% of a measured 31m46s run (6m36s assigning views to
  // 868k faces, then 14m59s packing 36k patches). Two corrections:
  //
  // 1. Never ask for an atlas the engine will not give. The old `detailed`
  //    profile requested 8192 and OpenMVS emitted 4096 anyway, so the request
  //    only inflated allocation and packing work. 4096 is the honest ceiling
  //    below the high tier.
  // 2. Do not texture from more pixels than the frames actually hold. With the
  //    source-width cap in place, resolution-level 0 on a 1080p clip is already
  //    native; there is nothing to gain by asking for a level the source cannot
  //    feed, and plenty of time to lose.
  if (hw.tier !== 'high') tuned.maxTextureSize = Math.min(tuned.maxTextureSize, 4096);
  if (mvsWidth && mvsWidth <= 1280) {
    // At or below 720p every pixel is precious: texture at full resolution.
    tuned.textureResolutionLevel = 0;
  } else if (mvsWidth && mvsWidth <= 1920 && hw.tier !== 'high') {
    // 1080p on an 8 GB card: full-resolution texturing is affordable only
    // because the source cap already removed the upscale.
    tuned.textureResolutionLevel = 0;
  }

  // The `fast` profile never set textureAtlas at all, so openmvs.mjs took its
  // early return and silently colour-shifted to the vertex-colour path. Three
  // profiles, three different-looking products, no warning. Every profile now
  // bakes an atlas; they differ in resolution, not in kind.
  if (tuned.textureAtlas === undefined) tuned.textureAtlas = atlas;

  // Refinement override, applied after the per-quality defaults.
  if (refineForced) tuned.refine = true;
  if (refineDisabled) tuned.refine = false;

  // Seam levelling: see the long note at the TextureMesh call site in
  // openmvs.mjs. Off unless explicitly re-enabled.
  tuned.seamLeveling = process.env.SKYFORGE_SEAM_LEVELING === '1' ? 1 : 0;

  // Texturing runs on the CPU by default.
  //
  // The texture stage is not GPU-bound in practice -- on a measured run the
  // 21m41s it took was 6m36s of view assignment plus 14m59s of atlas packing,
  // both CPU work, with a 4.11 GB host working set and heavy paging. So the
  // CUDA path buys little here, while the corrupted bake we measured (black
  // face interiors with saturated single-pixel RGB speckles along their
  // boundaries, which is what reading uninitialised memory looks like) is
  // consistent with a fault in it. -2 is OpenMVS's "CPU processing".
  // SKYFORGE_TEXTURE_CUDA=1 restores the GPU path.
  tuned.textureCudaDevice = process.env.SKYFORGE_TEXTURE_CUDA === '1' ? -1 : -2;

  return tuned;
}

export function qualitySettings(mode, durationSec = 10, overrides = {}, hardware = null) {
  const quality = normalizeQuality(mode);
  const band = durationSec <= 90 ? 'short' : durationSec >= 480 ? 'long' : 'mid';
  const hw = hardware || classifyHardware({ logicalThreads: cpuThreads(), ramMb: 16384, vramMb: 8192, gpuName: 'NVIDIA GPU' });
  const presets = {
    fast: {
      frames: { short: 72, mid: 96, long: 120 },
      features: '12000',
      overlap: '16',
      dense: { short: '1024', mid: '1024', long: '960' },
      samples: '6',
      iterations: '3',
      baGlobalRefinements: '1',
      baGlobalIterations: '20',
      baLocalIterations: '12',
      baGlobalRatio: '1.4',
      baGlobalFreq: '1000',
      baLocalImages: '8',
      peakThreshold: '0.006',
      initTriAngle: '4',
      minMatches: '12',
    },
    balanced: {
      frames: { short: 56, mid: 80, long: 96 },
      features: '10000',
      overlap: '12',
      dense: { short: '1280', mid: '1024', long: '960' },
      samples: '7',
      iterations: '4',
      // Global BA is the single most expensive, crash-prone step (it re-optimizes
      // every camera+point and can OOM the GPU on a full model). Make each pass cheap
      // (1 refinement, 12 iterations) and rare (fire only every 70% growth) so a short
      // drone clip does ~5 quick global passes instead of ~18 slow ones — much faster,
      // and the final BA still tightens everything.
      baGlobalRefinements: '1',
      baGlobalIterations: '12',
      baLocalIterations: '12',
      baGlobalRatio: '1.7',
      baGlobalFreq: '1500',
      baLocalImages: '6',
      peakThreshold: '0.006',
      initTriAngle: '6',
      minMatches: '15',
    },
    detailed: {
      frames: { short: 72, mid: 100, long: 120 },
      // Fewer features + less matching overlap => far fewer BA observations, which
      // is what makes each bundle-adjustment pass slow on a modest machine (a single
      // 57-frame global BA was taking >7 min). 10k features is still plenty for a
      // reliable solve; the dense OpenMVS stage re-extracts detail from full-res
      // pixels, so final texture detail is unaffected.
      features: '10000',
      overlap: '12',
      dense: { short: '1280', mid: '1280', long: '1024' },
      samples: '8',
      iterations: '4',
      // Cap global-BA cost AND frequency so Stage 4 cannot stall/crash in a
      // 10-minute full-model bundle adjustment: one refinement, 12 iterations, and
      // fire it only every 70% growth. Detail comes from the dense stage, not extra
      // sparse BA passes.
      baGlobalRefinements: '1',
      baGlobalIterations: '12',
      baLocalIterations: '15',
      baGlobalRatio: '1.7',
      baGlobalFreq: '1500',
      baLocalImages: '6',
      peakThreshold: '0.004',
      initTriAngle: '8',
      minMatches: '15',
    },
  };
  const preset = presets[quality];
  const scaledPreset = Math.round(preset.frames[band] * (hw.frameScale || 1));
  // The motion floor wins over a too-small manual count. Typing 40 frames
  // on a one-minute FPV pass is what left the last Rapid mesh empty.
  const motionFloor = framesForMotion(quality, durationSec, scaledPreset);
  const sourceCap = Math.max(12, Math.floor(Math.max(durationSec, 1) * 12));
  const targetFrames = Math.min(240, sourceCap, Math.max(motionFloor, overrides.frames || 0));
  let dense = Number(preset.dense[band]);
  if (hw.tier === 'mid' && quality === 'fast') dense = band === 'long' ? 960 : 1024;
  if (hw.tier === 'high') dense = Math.round(dense * (quality === 'fast' ? 1.1 : 1.2));
  dense = Math.min(Math.max(512, dense), hw.denseCap || 1024);
  const fps = Math.min(12, Math.max(0.15, targetFrames / Math.max(durationSec, 1)));
  const surface = surfaceFor(quality, hw.tier);
  // Extraction / undistort width.
  //
  // OpenMVS downsamples internally but can never recover detail that frame
  // extraction threw away, so we want the sharpest frames the hardware can
  // afford — but NEVER wider than the source. The preset used to be an absolute
  // target with no source cap, so a 1280x720 clip on `detailed` was lanczos
  // upscaled to 2560x1440 and every downstream stage paid 4x the pixels for
  // interpolated detail that does not exist. On one measured run that alone
  // drove TextureMesh to 21m41s of a 31m46s job.
  //
  // Upscaling is also rounded down to an even width: `scale=W:-2` needs an even
  // height, and an odd source width would otherwise shift the aspect ratio.
  const presetWidth = { fast: 1280, balanced: 1920, detailed: hw.tier === 'high' ? 3840 : 2560 }[quality] || 1280;
  const sourceWidth = Number(overrides.sourceWidth) > 0 ? Math.floor(Number(overrides.sourceWidth)) : 0;
  const mvsWidth = sourceWidth
    ? Math.max(640, Math.min(presetWidth, sourceWidth - (sourceWidth % 2)))
    : presetWidth;
  const upscalingAvoided = Boolean(sourceWidth && presetWidth > sourceWidth);
  return {
    quality,
    exportMode: quality,
    hardware: hw,
    surface,
    mvs: mvsFor(quality, hw, mvsWidth),
    mvsWidth,
    presetWidth,
    sourceWidth,
    upscalingAvoided,
    undistortSize: String(mvsWidth),
    fps: fps.toFixed(3),
    targetFrames,
    threads: overrides.threads ? normalizeThreads(overrides.threads) : String(hw.workerThreads),
    scale: `${mvsWidth}:-2`,
    features: preset.features,
    overlap: preset.overlap,
    denseSize: String(dense),
    samples: preset.samples,
    iterations: preset.iterations,
    windowRadius: '5',
    windowStep: '2',
    stereoCache: String(hw.cacheGb || 4),
    fusionCache: String(Math.min(4, hw.cacheGb || 4)),
    baGlobalRefinements: preset.baGlobalRefinements,
    baGlobalRatio: preset.baGlobalRatio,
    baGlobalIterations: preset.baGlobalIterations,
    baLocalIterations: preset.baLocalIterations,
    baGlobalFreq: preset.baGlobalFreq,
    baLocalImages: preset.baLocalImages,
    poissonDepth: quality === 'detailed' ? '11' : '10',
    poissonTrim: '3',
    peakThreshold: preset.peakThreshold,
    initTriAngle: preset.initTriAngle || '8',
    minMatches: preset.minMatches || '15',
    minTriangulation: '1.0',
    filterTriangulation: '2.0',
  };
}

/**
 * Altitude drives the metric scale of every exported product, so an absurd or
 * missing value would silently mis-size the whole model rather than fail.
 */
export function normalizeAltitude(value, fallback = 82) {
  const altitude = Number(value);
  if (!Number.isFinite(altitude) || altitude <= 0) return fallback;
  return Math.min(2000, Math.max(2, altitude));
}

async function firstExisting(paths) {
  for (const path of paths) {
    if (existsSync(path)) return path;
  }
  return null;
}

/**
 * COLMAP dense stereo + fusion — the fallback dense path.
 *
 * Used when OpenMVS is not installed, or when its textured-mesh chain fails on
 * this machine (most often VRAM). It produces a fused point cloud only (no
 * textured mesh), which still feeds the gravity-aligned DSM so the operator is
 * never left empty-handed. Runs under the single "densify" stage.
 *
 * @returns {Promise<string|null>} path to fused.ply, or null if it produced nothing
 */
async function colmapDenseFallback({ dense, work, job, tracker, quality, chosen, frameCount }) {
  const logger = job.logger;
  try {
    logger?.warn('Running COLMAP dense stereo fallback (produces a DSM, not a textured mesh).');
    await run(commands.colmap, [
      'patch_match_stereo',
      '--workspace_path', dense,
      '--workspace_format', 'COLMAP',
      // Cross-view geometric consistency is the difference between a surface and
      // a cloud of noise: it rejects depths neighbouring views disagree about.
      '--PatchMatchStereo.geom_consistency', '1',
      '--PatchMatchStereo.gpu_index', '0',
      '--PatchMatchStereo.max_image_size', quality.denseSize,
      '--PatchMatchStereo.window_radius', quality.windowRadius,
      '--PatchMatchStereo.window_step', quality.windowStep,
      '--PatchMatchStereo.num_samples', quality.samples,
      '--PatchMatchStereo.num_iterations', quality.iterations,
      '--PatchMatchStereo.min_triangulation_angle', quality.minTriangulation,
      '--PatchMatchStereo.filter_min_triangulation_angle', quality.filterTriangulation,
      '--PatchMatchStereo.filter_min_ncc', '0.10',
      '--PatchMatchStereo.filter_min_num_consistent', '2',
      '--PatchMatchStereo.cache_size', quality.stereoCache,
      '--PatchMatchStereo.num_threads', quality.threads,
      '--PatchMatchStereo.filter', '1',
    ], work, job, 'Dense stereo on GPU (COLMAP fallback)', {
      stage: 'densify',
      total: chosen.images || frameCount,
      passes: 2,
    });
    await run(commands.colmap, [
      'stereo_fusion',
      '--workspace_path', dense,
      '--workspace_format', 'COLMAP',
      '--input_type', 'geometric',
      '--output_path', join(dense, 'fused.ply'),
      '--StereoFusion.min_num_pixels', '5',
      '--StereoFusion.max_reproj_error', '2',
      '--StereoFusion.max_depth_error', '0.01',
      '--StereoFusion.max_normal_error', '15',
      '--StereoFusion.cache_size', quality.fusionCache,
      '--StereoFusion.num_threads', quality.threads,
    ], work, job, 'Fusing dense point cloud (COLMAP fallback)', {
      stage: 'densify',
      total: chosen.images || frameCount,
    });
    tracker.complete('densify');
    const fused = join(dense, 'fused.ply');
    if (existsSync(fused) && statSync(fused).size > 1024) return fused;
  } catch (error) {
    if (/cancelled|timeout/i.test(error.message)) throw error;
    tracker.skip('densify', error.message);
    job.warning = `${job.warning || ''} Dense stereo was skipped: ${error.message}. Using the sparse reconstruction.`.trim();
  }
  return null;
}

export async function reconstruct(job) {
  const work = job.work;
  const frames = join(work, 'frames');
  const database = join(work, 'database.db');
  const sparse = join(work, 'sparse');
  const dense = join(work, 'dense');
  const products = join(work, 'products');
  await mkdir(frames, { recursive: true });
  await mkdir(sparse, { recursive: true });
  await mkdir(dense, { recursive: true });
  await mkdir(products, { recursive: true });

  if (!existsSync(job.input) || statSync(job.input).size < 64 * 1024) {
    throw new Error('The uploaded video is empty or truncated.');
  }
  for (const [name, binary] of [['FFmpeg', commands.ffmpeg], ['COLMAP', commands.colmap]]) {
    if (!existsSync(binary)) {
      throw new Error(`${name} was not found at ${binary}. Set the ${name.toUpperCase()}_PATH environment variable.`);
    }
  }

  const info = await probeVideo(job.input);
  if (!info.reliable) {
    job.warning = `Could not read the clip's duration (${info.error || 'unknown error'}); sampling assumes ${info.duration}s, so frame coverage may be wrong.`;
  }
  const hardware = hardwareFromTelemetry(await sampleTelemetry());
  job.hardware = hardware;
  const quality = qualitySettings(job.quality, info.duration, {
    frames: normalizeFrames(job.frames),
    threads: job.threads || null,
    // Cap every width at the source. Upscaling buys no detail and costs
    // quadratically in the dense and texture stages.
    sourceWidth: info.reliable ? info.width : 0,
  }, hardware);
  job.profile = quality.quality;

  // Per-run structured logger: one file inside the job workspace for forensics,
  // one rolling file under logs/ for the whole machine, plus the live terminal.
  const useMvs = openmvsAvailable(commands.openmvs) && process.env.SKYFORGE_NO_MVS !== '1';
  // Live log ring buffer.
  //
  // The browser used to get only the stage name, so a twenty-minute texturing
  // pass looked identical to a hung process. Every line the terminal sees is
  // now mirrored here with a monotonic sequence number; the server streams the
  // lines each client has not yet seen. Bounded, because a long run emits tens
  // of thousands of engine lines and the job object lives in memory.
  job.log = [];
  job.logSeq = 0;
  job.lastOutputAt = Date.now();
  const LOG_RING = 1500;
  const logger = createLogger({
    label: `${job.fileName || 'capture'} · ${quality.quality}`,
    files: [join(work, 'pipeline.log'), resolve('logs', 'skyforge.log')],
    quiet: Boolean(job.quietLog),
    onLine: (line) => {
      job.lastOutputAt = line.at;
      job.logSeq += 1;
      job.log.push({ seq: job.logSeq, level: line.level, message: line.message, at: line.at });
      if (job.log.length > LOG_RING) job.log.splice(0, job.log.length - LOG_RING);
    },
  });
  job.logger = logger;
  logger.info(`Reconstruction ${job.id || ''} started`);
  logger.metric('source', job.fileName || 'capture');
  logger.metric('profile', `${quality.quality} (${quality.targetFrames} frames @ ${quality.fps} fps, ${quality.mvsWidth}px)`);
  logger.metric('hardware', `${hardware.gpuName || 'CPU only'} · ${hardware.vramGb} GB VRAM · ${hardware.ramGb} GB RAM · ${hardware.logicalThreads} threads`);
  logger.metric('dense path', useMvs ? 'OpenMVS textured mesh (GPU) + DSM' : 'COLMAP dense stereo + DSM (OpenMVS unavailable)');
  // Report which optional upgrade engines are active this run (hloc/GLOMAP/mask).
  const upgrades = detectAll({
    hlocScript: resolve('tools/python/hloc_match.py'),
    maskScript: resolve('tools/python/mask_frames.py'),
    splatScript: resolve('tools/python/train_splat.py'),
  });
  const activeUpgrades = Object.entries(upgrades)
    .filter(([, v]) => v.available)
    .map(([k, v]) => k + (v.mode ? `:${v.mode}` : ''));
  logger.metric('upgrades', activeUpgrades.length ? activeUpgrades.join(', ') : 'none (bundled baseline)');
  job.upgrades = upgrades;
  if (!info.reliable) logger.warn(job.warning);

  const tracker = createProgressTracker(job, buildStagePlan({
    durationSec: info.duration,
    frameCount: quality.targetFrames,
  }), () => Date.now(), logger);
  job.tracker = tracker;
  tracker.start('extract', `Extracting ${quality.targetFrames} frames from the full flight`);

  // ---- Frame extraction, GPU-decoded when the card can do it --------------
  // Drone footage is long-GOP H.264/HEVC at a high bitrate, so decoding is real
  // work: 88s of 1080p50 HEVC at 70 Mbps costs 31s on 16 CPU threads and 15s on
  // NVDEC, producing byte-identical JPEGs (verified channel means match to
  // 0.00). scale_cuda keeps the resize on the card too, so only the final
  // unsharp and JPEG encode touch the CPU.
  //
  // Hardware decode is attempted first and falls back to software on any
  // failure: codecs vary, and an unsupported profile must cost us a retry, not
  // the job.
  const vfTail = `unsharp=5:5:0.45:5:5:0.0`;
  const extractBase = ['-y', '-an', '-sn', '-dn'];
  const extractTail = ['-q:v', '2', join(frames, 'frame_%06d.jpg')];
  const gpuDecodeArgs = [
    ...extractBase,
    '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
    '-i', job.input,
    '-vf', `fps=${quality.fps},scale_cuda=${quality.scale}:interp_algo=lanczos,hwdownload,format=nv12,format=yuv420p,${vfTail}`,
    ...extractTail,
  ];
  const cpuDecodeArgs = [
    ...extractBase,
    '-threads', quality.threads,
    '-i', job.input,
    '-vf', `fps=${quality.fps},scale=${quality.scale}:flags=lanczos,${vfTail}`,
    ...extractTail,
  ];

  const extractLabel = `Extracting ${quality.targetFrames} frames from the full flight`;
  let decodedOnGpu = false;
  if (process.env.SKYFORGE_NO_NVDEC !== '1') {
    try {
      await run(commands.ffmpeg, gpuDecodeArgs, work, job, `${extractLabel} (GPU decode)`,
        { stage: 'extract', total: quality.targetFrames });
      decodedOnGpu = (await readdir(frames)).some((name) => name.endsWith('.jpg'));
      if (decodedOnGpu) logger.ok('Frame extraction: NVDEC hardware decode');
      else logger.warn('NVDEC produced no frames; retrying with software decode.');
    } catch (error) {
      if (/cancelled|timeout/i.test(error.message)) throw error;
      logger.warn(`NVDEC decode failed (${error.message}); retrying with software decode.`);
    }
  }
  if (!decodedOnGpu) {
    await run(commands.ffmpeg, cpuDecodeArgs, work, job, `${extractLabel} (CPU decode)`,
      { stage: 'extract', total: quality.targetFrames });
  }
  if (quality.upscalingAvoided) {
    logger.info(
      `Source is ${quality.sourceWidth}px wide, so frames are extracted at ${quality.mvsWidth}px rather than the `
      + `${quality.presetWidth}px profile target. Upscaling adds no detail and costs quadratically downstream.`,
    );
  }

  // --- Blur / keyframe filtering: drop the blurriest frames before matching ---
  // (SkyForge_Upgrade_Plan §6.1 — biggest quality-per-hour; Node + FFmpeg only.)
  if (process.env.SKYFORGE_NO_BLURFILTER !== '1') {
    try {
      job.blurFilter = await selectSharpFrames(commands.ffmpeg, frames, {}, logger);
    } catch (error) {
      logger.warn(`Blur filter skipped: ${error.message}`);
    }
  }

  // --- Optional sky / dynamic-object masking -> COLMAP mask_path ---
  // Removes the sky/cloud "ceiling" and moving cars/people at the source, so they
  // never become ghost geometry (the fix for the white-haze artefacts).
  let maskDir = null;
  const masking = detectMasking(resolve('tools/python/mask_frames.py'));
  if (masking.available) {
    maskDir = join(work, 'masks');
    try {
      await mkdir(maskDir, { recursive: true });
      logger.info(`Masking sky/dynamic objects (${masking.mode} backend)`);
      await run(masking.python, [
        masking.script, '--images', frames, '--out', maskDir, '--backend', 'auto',
        ...(job.maskDynamic ? ['--dynamic'] : []),
      ], work, job, 'Masking sky / dynamic objects', { stage: 'extract', timeoutMs: 30 * 60 * 1000 });
      const produced = (await readdir(maskDir)).filter((n) => n.endsWith('.png')).length;
      if (!produced) { maskDir = null; logger.warn('Masking produced no masks; continuing without.'); }
      else logger.ok(`Masking: ${produced} frame masks written`);
    } catch (error) {
      logger.warn(`Masking skipped: ${error.message}`);
      maskDir = null;
    }
  } else {
    logger.info(`Masking off (${masking.reason})`);
  }

  const frameCount = (await readdir(frames)).filter((name) => name.endsWith('.jpg')).length;
  if (frameCount < 8) throw new Error(`Only ${frameCount} frames were extracted. Need a longer or more detailed clip.`);
  tracker.complete('extract');

  // --- Features + matching: learned (hloc + LightGlue) when available, else SIFT ---
  const hloc = detectHloc(resolve('tools/python/hloc_match.py'));
  let featuresDone = false;
  if (hloc.available) {
    logger.info('Features + matching: hloc + LightGlue (SuperPoint) — learned, robust to blur/low texture');
    tracker.start('features', 'Learned features + matching (hloc + LightGlue)');
    try {
      await run(hloc.python, [
        hloc.script, '--images', frames, '--database', database,
        '--window', String(Math.max(5, Number(quality.overlap) || 10)), '--loop', '0',
      ], work, job, 'hloc + LightGlue features & matching', { stage: 'features', total: frameCount, timeoutMs: 90 * 60 * 1000 });
      featuresDone = existsSync(database) && statSync(database).size > 4096;
      if (featuresDone) { tracker.complete('features'); tracker.complete('match'); }
      else logger.warn('hloc produced no usable database; falling back to COLMAP SIFT.');
    } catch (error) {
      if (/cancelled|timeout/i.test(error.message)) throw error;
      logger.error(`hloc failed (${error.message}); falling back to COLMAP SIFT.`, error);
    }
    // A half-written db would confuse COLMAP — start the fallback clean.
    if (!featuresDone) { try { await rm(database, { force: true }); } catch { /* ignore */ } }
  }
  if (!featuresDone) {
    // ---- Learned features in-process (COLMAP 4.2 ALIKED + LightGlue) --------
    // Preferred over SIFT for aerial work. SIFT keys off corners and blobs, and
    // a drone pass is mostly repetitive facades, regular window grids and large
    // low-texture roofs and roads — exactly where SIFT produces either too few
    // matches or confidently wrong ones between near-identical windows.
    // LightGlue matches with global context and rejects those.
    //
    // Needs the ONNX CUDA provider, which needs cuDNN 9. Without it the
    // provider silently falls back to CPU and is slower than SIFT for no gain,
    // so we check first and say which path we took rather than quietly
    // degrading.
    const wantLearned = process.env.SKYFORGE_FEATURES !== 'sift';
    const learned = wantLearned && cudnnAvailable();
    if (wantLearned && !learned) {
      logger.warn(
        'cuDNN 9 not found in engines/cudnn, so ALIKED + LightGlue would run on the CPU. '
        + 'Using SIFT instead. Run `npm run setup:cudnn` to enable the learned path.',
      );
    }

    logger.info(
      learned
        ? 'Features + matching: COLMAP ALIKED + LightGlue (GPU, learned)'
        : `Features + matching: COLMAP SIFT${hloc.available ? ' (hloc fallback)' : ` (${hloc.reason})`}`,
    );

    const extractorArgs = learned
      ? [
        '--FeatureExtraction.type', 'ALIKED_N16ROT',
        // ALIKED keypoints are far better localised than SIFT, so a smaller
        // budget carries the same geometry at a fraction of the match cost.
        '--AlikedExtraction.max_num_features', String(Math.min(8192, Math.round(Number(quality.features) / 2) || 4096)),
      ]
      : [
        '--SiftExtraction.max_num_features', quality.features,
        '--SiftExtraction.max_num_orientations', '2',
        '--SiftExtraction.peak_threshold', quality.peakThreshold,
        '--SiftExtraction.edge_threshold', '10',
      ];

    await run(commands.colmap, [
      'feature_extractor',
      '--database_path', database,
      '--image_path', frames,
      '--ImageReader.single_camera', '1',
      ...(maskDir ? ['--ImageReader.mask_path', maskDir] : []),
      ...extractorArgs,
      '--FeatureExtraction.use_gpu', '1',
      '--FeatureExtraction.num_threads', quality.threads,
    ], work, job, learned ? 'Detecting learned GPU features (ALIKED)' : 'Detecting GPU features (SIFT)',
    { stage: 'features', total: frameCount });
    tracker.complete('features');

    await run(commands.colmap, [
      'sequential_matcher',
      '--database_path', database,
      ...(learned
        ? ['--FeatureMatching.type', 'ALIKED_LIGHTGLUE']
        // Guided matching is a SIFT-era second pass; LightGlue already uses
        // global context, so it is redundant (and costly) on the learned path.
        : ['--FeatureMatching.guided_matching', '1']),
      '--FeatureMatching.use_gpu', '1',
      '--FeatureMatching.num_threads', quality.threads,
      '--SequentialMatching.overlap', quality.overlap,
      '--SequentialMatching.quadratic_overlap', '1',
      // A single drone pass that circles back over the same block gets no loop
      // closure from sequential pairing alone, and the two visits drift apart
      // into a doubled, smeared reconstruction. Vocab-tree loop detection costs
      // little and ties them together.
      '--SequentialMatching.loop_detection', '1',
      '--SequentialMatching.num_threads', quality.threads,
    ], work, job, 'Matching sequential drone views', { stage: 'match', total: frameCount });
    tracker.complete('match');
    job.metrics = { ...(job.metrics || {}), featureBackend: learned ? 'aliked+lightglue' : 'sift' };
  }

  // Incremental-mapper args, hoisted so the global-mapper branch, the COLMAP
  // fallback and the thin-model self-heal below all share one definition.
  // Hover / slow-pan clips have tiny baselines, so the init triangulation floor
  // sits well below COLMAP's 16 degree default or the mapper never finds a
  // first pair.
  //
  // Note on GPU bundle adjustment: this COLMAP build reports
  // "Ceres was compiled without CUDA support" and
  // "Ceres was compiled without cuDSS support", so --Mapper.ba_use_gpu is a
  // no-op that silently falls back to the CPU solver. The old code branched on
  // it, capped it at 64 frames to avoid a VRAM OOM that could never happen, and
  // labelled the stage "Solving camera poses on GPU" — all of it inert and the
  // label simply untrue. Bundle adjustment runs on the CPU here; the GPU is
  // saturated later by densify and texturing.
  const mapperArgs = () => [
    'mapper',
    '--database_path', database,
    '--image_path', frames,
    '--output_path', sparse,
    '--Mapper.num_threads', quality.threads,
    '--Mapper.ba_local_num_images', quality.baLocalImages,
    '--Mapper.ba_local_max_num_iterations', quality.baLocalIterations,
    '--Mapper.ba_global_frames_freq', quality.baGlobalFreq,
    '--Mapper.ba_global_max_refinements', quality.baGlobalRefinements,
    '--Mapper.ba_global_frames_ratio', quality.baGlobalRatio,
    '--Mapper.ba_global_points_ratio', quality.baGlobalRatio,
    '--Mapper.ba_global_max_num_iterations', quality.baGlobalIterations,
    '--Mapper.min_num_matches', quality.minMatches,
    '--Mapper.init_min_tri_angle', quality.initTriAngle,
    '--Mapper.filter_min_tri_angle', '1.0',
    '--Mapper.init_max_forward_motion', '0.99',
  ];

  // --- Global SfM first (faster, and better on loops) ----------------------
  // GLOMAP is now in-tree as COLMAP's `global_mapper`, so there is no separate
  // binary to detect on this build. Global SfM solves all rotations and
  // positions at once instead of growing the model one camera at a time, which
  // on aerial blocks is roughly 3x faster than incremental and does not
  // accumulate drift along the flight line. Incremental remains the fallback:
  // it is still the more accurate solver when the view graph is weak.
  let globalSolved = false;
  const externalGlomap = detectGlomap();
  const globalMapper = externalGlomap.available
    ? { exe: externalGlomap.exe, args: ['mapper'], label: 'GLOMAP' }
    : { exe: commands.colmap, args: ['global_mapper'], label: 'COLMAP global_mapper' };

  if (process.env.SKYFORGE_MAPPER !== 'incremental') {
    try {
      logger.info(`Solving camera poses with ${globalMapper.label} (global SfM)`);
      tracker.start('mapper', `Global SfM (${globalMapper.label})`);
      await run(globalMapper.exe, [
        ...globalMapper.args,
        '--database_path', database,
        '--image_path', frames,
        '--output_path', sparse,
        ...(externalGlomap.available ? [] : ['--GlobalMapper.num_threads', quality.threads]),
      ], work, job, `Global SfM (${globalMapper.label})`, { stage: 'mapper', total: frameCount, timeoutMs: 50 * 60 * 1000 });
      const solved = chooseSparseModel(sparse).images;
      if (solved >= 8) {
        globalSolved = true;
        logger.ok(`${globalMapper.label}: ${solved} cameras solved`);
      } else {
        logger.warn(`${globalMapper.label} registered only ${solved} cameras; retrying with the incremental mapper`);
      }
    } catch (error) {
      if (/cancelled|timeout/i.test(error.message)) throw error;
      logger.warn(`${globalMapper.label} failed (${error.message}); falling back to the incremental mapper`);
    }
    // A partial global solve would confuse the incremental mapper; start clean.
    if (!globalSolved) { try { await rm(sparse, { recursive: true, force: true }); await mkdir(sparse, { recursive: true }); } catch { /* ignore */ } }
  }
  const glomapSolved = globalSolved;
  if (!glomapSolved) {
    // Incremental COLMAP mapper. Capping global BA refinements is the biggest
    // sparse saving; it does not touch the dense settings that decide surface
    // vs. noise. mapperArgs is hoisted above so the self-heal below reuses it.
    try {
      await run(commands.colmap, mapperArgs(), work, job, `Solving camera poses (incremental, ${quality.threads} threads)`, { stage: 'mapper', total: frameCount });
    } catch (error) {
      const message = error.message || String(error);
      if (message.includes('unrecognised option') || message.includes('cancelled') || message.includes('timeout')) throw error;
      if (!message.includes('No good initial image pair') && !message.includes('Failed to create') && !message.includes('failed with code')) throw error;
      await run(commands.colmap, [
        'exhaustive_matcher',
        '--database_path', database,
        ...(job.metrics?.featureBackend === 'aliked+lightglue'
          ? ['--FeatureMatching.type', 'ALIKED_LIGHTGLUE']
          : ['--FeatureMatching.guided_matching', '1']),
        '--FeatureMatching.use_gpu', '1',
        '--FeatureMatching.num_threads', quality.threads,
      ], work, job, 'Retry: exhaustive matching', { stage: 'match', total: frameCount });
      await run(commands.colmap, mapperArgs(), work, job, 'Retry: solving camera poses', { stage: 'mapper', total: frameCount });
    }
  }

  // ---- Stage-4 self-heal: rescue a thin / thrashed reconstruction --------
  // A tiny sparse model means SfM could not bootstrap on a marginal single-pass
  // scene — it kept picking degenerate, near-zero-baseline initial pairs (what
  // happened on the village clip: stuck at num_reg_frames=2-3 with "structure-
  // less fallback"). Newer COLMAP exits 0 with that 2-3 frame model instead of
  // erroring, so the "No good initial image pair" retry inside the catch above
  // never fires. Exhaustive matching hands the mapper every possible pair, which
  // reliably rescues these clips (sky masking upstream removes the ~40% of
  // features that make such scenes marginal in the first place).
  let chosen = chooseSparseModel(sparse);
  const positionFloor = Math.max(12, Math.floor(frameCount * 0.5));
  if (chosen.images < positionFloor && !job.cancelled) {
    logger.warn(`Only ${chosen.images}/${frameCount} frames positioned (below ${positionFloor}); rescuing with exhaustive matching.`);
    try {
      await run(commands.colmap, [
        'exhaustive_matcher',
        '--database_path', database,
        ...(job.metrics?.featureBackend === 'aliked+lightglue'
          ? ['--FeatureMatching.type', 'ALIKED_LIGHTGLUE']
          : ['--FeatureMatching.guided_matching', '1']),
        '--FeatureMatching.use_gpu', '1',
        '--FeatureMatching.num_threads', quality.threads,
      ], work, job, 'Rescue: exhaustive matching (thin reconstruction)', { stage: 'match', total: frameCount });
      await run(commands.colmap, mapperArgs(), work, job, 'Rescue: solving camera poses (exhaustive)', { stage: 'mapper', total: frameCount });
      const rescued = chooseSparseModel(sparse);
      if (rescued.images > chosen.images) {
        chosen = rescued;
        logger.ok(`Exhaustive rescue positioned ${chosen.images}/${frameCount} frames.`);
      }
    } catch (error) {
      if (/cancelled|timeout/i.test(error.message)) throw error;
      logger.warn(`Exhaustive rescue failed: ${error.message}`);
    }
  }
  const sparseModel = chosen.path;
  if (chosen.images && chosen.images < 8) {
    throw new Error(`Only ${chosen.images} of ${frameCount} frames could be positioned. The clip needs more overlap or more texture to reconstruct.`);
  }
  if (chosen.models > 1) {
    job.warning = `${job.warning || ''} Tracking broke into ${chosen.models} separate models; using the largest (${chosen.images} frames).`.trim();
  } else if (chosen.images && chosen.images < frameCount * 0.6) {
    job.warning = `${job.warning || ''} Only ${chosen.images} of ${frameCount} frames were positioned, so coverage is partial.`.trim();
  }
  const sparsePly = join(work, 'sparse_points.ply');
  await run(commands.colmap, [
    'model_converter',
    '--input_path', sparseModel,
    '--output_path', sparsePly,
    '--output_type', 'PLY',
  ], work, job, 'Exporting sparse point cloud', { stage: 'mapper' });
  tracker.complete('mapper');

  // ---- Undistort: shared entry to both dense paths ----------------------
  // Undistort at full frame resolution (undistortSize), not the old depth cap:
  // OpenMVS downsamples internally, but it can only work with the detail the
  // undistorted frames carry.
  let denseCloud = existsSync(sparsePly) ? sparsePly : null;
  let texturedGlbPath = null;   // OpenMVS high-detail textured mesh (GLB)
  let texturedMeshPly = null;   // its triangle mesh (PLY)
  let meshRefined = false;
  await run(commands.colmap, [
    'image_undistorter',
    '--image_path', frames,
    '--input_path', sparseModel,
    '--output_path', dense,
    '--output_type', 'COLMAP',
    '--max_image_size', quality.undistortSize,
  ], work, job, 'Undistorting images for the dense workspace', { stage: 'undistort', total: frameCount });
  tracker.complete('undistort');

  if (useMvs) {
    // ---- Primary path: OpenMVS dense -> mesh -> texture -> GLB ----------
    const mvsDir = join(work, 'mvs');
    await mkdir(mvsDir, { recursive: true });
    try {
      const mvsResult = await buildTexturedMesh({
        run,
        commands: openmvsCommands(commands.openmvs),
        dense,
        mvsDir,
        job,
        tracker,
        mvs: quality.mvs,
        logger,
        imageCount: chosen.images || frameCount,
      });
      tracker.complete('texture');
      if (mvsResult.denseCloud) denseCloud = mvsResult.denseCloud;
      texturedGlbPath = mvsResult.modelGlb;
      texturedMeshPly = mvsResult.meshPly;
      meshRefined = mvsResult.refined;
      logger.ok(`OpenMVS textured mesh ready${meshRefined ? ' (photo-refined)' : ''}`);
    } catch (error) {
      // Only a cancel aborts. A mesh/texture TIMEOUT falls through to the
      // recovery below, which reuses the dense cloud densify already produced —
      // the difference between "no model at all" and "a slightly coarser model".
      if (isCancellation(error)) throw error;
      logger.error(`OpenMVS textured-mesh path failed: ${error.message}`, error);
      job.warning = `${job.warning || ''} High-detail mesh failed (${error.message}); fell back to dense stereo DSM.`.trim();
      // Densify may already have produced a cloud before a later stage failed.
      const mvsDense = join(mvsDir, 'scene_dense.ply');
      if (existsSync(mvsDense) && statSync(mvsDense).size > 1024) {
        denseCloud = mvsDense;
        tracker.skip('mesh', 'mesh/texture failed after densify');
        tracker.skip('texture', 'mesh/texture failed after densify');
      } else {
        tracker.skip('mesh', 'OpenMVS failed');
        tracker.skip('texture', 'OpenMVS failed');
        denseCloud = (await colmapDenseFallback({ dense, work, job, tracker, quality, chosen, frameCount })) || denseCloud;
      }
    }
  } else {
    // ---- Fallback: COLMAP dense stereo (no OpenMVS on this machine) -----
    tracker.skip('mesh', 'OpenMVS unavailable');
    tracker.skip('texture', 'OpenMVS unavailable');
    denseCloud = (await colmapDenseFallback({ dense, work, job, tracker, quality, chosen, frameCount })) || denseCloud;
  }

  if (!denseCloud) throw new Error('No point cloud was produced.');
  logger.metric('dense cloud', denseCloud.split(/[/\\]/).pop());

  // ---- Terrain / DSM products from the dense cloud ----------------------
  tracker.start('products', 'Building terrain products and georeferencing');
  // ---- Scene transform, and the 2.5D DSM only if it was asked for ---------
  // The DSM (terrain.glb / terrain.ply / orthomosaic.png) is a genuine product
  // for elevation and ortho work, but it is a heightfield: one height per XY
  // cell, with steep quads deleted rather than walled, so it has no building
  // facades and reads as a draped sheet. It is NOT the 3D model, and building
  // it unconditionally cost ~2m29s of single-threaded JS on every job purely to
  // obtain the gravity/scale transform. Ask for the transform directly, and
  // produce the DSM only when the operator wants that deliverable.
  const wantDsm = job.wantDsm === true || process.env.SKYFORGE_DSM === '1';
  const altitudeM = normalizeAltitude(job.altitude);
  let productsResult;
  if (wantDsm) {
    logger.step('Building 2.5D terrain products (DSM, orthophoto)');
    productsResult = await buildProducts({
      cloudPath: denseCloud,
      meshPath: denseCloud,
      outDir: products,
      workDir: work,
      altitude: altitudeM,
      sourceName: job.fileName || 'drone-pass.mp4',
      mode: quality.exportMode,
      surface: quality.surface,
    });
  } else {
    const transform = await computeSceneTransform({
      cloudPath: denseCloud,
      meshPath: denseCloud,
      workDir: work,
      altitude: altitudeM,
    });
    logger.ok(
      `Scene aligned from ${transform.cameraPoses} camera poses (${transform.orientation}); `
      + `metric scale ${transform.scale.toFixed(4)} via ${transform.scaleSource}`
      + `${transform.clearance ? ` (median flight clearance ${transform.clearance.toFixed(3)} units = ${altitudeM} m)` : ''}`,
    );
    const cloudProducts = await writeCloudProducts({
      cloudPath: denseCloud,
      outDir: products,
      transform,
      altitude: altitudeM,
      sourceName: job.fileName || 'drone-pass.mp4',
      maxPoints: quality.surface?.points || 400000,
    });
    productsResult = { files: cloudProducts.files, metadata: cloudProducts.metadata, transform };
  }

  // ---- Re-orient the OpenMVS textured mesh and keep it as the primary GLB.
  // OpenMVS writes in COLMAP's arbitrary frame; buildProducts handed back the
  // gravity rotation + metric scale it derived, so one matrix stands the mesh
  // upright and sizes it in metres without rewriting a vertex.
  let modelGlbFile = null;
  if (texturedGlbPath && existsSync(texturedGlbPath)) {
    const raw = await readFile(texturedGlbPath);
    // OpenMVS writes the texture as an external PNG beside the GLB; inline it so
    // the model is a single self-contained, shareable file.
    let oriented = raw;
    try {
      const { buffer, embedded } = embedExternalImages(raw, texturedGlbPath.replace(/[/\\][^/\\]+$/, ''));
      oriented = buffer;
      if (embedded) logger.ok(`Embedded ${embedded} texture image(s) into the GLB`);
    } catch (error) {
      logger.warn(`Could not embed the texture into the GLB (${error.message}); it may need its sidecar PNG.`);
    }
    // NO atlas V flip here. Measured, do not "fix" this again.
    //
    // flipTextureV exists and its docstring argues OpenMVS bakes bottom-left
    // origin UVs that glTF needs inverted. That was true of older OpenMVS; it
    // is NOT true of the 2.4.0 build this project ships, which writes
    // glTF-conformant top-left UVs in its GLB export.
    //
    // Applying the flip anyway sent every face into the unused region of the
    // atlas. Sampled over 44,686 faces at four points each on a real bake:
    //   as written by OpenMVS   97.7% of faces on photography, 2.2% on empty
    //   with the flip applied    0.0% on photography, 100% on empty fill
    // which rendered as a uniform grey blob with no texture at all.
    //
    // tools/validate-glb.mjs reports this directly if it ever regresses.
    try {
      if (productsResult.transform?.rotation) oriented = reorientGlb(oriented, productsResult.transform);
    } catch (error) {
      // A re-orientation bug must not cost us the textured mesh — ship it raw
      // (possibly tilted) rather than falling back to the plain DSM.
      logger.warn(`Could not re-orient the textured GLB (${error.message}); keeping it unaligned.`);
    }
    try {
      modelGlbFile = join(products, 'model.glb');
      await writeFile(modelGlbFile, oriented);
      logger.ok(`Textured 3D model written: model.glb (${(oriented.length / 1e6).toFixed(1)} MB)`);
    } catch (error) {
      logger.error(`Could not write model.glb: ${error.message}`, error);
      modelGlbFile = null;
    }
  } else if (texturedMeshPly && existsSync(texturedMeshPly) && denseCloud && existsSync(denseCloud)) {
    // Primary path: vertex-colour the dense OpenMVS mesh from the coloured point
    // cloud, clean sky/floaters, and orient it to match the DSM. This is the
    // robust, always-photographic model — no fragile texture atlas.
    try {
      const mesh = parsePly(await readFile(texturedMeshPly));
      const cloud = parsePly(await readFile(denseCloud));
      if (mesh.indices?.length && cloud.colors) {
        const { glb, faces, removed } = buildColoredMeshGlb({
          positions: mesh.positions,
          indices: mesh.indices,
          cloudPositions: cloud.positions,
          cloudColors: cloud.colors,
          transform: productsResult.transform,
        });
        modelGlbFile = join(products, 'model.glb');
        await writeFile(modelGlbFile, glb);
        logger.ok(`Vertex-coloured 3D model: model.glb (${faces.toLocaleString()} faces, ${(glb.length / 1e6).toFixed(1)} MB; removed ${removed.toLocaleString()} sky/floater faces)`);
      }
    } catch (error) {
      logger.error(`Could not build the coloured mesh (${error.message}); falling back to the DSM.`, error);
      modelGlbFile = null;
    }
  }
  if (texturedMeshPly && existsSync(texturedMeshPly)) {
    try { await copyFile(texturedMeshPly, join(products, 'mesh.ply')); } catch { /* best effort */ }
  }

  const publicOut = resolve('output/latest');
  await mkdir(publicOut, { recursive: true });
  for (const file of Object.values(productsResult.files)) {
    if (!existsSync(file)) continue;
    await copyFile(file, join(publicOut, file.split(/[/\\]/).pop()));
  }
  if (modelGlbFile) await copyFile(modelGlbFile, join(publicOut, 'model.glb')).catch(() => {});

  // ---- Turntable video ---------------------------------------------------
  // A 360-degree MP4 of whatever model we just produced. This is the artefact
  // that answers "did it work?" without WebGL, a GPU or any knowledge of how
  // to orbit a viewer, and it is the only output that can be dropped into a
  // chat or a slide as-is.
  //
  // Strictly best-effort: by the time we get here the real deliverable is
  // already on disk, so a renderer or encoder failure is a cosmetic gap. It
  // logs a warning and the job completes normally.
  const turntableSource = modelGlbFile || productsResult.files.glb;
  let turntableFile = null;
  let turntablePoster = null;
  if (turntableSource && existsSync(turntableSource) && !job.cancelled) {
    tracker.start('turntable', 'Rendering the 360-degree turntable video');
    const settings = turntableSettings(quality.quality);
    const target = turntablePath(products);
    const outcome = await renderTurntable({
      glbPath: turntableSource,
      outPath: target,
      ffmpeg: commands.ffmpeg,
      elevation: 30,
      ...settings,
      onProgress: ({ fraction, frame, total }) => {
        tracker.fraction(fraction, { kind: 'frame', done: frame, total });
      },
    });
    if (outcome.ok) {
      turntableFile = outcome.path;
      turntablePoster = outcome.posterPath;
      logger.ok(
        `Turntable video: turning.mp4 (${outcome.frames} frames, ${outcome.width}x${outcome.height}, `
        + `${outcome.durationSec.toFixed(1)}s, ${(outcome.bytes / 1e6).toFixed(1)} MB)`,
        outcome.elapsedMs,
      );
      for (const [file, name] of [[turntableFile, 'turning.mp4'], [turntablePoster, 'turning.png']]) {
        if (file && existsSync(file)) await copyFile(file, join(publicOut, name)).catch(() => {});
      }
    } else {
      tracker.skip('turntable', outcome.reason);
      logger.warn(`Could not render the turntable video (${outcome.reason}); the model itself is unaffected.`);
    }
  } else {
    tracker.skip('turntable', 'no model to render');
  }

  tracker.finish();
  // The textured 3D mesh is THE product. The DSM only stands in when it was
  // explicitly requested and the mesh genuinely could not be built — it is a
  // heightfield with no facades, so silently shipping it as "the 3D model" is
  // what made every failed texture bake look like a rotating photo.
  const primaryModel = modelGlbFile || productsResult.files.glb;
  if (!primaryModel) {
    throw new Error(
      'Reconstruction produced no viewable model: the dense mesh could not be built or textured. '
      + 'The run log above records which stage degraded. Re-run with SKYFORGE_DSM=1 to export the '
      + '2.5D height model instead, bearing in mind it has no building facades.',
    );
  }
  if (!modelGlbFile) {
    job.warning = `${job.warning || ''} The 3D mesh could not be built, so the 2.5D height model is standing in. `
      + 'It has no building facades and cannot be inspected at street level.'.trim();
  }
  job.metrics = {
    ...productsResult.metadata,
    profile: quality.quality,
    framesUsed: frameCount,
    threads: Number(quality.threads),
    dsmExported: wantDsm,
    densePath: !modelGlbFile ? 'colmap-dsm'
      : texturedGlbPath ? 'openmvs-textured-atlas-mesh'
        : 'openmvs-vertex-coloured-mesh',
    texturedMesh: Boolean(modelGlbFile),
    meshRefined,
    denseSize: Number(quality.denseSize),
    hardware: {
      tier: hardware.tier,
      gpu: hardware.gpuName,
      vramGb: hardware.vramGb,
      ramGb: hardware.ramGb,
      threads: hardware.workerThreads,
      cacheGb: hardware.cacheGb,
    },
  };
  job.result = {
    pointCloud: productsResult.files.pointCloud,
    model: primaryModel,
    terrain: productsResult.files.glb,
    mesh: productsResult.files.mesh,
    texturedModel: modelGlbFile,
    ortho: productsResult.files.ortho,
    metadata: productsResult.files.metadata,
    // Null when the turntable could not be rendered; the UI falls back to a
    // "generate" button rather than a broken <video>.
    video: turntableFile,
    poster: turntablePoster,
    format: modelGlbFile ? 'GLB textured 3D mesh + DSM terrain + PLY cloud' : 'GLB terrain mesh + PLY point cloud',
    frameCount,
  };
  // The mesh is a finished, downloadable deliverable from this moment. Say so
  // now rather than at the end of the run: the optional splat stage that
  // follows can take longer than everything before it combined, and a model
  // sitting on disk that the console will not hand over for another 25 minutes
  // is indistinguishable, to the operator, from a model that was never built.
  job.modelReady = true;
  logger.metric('result', primaryModel.split(/[/\\]/).pop());

  // ---- Optional photoreal branch: 3D Gaussian Splatting (gsplat) ---------
  // Opt-in per job (the "Photoreal Gaussian splat" switch), because training is
  // a real cost on top of the mesh. It shares the COLMAP poses, so there is no
  // second solve, and it is NON-FATAL: a splat failure never costs the
  // measurable mesh and cloud. Mesh for measurement, splat for looking.
  //
  // This is the branch that answers "I want to zoom into the street and see the
  // front of the houses": a mesh gives one colour per surface point, a splat
  // reproduces what the camera actually saw from any viewpoint.
  const wantSplat = job.wantSplat === true || process.env.SKYFORGE_SPLAT === '1';
  if (wantSplat && !job.cancelled) {
    const splat = detectSplat();
    if (!splat.available) {
      logger.warn(`Gaussian splat requested but the trainer is unavailable: ${splat.reason}. Run \`npm run setup:splat\`.`);
      job.warning = `${job.warning || ''} Gaussian splat was requested but skipped: ${splat.reason}.`.trim();
    } else {
      try {
        const splatDir = join(products, 'splat');
        await mkdir(splatDir, { recursive: true });
        // Cap the Gaussian count to the card. Each Gaussian carries position,
        // scale, rotation, opacity and 48 SH floats plus Adam moments, so an
        // 8 GB card runs out well before a 24 GB one.
        const vramGb = hardware.vramGb || 8;
        const cap = vramGb >= 20 ? 3000000 : vramGb >= 12 ? 1800000 : 900000;
        const iters = quality.quality === 'detailed' ? 20000 : quality.quality === 'balanced' ? 14000 : 8000;
        const maxEdge = Math.min(1600, quality.mvsWidth || 1600);
        logger.info(
          `Training photoreal Gaussian splat on the solved poses — `
          + `${iters.toLocaleString()} iterations, up to ${cap.toLocaleString()} Gaussians, ${maxEdge}px frames`,
        );
        await run(splat.python, [
          splat.script,
          '--work', work,
          '--out', splatDir,
          '--iters', String(iters),
          '--cap', String(cap),
          '--max-size', String(maxEdge),
        ], work, job, 'Training Gaussian splat (gsplat)', { stage: 'products', timeoutMs: 180 * 60 * 1000 });

        const splatPly = await firstExisting([
          join(splatDir, 'splat.ply'),
          join(splatDir, 'point_cloud.ply'),
        ]);
        if (splatPly) {
          job.result.splat = splatPly;
          await copyFile(splatPly, join(publicOut, 'splat.ply')).catch(() => {});
          try {
            const stats = JSON.parse(await readFile(join(splatDir, 'splat.json'), 'utf8'));
            job.metrics.splat = stats;
            logger.ok(`Gaussian splat ready: ${Number(stats.gaussians).toLocaleString()} Gaussians in ${(stats.trainingSeconds / 60).toFixed(1)} min`);
          } catch {
            logger.ok('Gaussian splat ready: splat.ply');
          }
        } else {
          logger.warn('Splat training finished but produced no .ply.');
        }
      } catch (error) {
        if (/cancelled/i.test(error.message)) throw error;
        logger.warn(`Gaussian splat branch failed: ${error.message}`);
        job.warning = `${job.warning || ''} Gaussian splat training failed: ${error.message}`.trim();
      }
    }
  }

  // ---- Honest processing report + self-evaluation ----------------------
  try {
    const report = buildReport(job, job.metrics, job.stages);
    job.report = report;
    await writeFile(join(products, 'report.json'), JSON.stringify(report, null, 2));
    await writeFile(join(products, 'report.md'), reportToMarkdown(report));
    await copyFile(join(products, 'report.json'), join(publicOut, 'report.json')).catch(() => {});
    await copyFile(join(products, 'report.md'), join(publicOut, 'report.md')).catch(() => {});
    job.result.report = join(products, 'report.json');
    logger.ok(`Report: ${report.metrics.framesRegistered}/${report.metrics.framesUsed} frames, ${report.metrics.densePoints.toLocaleString()} points, ${report.processingSeconds}s`);
  } catch (error) {
    logger.warn(`Could not write the processing report: ${error.message}`);
  }
}

export async function exportExistingJob(work, altitude = 82, fileName = 'existing-pass.mp4', mode = 'view') {
  const denseCloud = await firstExisting([
    join(work, 'dense', 'fused.ply'),
    join(work, 'dense', 'scene.ply'),
    join(work, 'sparse_points.ply'),
  ]);
  if (!denseCloud) throw new Error(`No reconstructed cloud found in ${work}`);
  const products = join(work, 'products');
  const result = await buildProducts({
    cloudPath: denseCloud,
    meshPath: existsSync(join(work, 'dense', 'poisson.ply')) ? join(work, 'dense', 'poisson.ply') : denseCloud,
    outDir: products,
    workDir: work,
    altitude: normalizeAltitude(altitude),
    sourceName: fileName,
    mode,
  });
  const publicOut = resolve('output/latest');
  await mkdir(publicOut, { recursive: true });
  for (const file of Object.values(result.files)) {
    if (!existsSync(file)) continue;
    await copyFile(file, join(publicOut, file.split(/[/\\]/).pop()));
  }
  return result;
}
