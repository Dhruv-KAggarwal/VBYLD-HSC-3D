import { createServer } from 'node:http';
import { createWriteStream, createReadStream, existsSync, statSync } from 'node:fs';
import { mkdir, readFile, readdir, stat } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { extname, join, resolve, normalize, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from 'node:stream/promises';
import { rm } from 'node:fs/promises';
import {
  commands,
  engineAvailable,
  normalizeAltitude,
  normalizeFrames,
  normalizeQuality,
  normalizeThreads,
  queueExclusive,
  reconstruct,
  terminateAllEngines,
  buildStagePlan,
  cudnnAvailable,
} from './lib/reconstruct.mjs';
import { detectSplat } from './lib/external.mjs';
import { renderTurntable, turntablePath, turntableSettings } from './lib/turntable.mjs';
import { sampleTelemetry } from './lib/telemetry.mjs';
import { hardwareFromTelemetry, hardwareLabel } from './lib/hardware.mjs';
import { createApi, isSafeId, normalizeEnhance, readBody } from './lib/api.mjs';
import { createUploadStore } from './lib/uploads.mjs';
import { createTaskManager } from './lib/tasks.mjs';
import { jobViews } from './lib/views.mjs';
import { readJson, writeJson } from './lib/multivideo/manifest.mjs';

const port = Number(process.env.PORT || 8787);

// Anchor every path to this file, not to the shell working directory. The
// previous resolve() calls meant starting the server from anywhere but the
// project root silently served nothing and wrote jobs into the wrong place.
const here = dirname(fileURLToPath(import.meta.url));
const root = process.env.SKYFORGE_RUNTIME ? resolve(process.env.SKYFORGE_RUNTIME) : join(here, 'runtime');
const dist = join(here, 'dist');
const outputLatest = join(here, 'output', 'latest');
const jobs = new Map();
const uploads = createUploadStore({ root, commands });
const tasks = createTaskManager();

/**
 * Splat-trainer availability, probed once and cached.
 *
 * detectSplat shells out to Python to check for torch and gsplat, and
 * spawnSync blocks the event loop. Probe lazily on first use, then reuse; a trainer installed while the server is running needs a restart,
 * which is the same as every other engine here.
 */
let splatProbe = null;
const splatSupport = () => (splatProbe ??= detectSplat());
/**
 * Job ids whose turntable is being rendered right now.
 *
 * The render is CPU-bound and takes tens of seconds. Without this guard an
 * impatient double-click starts two rasterisers writing the same MP4.
 */
const renderingTurntables = new Set();
/** Open SSE sockets, closed on client disconnect and on process shutdown. */
const liveStreams = new Set();

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.glb': 'model/gltf-binary',
  '.ply': 'application/octet-stream',
  '.svg': 'image/svg+xml',
  // The turntable. Browsers will not play an MP4 served as
  // application/octet-stream, and they will not seek one served without
  // byte-range support — see streamFile.
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.md': 'text/markdown; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm',
};

function cors(response, type = 'application/json') {
  response.setHeader('Access-Control-Allow-Origin', '*');
  // X-Crs is gone: the old UI advertised a coordinate-system picker it never
  // sent and the server never read. Without GPS there is no CRS to pick.
  response.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-File-Name, X-Quality, X-Altitude, X-Frames, X-Threads, X-Splat, X-Dsm, X-Mask-Dynamic, X-Enhance');
  response.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, POST, PATCH, DELETE, OPTIONS');
  // A cross-origin reader needs these two visible to seek a video.
  response.setHeader('Access-Control-Expose-Headers', 'Accept-Ranges, Content-Range, Content-Length, Content-Disposition');
  response.setHeader('Content-Type', type);
}

function send(response, status, body, type = 'application/json') {
  // Without an explicit charset the browser decodes JSON as latin-1 and any
  // non-ASCII character (the middot in the hardware label, for one) arrives
  // as mojibake.
  if (type === 'application/json') type = 'application/json; charset=utf-8';
  cors(response, type);
  response.writeHead(status);
  response.end(type.startsWith('application/json') && typeof body !== 'string' && !Buffer.isBuffer(body) ? JSON.stringify(body) : body);
}

/** Stage list for a job that has not reached the pipeline yet. */
const queuedStages = () => buildStagePlan().map((stage) => ({
  key: stage.key,
  label: stage.label,
  detail: stage.detail,
  weight: Math.round(stage.weight * 10000) / 10000,
  status: 'pending',
  progress: 0,
  startedAt: null,
  endedAt: null,
  durationMs: null,
}));

function jobPayload(job) {
  return {
    id: job.id,
    status: job.status,
    progress: job.progress,
    stage: job.stage,
    stageKey: job.stageKey ?? null,
    stageIndex: job.stageIndex ?? -1,
    stageCount: job.stageCount ?? null,
    stageProgress: job.stageProgress ?? 0,
    stageDetail: job.stageDetail ?? null,
    stages: job.stages || queuedStages(),
    startedAt: job.startedAt ?? null,
    elapsedMs: job.elapsedMs ?? null,
    etaMs: job.status === 'processing' ? (job.etaMs ?? null) : 0,
    cancelled: Boolean(job.cancelled),
    // How long since ANY engine wrote a line. This is the stall signal: the
    // progress bar alone cannot distinguish "this stage is slow" from "this
    // process died", and some stages legitimately go minutes without output.
    idleMs: job.lastOutputAt ? Date.now() - job.lastOutputAt : null,
    logSeq: job.logSeq ?? 0,
    // True once the mesh is on disk, which happens before the optional splat
    // stage. The UI uses this to release the viewer and downloads early.
    modelReady: Boolean(job.modelReady),
    warning: job.warning,
    error: job.error,
    result: job.result,
    metrics: job.metrics,
    fileName: job.fileName,
    mode: job.mode || 'single',
    videoCount: job.videoCount ?? null,
  };
}

/** What survives a restart about a job, beside its products. */
async function saveJobRecord(job) {
  await writeJson(join(job.work, 'job.json'), {
    id: job.id,
    mode: job.mode || 'single',
    fileName: job.fileName,
    videoCount: job.videoCount ?? 1,
    createdAt: job.createdAt,
    finishedAt: job.status === 'processing' ? null : new Date().toISOString(),
    status: job.status,
    quality: job.quality,
    altitude: job.altitude,
    maxFrames: job.maxFrames ?? job.frames ?? null,
    matching: job.matching || null,
    splat: Boolean(job.wantSplat),
    dsm: Boolean(job.wantDsm),
    maskDynamic: Boolean(job.maskDynamic),
    enhance: job.enhance || null,
    warning: job.warning || null,
    error: job.error || null,
  }).catch(() => {});
}

function startJob(job) {
  job.createdAt = new Date().toISOString();
  jobs.set(job.id, job);
  saveJobRecord(job);
  queueExclusive(() => reconstruct(job))
    .then(() => { job.status = 'complete'; })
    .catch((error) => {
      job.status = 'error';
      job.error = error.message;
      job.peaks?.stop();
    })
    .finally(() => saveJobRecord(job));
}

async function listJobs() {
  const listed = [...jobs.values()].map(jobPayload);
  if (!existsSync(root)) return listed;
  const folders = await readdir(root);
  for (const id of folders) {
    if (listed.some((job) => job.id === id)) continue;
    // Recover by the HEADLINE product, not the DSM.
    //
    // This looked only for terrain.glb, which was fine when the 2.5D height
    // model was built on every run. It is now opt-in, so a perfectly good
    // textured mesh on disk made the job invisible here: the library showed
    // nothing and the operator was told the model "was not saved".
    const products = join(root, id, 'products');
    const glb = [
      join(products, 'model.glb'),
      join(products, 'terrain.glb'),
    ].find((path) => existsSync(path));
    const metaPath = join(products, 'metadata.json');
    if (!glb) continue;
    const pick = (...names) => names.map((name) => join(products, name)).find((path) => existsSync(path)) || null;
    let metrics = null;
    try { metrics = JSON.parse(await readFile(metaPath, 'utf8')); } catch { /* ignore */ }
    const record = await readJson(join(root, id, 'job.json'));
    listed.push({
      id,
      status: 'complete',
      progress: 100,
      stage: 'Complete',
      stageKey: null,
      stageIndex: -1,
      stageProgress: 100,
      stageDetail: null,
      stages: queuedStages().map((stage) => ({ ...stage, status: 'complete', progress: 100 })),
      startedAt: null,
      elapsedMs: null,
      etaMs: 0,
      result: {
        model: glb,
        texturedModel: pick('model.glb'),
        terrain: pick('terrain.glb'),
        pointCloud: pick('pointcloud.ply'),
        mesh: pick('mesh.ply', 'terrain.ply'),
        splat: pick('splat/splat.ply'),
        ortho: pick('orthomosaic.png'),
        // Old jobs predate the turntable stage, so this is usually null and
        // the UI offers to render one on demand.
        video: pick('turning.mp4'),
        poster: pick('turning.png'),
        metadata: existsSync(metaPath) ? metaPath : null,
        format: glb.endsWith('model.glb') ? 'GLB textured 3D mesh + PLY cloud' : 'GLB terrain mesh + PLY point cloud',
      },
      metrics,
      fileName: record?.fileName || metrics?.source || id,
      mode: record?.mode || 'single',
      videoCount: record?.videoCount ?? null,
      warning: record?.warning || undefined,
    });
  }
  return listed.sort((a, b) => String(b.metrics?.generatedAt || '').localeCompare(String(a.metrics?.generatedAt || '')));
}

function fileFor(job, kind) {
  const recovered = join(root, job?.id || '', 'products');
  // Never let one job's download resolve to another job's file. The
  // output/latest fallback exists for the pseudo-job that has no runtime
  // directory at all; a real job that simply did not build a DSM or an
  // orthophoto was being handed the previous run's copy instead, which the
  // dashboard then presented as this reconstruction's output.
  const latest = existsSync(recovered) ? recovered : outputLatest;
  const table = {
    // Prefer the full-resolution photo-coloured surface when a job has one.
    // terrain.glb is the lighter DSM and is still available as kind=terrain.
    glb: [join(recovered, 'model.glb'), job?.result?.texturedModel, job?.result?.model, join(recovered, 'terrain.glb'), join(latest, 'terrain.glb')],
    // The Gaussian splat is optional; the viewer probes for it and quietly
    // skips the layer when a job did not train one.
    splat: [job?.result?.splat, join(recovered, 'splat', 'splat.ply'), join(latest, 'splat.ply')],
    terrain: [job?.result?.terrain, join(recovered, 'terrain.glb'), join(latest, 'terrain.glb')],
    full: [join(recovered, 'terrain-full.glb'), join(latest, 'terrain-full.glb'), join(recovered, 'terrain.glb')],
    cloud: [job?.result?.pointCloud, join(recovered, 'pointcloud.ply'), join(latest, 'pointcloud.ply')],
    mesh: [job?.result?.mesh, join(recovered, 'terrain.ply'), join(latest, 'terrain.ply')],
    ortho: [job?.result?.ortho, join(recovered, 'orthomosaic.png'), join(latest, 'orthomosaic.png')],
    meta: [job?.result?.metadata, join(recovered, 'metadata.json'), join(latest, 'metadata.json')],
    // The 360-degree turntable MP4 and its poster still. `turning` is accepted
    // as an alias because that is the filename, and it is what anyone reading
    // the products directory will reach for.
    video: [job?.result?.video, join(recovered, 'turning.mp4'), join(latest, 'turning.mp4')],
    poster: [job?.result?.poster, join(recovered, 'turning.png'), join(latest, 'turning.png')],
    report: [join(recovered, 'report.md')],
    // Enhancement products belong to one job; none of them falls back.
    original: [join(recovered, 'model.glb')],
    repaired: [join(recovered, 'model-repaired.glb')],
    completed: [join(recovered, 'model-completed.glb')],
    removed: [join(recovered, 'repair', 'removed.glb')],
    'repair-report': [join(recovered, 'repair', 'repair-report.json')],
    'confidence-glb': [join(recovered, 'confidence', 'confidence.glb')],
    'confidence-ply': [join(recovered, 'confidence', 'confidence.ply')],
    'confidence-bin': [join(recovered, 'confidence', 'confidence.bin')],
    'confidence-report': [join(recovered, 'confidence', 'confidence-report.md')],
    'lod-report': [join(recovered, 'reports', 'lod-report.json')],
  };
  table.turning = table.video;
  return (table[kind] || table.glb).find((path) => path && existsSync(path));
}

/** Every kind fileFor understands, for the downloads list and HEAD probes. */
const FILE_KINDS = ['glb', 'splat', 'terrain', 'full', 'cloud', 'mesh', 'ortho', 'meta', 'video', 'poster',
  'report', 'original', 'repaired', 'completed', 'removed', 'repair-report', 'confidence-glb', 'confidence-ply', 'confidence-bin', 'confidence-report', 'lod-report'];

/**
 * Serve a file, honouring HTTP byte ranges.
 *
 * Range support is not an optimisation here, it is a requirement: a <video>
 * element will not let the user seek — and Safari will not play at all —
 * unless the server advertises `Accept-Ranges` and answers a `Range` request
 * with 206 and the exact slice asked for. The same path serves GLBs and PLYs,
 * which only benefits from resumable downloads.
 */
function streamFile(response, filePath, type, request = null) {
  const name = filePath.split(/[/\\]/).pop();
  const size = statSync(filePath).size;
  const rangeHeader = request?.headers?.range;

  const head = (status, headers) => {
    cors(response, type);
    response.setHeader('Content-Disposition', `inline; filename="${name}"`);
    response.setHeader('Accept-Ranges', 'bytes');
    // Products are rewritten in place when a job is re-run, so a cached copy
    // is a stale copy. The UI also cache-busts, but belt and braces.
    response.setHeader('Cache-Control', 'no-store');
    for (const [key, value] of Object.entries(headers)) response.setHeader(key, String(value));
    response.writeHead(status);
  };

  // A HEAD request is how the UI asks "does this file exist and how big is
  // it?" without pulling a 40 MB video down to find out.
  if (request?.method === 'HEAD') {
    head(200, { 'Content-Length': size });
    response.end();
    return;
  }

  let start = 0;
  let end = size - 1;
  if (rangeHeader) {
    // Only the single-range form is implemented; multipart ranges are not
    // something any browser asks a video server for.
    const match = /^bytes=(\d*)-(\d*)$/.exec(String(rangeHeader).trim());
    if (!match || (match[1] === '' && match[2] === '')) {
      head(416, { 'Content-Range': `bytes */${size}` });
      response.end();
      return;
    }
    if (match[1] === '') {
      // Suffix form: the last N bytes.
      const length = Number(match[2]);
      start = Math.max(0, size - length);
    } else {
      start = Number(match[1]);
      if (match[2] !== '') end = Math.min(end, Number(match[2]));
    }
    if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) {
      head(416, { 'Content-Range': `bytes */${size}` });
      response.end();
      return;
    }
    head(206, {
      'Content-Range': `bytes ${start}-${end}/${size}`,
      'Content-Length': end - start + 1,
    });
  } else {
    head(200, { 'Content-Length': size });
  }

  const stream = createReadStream(filePath, { start, end });
  stream.on('error', () => { response.destroy(); });
  response.on('close', () => stream.destroy());
  stream.pipe(response);
}

async function serveStatic(request, response) {
  const url = new URL(request.url, `http://localhost:${port}`);
  let pathname = decodeURIComponent(url.pathname);
  if (pathname === '/') pathname = '/index.html';
  const filePath = resolve(dist, normalize(pathname.replace(/^[/\\]+/, '')));
  const escaped = relative(dist, filePath);
  if (!escaped || escaped.startsWith('..') || escaped.startsWith('/') || !existsSync(filePath)) {
    return send(response, 404, { error: 'Not found.' });
  }
  const info = await stat(filePath);
  if (info.isDirectory()) return send(response, 404, { error: 'Not found.' });
  streamFile(response, filePath, mime[extname(filePath)] || 'application/octet-stream', request);
}

function writeSse(response, event, data) {
  response.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

async function openJobStream(id, request, response) {
  cors(response, 'text/event-stream; charset=utf-8');
  response.writeHead(200, {
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  if (typeof response.flushHeaders === 'function') response.flushHeaders();

  let closed = false;
  const client = { response, id };
  liveStreams.add(client);
  const close = () => {
    if (closed) return;
    closed = true;
    clearInterval(timer);
    liveStreams.delete(client);
    try { response.end(); } catch { /* already gone */ }
  };
  request.on('close', close);

  // Per-socket cursor into the job log, so each client is sent only the lines
  // it has not seen rather than the whole buffer every second.
  let sentSeq = 0;
  const pumpLog = (live) => {
    if (!live?.log?.length) return;
    const fresh = live.log.filter((line) => line.seq > sentSeq);
    if (!fresh.length) return;
    sentSeq = fresh[fresh.length - 1].seq;
    // Chunked: one SSE frame per batch keeps the payload bounded when a noisy
    // stage emits hundreds of lines between polls.
    for (let i = 0; i < fresh.length; i += 120) {
      writeSse(response, 'log', { lines: fresh.slice(i, i + 120) });
    }
  };

  const pump = async () => {
    if (closed) return;
    const telemetry = await sampleTelemetry();
    const live = jobs.get(id);
    if (live) {
      pumpLog(live);
      writeSse(response, 'progress', { ...jobPayload(live), telemetry });
      if (live.status !== 'processing') {
        writeSse(response, 'done', { ...jobPayload(live), telemetry });
        close();
      }
      return;
    }
    const listed = (await listJobs()).find((job) => job.id === id);
    if (listed) {
      writeSse(response, 'done', { ...listed, telemetry });
      close();
      return;
    }
    writeSse(response, 'error', { error: 'Job not found.', telemetry });
    close();
  };

  const timer = setInterval(() => { pump().catch(() => close()); }, 1000);
  await pump().catch(() => close());
}

/** `X-Enhance` carries URI-encoded JSON, same shape as the multi-video options. */
function enhanceFromHeader(value) {
  if (!value) return normalizeEnhance();
  try { return normalizeEnhance(JSON.parse(decodeURIComponent(String(value)))); } catch { return normalizeEnhance(); }
}

/** Progress of the enhancement task running against a job. */
async function openTaskStream(id, request, response) {
  cors(response, 'text/event-stream; charset=utf-8');
  response.writeHead(200, {
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  response.flushHeaders?.();
  const client = { response, id };
  liveStreams.add(client);
  let closed = false;
  let timer = null;
  const close = () => {
    if (closed) return;
    closed = true;
    clearInterval(timer);
    liveStreams.delete(client);
    try { response.end(); } catch { /* already gone */ }
  };
  request.on('close', close);
  const pump = () => {
    if (closed) return;
    const task = tasks.get(id);
    if (!task) {
      writeSse(response, 'done', { task: null });
      close();
      return;
    }
    const finished = task.status !== 'queued' && task.status !== 'running';
    writeSse(response, finished ? 'done' : 'progress', { task });
    if (finished) close();
  };
  timer = setInterval(pump, 750);
  pump();
}

const api = createApi({
  root,
  uploads,
  tasks,
  jobs,
  send,
  streamFile,
  commands,
  openTaskStream,
  hardware: async () => hardwareFromTelemetry(await sampleTelemetry()),
});

async function handle(request, response) {
  if (request.method === 'OPTIONS') return send(response, 204, '');
  const url = new URL(request.url, `http://localhost:${port}`);
  const idMatch = url.pathname.match(/^\/api\/reconstruct\/([^/]+)/);
  if (idMatch && !isSafeId(idMatch[1])) return send(response, 400, { error: 'Invalid job id.' });
  if (url.pathname.startsWith('/api/') && await api(request, response, url)) return undefined;

  if (request.method === 'GET' && url.pathname === '/api/health') {
    const [ffmpeg, colmap, telemetry] = await Promise.all([
      engineAvailable(commands.ffmpeg),
      engineAvailable(commands.colmap),
      sampleTelemetry(),
    ]);
    const splat = splatSupport();
    const running = [...jobs.values()].filter((job) => job.status === 'processing');
    return send(response, 200, {
      // Report what the GPU actually is, not whether an env var happens to be
      // unset — the old check was true on any machine with no GPU at all.
      gpu: Boolean(telemetry?.gpu?.available),
      gpuName: telemetry?.gpu?.name || null,
      ffmpeg,
      colmap,
      openmvs: existsSync(commands.openmvs),
      cudnn: cudnnAvailable(),
      splat: splat.available,
      splatReason: splat.available ? null : splat.reason,
      latest: existsSync(join(outputLatest, 'model.glb')) || existsSync(join(outputLatest, 'terrain.glb')),
      ready: ffmpeg && colmap,
      activeJobs: running.length,
      telemetry,
      hardware: hardwareFromTelemetry(telemetry),
      hardwareLabel: hardwareLabel(hardwareFromTelemetry(telemetry)),
      uploadLimits: uploads.limits(),
    });
  }

  if (request.method === 'GET' && url.pathname === '/api/telemetry') {
    return send(response, 200, await sampleTelemetry());
  }

  if (request.method === 'GET' && url.pathname === '/api/jobs') {
    return send(response, 200, await listJobs());
  }

  // Multi-video: the videos are already staged, so the body is JSON options.
  if (request.method === 'POST' && url.pathname === '/api/reconstruct'
    && String(request.headers['content-type'] || '').startsWith('application/json')) {
    for (const [name, binary] of [['FFmpeg', commands.ffmpeg], ['COLMAP', commands.colmap]]) {
      if (!existsSync(binary)) return send(response, 503, { error: `${name} is not installed at ${binary}.` });
    }
    const body = await readBody(request);
    const options = body.options || {};
    if (typeof body.uploadId !== 'string') return send(response, 400, { error: 'uploadId is required.' });
    const id = randomUUID();
    const work = join(root, id);
    let session;
    try {
      await mkdir(work, { recursive: true });
      session = await uploads.claim(body.uploadId, work);
    } catch (error) {
      await rm(work, { recursive: true, force: true }).catch(() => {});
      throw error;
    }
    const enabled = session.videos.filter((v) => v.settings.enabled);
    const maxFrames = Number(options.frames);
    const job = {
      id,
      work,
      mode: 'multi',
      videoCount: enabled.length,
      fileName: enabled.length === 1 ? enabled[0].originalName : `${enabled.length} videos: ${enabled[0].originalName} …`,
      quality: normalizeQuality(options.quality),
      maxFrames: Number.isFinite(maxFrames) && maxFrames > 0 ? Math.min(2000, Math.max(24, Math.round(maxFrames))) : null,
      frames: null,
      threads: options.threads ? normalizeThreads(options.threads) : null,
      altitude: normalizeAltitude(options.altitude),
      matching: options.matching === 'exhaustive' ? 'exhaustive' : 'auto',
      separateComponents: options.separateComponents !== false,
      wantSplat: options.splat === true,
      wantDsm: options.dsm === true,
      maskDynamic: options.maskDynamic === true,
      enhance: normalizeEnhance(options.enhance),
      progress: 0,
      stage: 'Queued behind another reconstruction',
      status: 'processing',
      cancelHooks: new Set(),
    };
    startJob(job);
    return send(response, 202, { id });
  }

  if (request.method === 'POST' && url.pathname === '/api/reconstruct') {
    const chunksHint = Number(request.headers['content-length'] || 0);
    if (chunksHint > 3.5 * 1024 * 1024 * 1024) return send(response, 413, { error: 'Video is larger than 3.5 GB.' });

    // Fail before the upload rather than after it: a multi-gigabyte transfer
    // that dies on a missing binary wastes the operator's time for nothing.
    for (const [name, binary] of [['FFmpeg', commands.ffmpeg], ['COLMAP', commands.colmap]]) {
      if (!existsSync(binary)) {
        return send(response, 503, { error: `${name} is not installed at ${binary}.` });
      }
    }

    const id = randomUUID();
    const work = join(root, id);
    await mkdir(work, { recursive: true });
    const fileName = String(request.headers['x-file-name'] || 'capture.mp4').replace(/[/\\]/g, '_');
    const input = join(work, `input${extname(fileName) || '.mp4'}`);
    try {
      await pipeline(request, createWriteStream(input));
    } catch (error) {
      await rm(work, { recursive: true, force: true }).catch(() => {});
      return send(response, 400, { error: `Upload failed: ${error.message}` });
    }
    if (!existsSync(input) || statSync(input).size < 64 * 1024) {
      await rm(work, { recursive: true, force: true }).catch(() => {});
      return send(response, 400, { error: 'The uploaded video is empty or truncated.' });
    }

    const job = {
      id,
      work,
      input,
      fileName,
      quality: normalizeQuality(request.headers['x-quality']),
      frames: normalizeFrames(request.headers['x-frames']),
      threads: request.headers['x-threads'] ? normalizeThreads(request.headers['x-threads']) : null,
      altitude: normalizeAltitude(request.headers['x-altitude']),
      // Opt-in stages. The splat is the photoreal street-level branch and the
      // DSM is the 2.5D heightfield; both cost real time, so neither runs
      // unless the operator asked for it.
      wantSplat: request.headers['x-splat'] === '1',
      wantDsm: request.headers['x-dsm'] === '1',
      maskDynamic: request.headers['x-mask-dynamic'] === '1',
      // Optional and absent by default, so existing clients run exactly the
      // pipeline they always did.
      enhance: enhanceFromHeader(request.headers['x-enhance']),
      progress: 0,
      stage: 'Queued behind another reconstruction',
      status: 'processing',
      cancelHooks: new Set(),
    };
    startJob(job);
    return send(response, 202, { id });
  }

  const cancelMatch = url.pathname.match(/^\/api\/reconstruct\/([^/]+)\/cancel$/);
  if (request.method === 'POST' && cancelMatch) {
    const job = jobs.get(cancelMatch[1]);
    if (!job) return send(response, 404, { error: 'Job not found.' });
    if (job.status !== 'processing') return send(response, 409, { error: `Job is already ${job.status}.` });
    job.cancelled = true;
    for (const hook of job.cancelHooks || []) hook();
    return send(response, 200, { id: job.id, status: 'cancelling' });
  }

  const streamMatch = url.pathname.match(/^\/api\/reconstruct\/([^/]+)\/stream$/);
  if (request.method === 'GET' && streamMatch) {
    await openJobStream(streamMatch[1], request, response);
    return;
  }

  // Polling fallback for the log, used when SSE is unavailable (a proxy that
  // buffers event streams, for instance). `since` is the last sequence number
  // the client holds.
  const logMatch = url.pathname.match(/^\/api\/reconstruct\/([^/]+)\/log$/);
  if (request.method === 'GET' && logMatch) {
    const job = jobs.get(logMatch[1]);
    if (!job) return send(response, 404, { error: 'Job not found.' });
    const since = Number(url.searchParams.get('since') || 0);
    const lines = (job.log || []).filter((line) => line.seq > since);
    return send(response, 200, { lines, logSeq: job.logSeq ?? 0, idleMs: job.lastOutputAt ? Date.now() - job.lastOutputAt : null });
  }

  const match = url.pathname.match(/^\/api\/reconstruct\/([^/]+)$/);
  if (request.method === 'GET' && match) {
    const live = jobs.get(match[1]);
    if (live) return send(response, 200, { ...jobPayload(live), telemetry: await sampleTelemetry() });
    const listed = (await listJobs()).find((job) => job.id === match[1]);
    return listed ? send(response, 200, listed) : send(response, 404, { error: 'Job not found.' });
  }

  // Render a turntable for a job that has not got one — an older job from
  // before this stage existed, or one whose render failed. Synchronous on
  // purpose: it is a bounded CPU job of tens of seconds, and the caller wants
  // the video, not a second job to poll.
  const videoMatch = url.pathname.match(/^\/api\/reconstruct\/([^/]+)\/turntable$/);
  if (request.method === 'POST' && videoMatch) {
    const id = videoMatch[1];
    const job = jobs.get(id) || (await listJobs()).find((entry) => entry.id === id) || { id, result: null };
    if (job.status === 'processing') {
      return send(response, 409, { error: 'This job is still running; its turntable is rendered automatically.' });
    }
    const source = fileFor(job, 'glb');
    if (!source) return send(response, 404, { error: 'No model to render a turntable from.' });
    if (renderingTurntables.has(id)) return send(response, 409, { error: 'A turntable render is already in progress.' });
    // A recovered job keeps its products in runtime/<id>/products; a model
    // served out of output/latest has no job directory, so write beside it.
    const products = join(root, id, 'products');
    const outDir = existsSync(products) ? products : outputLatest;
    renderingTurntables.add(id);
    try {
      const filmed = existsSync(products) ? await jobViews(join(root, id)).catch(() => null) : null;
      const outcome = await renderTurntable({
        focus: filmed?.focus,
        glbPath: source,
        outPath: turntablePath(outDir),
        ffmpeg: commands.ffmpeg,
        ...turntableSettings(job.metrics?.profile || 'balanced'),
        elevation: 30,
      });
      if (!outcome.ok) return send(response, 500, { error: `Turntable render failed: ${outcome.reason}` });
      if (job.result) {
        job.result.video = outcome.path;
        job.result.poster = outcome.posterPath;
      }
      return send(response, 200, {
        id,
        video: `/api/reconstruct/${id}/file?kind=video`,
        frames: outcome.frames,
        bytes: outcome.bytes,
        durationSec: outcome.durationSec,
        elapsedMs: outcome.elapsedMs,
      });
    } finally {
      renderingTurntables.delete(id);
    }
  }

  const fileMatch = url.pathname.match(/^\/api\/reconstruct\/([^/]+)\/file$/);
  if ((request.method === 'GET' || request.method === 'HEAD') && fileMatch) {
    const kind = url.searchParams.get('kind') || 'glb';
    const job = jobs.get(fileMatch[1]) || { id: fileMatch[1], result: null };
    const filePath = fileFor(job, kind);
    if (!filePath) {
      // A HEAD with a JSON body is a protocol violation and confuses fetch();
      // the status alone is the answer the UI is probing for.
      if (request.method === 'HEAD') { cors(response); response.writeHead(404); response.end(); return; }
      return send(response, 404, { error: 'Reconstruction output is not available.' });
    }
    streamFile(response, filePath, mime[extname(filePath)] || 'application/octet-stream', request);
    return;
  }

  // Which outputs this job actually has, so the downloads menu can show only
  // real files instead of a row of links that 404.
  const filesMatch = url.pathname.match(/^\/api\/reconstruct\/([^/]+)\/files$/);
  if (request.method === 'GET' && filesMatch) {
    const job = jobs.get(filesMatch[1]) || (await listJobs()).find((entry) => entry.id === filesMatch[1]) || { id: filesMatch[1], result: null };
    const available = {};
    for (const kind of FILE_KINDS) {
      const path = fileFor(job, kind);
      if (!path) continue;
      let bytes = null;
      try { bytes = statSync(path).size; } catch { /* vanished between calls */ }
      available[kind] = { name: path.split(/[/\\]/).pop(), bytes };
    }
    return send(response, 200, { id: job.id, files: available });
  }

  if (request.method === 'GET' || request.method === 'HEAD') return serveStatic(request, response);
  return send(response, 404, { error: 'Not found.' });
}

await mkdir(root, { recursive: true });
await uploads.cleanupStale().catch(() => {});
setInterval(() => { uploads.cleanupStale().catch(() => {}); }, 3600 * 1000).unref();
const server = createServer((request, response) => handle(request, response).catch((error) => {
  if (response.headersSent) { response.destroy(); return; }
  send(response, error.status || 500, { error: error.message });
}));
// Multi-gigabyte uploads legitimately take longer than Node's default.
server.requestTimeout = 0;
server.on('error', (error) => {
  if (error.code === 'EADDRINUSE') {
    console.error(`Skyforge API is already running on http://localhost:${port}.`);
    process.exitCode = 0;
    return;
  }
  console.error(error);
  process.exitCode = 1;
});
server.timeout = 0;
server.listen(port, () => {
  console.log(`Skyforge reconstruction console: http://localhost:${port}`);
});

// Without this the engines survive Ctrl+C, keep the GPU busy, and corrupt the
// dense workspace of whichever run starts next.
let shuttingDown = false;
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(signal, () => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`\nStopping engines and shutting down (${signal})...`);
    tasks.cancelAll();
    for (const job of jobs.values()) {
      if (job.status !== 'processing') continue;
      job.cancelled = true;
      for (const hook of job.cancelHooks || []) hook();
    }
    terminateAllEngines();
    for (const client of liveStreams) {
      try { client.response.end(); } catch { /* already closed */ }
    }
    liveStreams.clear();
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 4000).unref();
  });
}
process.on('exit', terminateAllEngines);
