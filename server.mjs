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
import { sampleTelemetry } from './lib/telemetry.mjs';
import { hardwareFromTelemetry, hardwareLabel } from './lib/hardware.mjs';

const port = Number(process.env.PORT || 8787);

// Anchor every path to this file, not to the shell working directory. The
// previous resolve() calls meant starting the server from anywhere but the
// project root silently served nothing and wrote jobs into the wrong place.
const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, 'runtime');
const dist = join(here, 'dist');
const outputLatest = join(here, 'output', 'latest');
const jobs = new Map();

/**
 * Splat-trainer availability, probed once and cached.
 *
 * detectSplat shells out to Python to check for torch and gsplat, which takes
 * seconds. Calling it inside the /api/health handler blocked the event loop
 * for long enough that the request timed out. Probe lazily on first use, then
 * reuse; a trainer installed while the server is running needs a restart,
 * which is the same as every other engine here.
 */
let splatProbe = null;
const splatSupport = () => (splatProbe ??= detectSplat());
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
};

function cors(response, type = 'application/json') {
  response.setHeader('Access-Control-Allow-Origin', '*');
  // X-Crs is gone: the old UI advertised a coordinate-system picker it never
  // sent and the server never read. Without GPS there is no CRS to pick.
  response.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-File-Name, X-Quality, X-Altitude, X-Frames, X-Threads, X-Splat, X-Dsm, X-Mask-Dynamic');
  response.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
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
  };
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
        metadata: existsSync(metaPath) ? metaPath : null,
        format: glb.endsWith('model.glb') ? 'GLB textured 3D mesh + PLY cloud' : 'GLB terrain mesh + PLY point cloud',
      },
      metrics,
      fileName: metrics?.source || id,
    });
  }
  return listed.sort((a, b) => String(b.metrics?.generatedAt || '').localeCompare(String(a.metrics?.generatedAt || '')));
}

function fileFor(job, kind) {
  const recovered = join(root, job?.id || '', 'products');
  const latest = outputLatest;
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
  };
  return (table[kind] || table.glb).find((path) => path && existsSync(path));
}

function streamFile(response, filePath, type) {
  cors(response, type);
  const name = filePath.split(/[/\\]/).pop();
  const size = statSync(filePath).size;
  response.setHeader('Content-Disposition', `inline; filename="${name}"`);
  response.setHeader('Cache-Control', 'no-store');
  response.setHeader('Content-Length', String(size));
  const stream = createReadStream(filePath);
  stream.on('error', (error) => {
    if (!response.headersSent) send(response, 500, { error: error.message });
  });
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
  streamFile(response, filePath, mime[extname(filePath)] || 'application/octet-stream');
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

async function handle(request, response) {
  if (request.method === 'OPTIONS') return send(response, 204, '');
  const url = new URL(request.url, `http://localhost:${port}`);

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
    });
  }

  if (request.method === 'GET' && url.pathname === '/api/telemetry') {
    return send(response, 200, await sampleTelemetry());
  }

  if (request.method === 'GET' && url.pathname === '/api/jobs') {
    return send(response, 200, await listJobs());
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
      progress: 0,
      stage: 'Queued behind another reconstruction',
      status: 'processing',
      cancelHooks: new Set(),
    };
    jobs.set(id, job);
    queueExclusive(() => reconstruct(job))
      .then(() => { job.status = 'complete'; })
      .catch((error) => {
        job.status = 'error';
        job.error = error.message;
      });
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

  const fileMatch = url.pathname.match(/^\/api\/reconstruct\/([^/]+)\/file$/);
  if (request.method === 'GET' && fileMatch) {
    const kind = url.searchParams.get('kind') || 'glb';
    const job = jobs.get(fileMatch[1]) || { id: fileMatch[1], result: null };
    const filePath = fileFor(job, kind);
    if (!filePath) return send(response, 404, { error: 'Reconstruction output is not available.' });
    streamFile(response, filePath, mime[extname(filePath)] || 'application/octet-stream');
    return;
  }

  if (request.method === 'GET') return serveStatic(request, response);
  return send(response, 404, { error: 'Not found.' });
}

await mkdir(root, { recursive: true });
const server = createServer((request, response) => handle(request, response).catch((error) => send(response, 500, { error: error.message })));
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
