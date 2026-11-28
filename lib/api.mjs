/**
 * HTTP routes for multi-video uploads and the enhancement products.
 * The core job routes stay in server.mjs; this module returns `false` for any
 * request it does not own.
 */

import { existsSync, statSync } from 'node:fs';
import { join } from 'node:path';
import {
  activeModel, allStates, artifactState, completedModelPath, originalModelPath, productsDir, repairedModelPath, setActiveModel,
} from './artifacts.mjs';
import { connectivityPath, framesPath, inputsPath, readJson, registrationPath } from './multivideo/manifest.mjs';
import { estimateRequirements } from './uploads.mjs';
import { jobViews } from './views.mjs';
import { normalizeRepairSettings } from './meshrepair/repair.mjs';
import { PROVENANCE_INFO } from './meshrepair/provenance.mjs';
import { normalizeConfidenceSettings } from './confidence/analyze.mjs';
import { BANDS, DEFAULT_WEIGHTS, PROVENANCE_CAPS } from './confidence/scoring.mjs';
import { PALETTES } from './confidence/export.mjs';
import { compressionSupport, normalizeLodSettings } from './lod/settings.mjs';

/** Job ids become path segments, so only plain names are accepted. */
export const isSafeId = (id) => /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(id) && !id.includes('..') && id !== 'uploads';

export async function readBody(request, limit = 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > limit) throw Object.assign(new Error('Request body is too large.'), { status: 413 });
    chunks.push(chunk);
  }
  if (!size) return {};
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return value && typeof value === 'object' ? value : {};
  } catch {
    throw Object.assign(new Error('The request body is not valid JSON.'), { status: 400 });
  }
}

/** Which enhancement stages a new reconstruction should run, with settings. */
export function normalizeEnhance(raw = {}) {
  const enhance = {
    analysis: raw.analysis === true,
    confidence: raw.confidence === true,
    repair: raw.repair === true,
    lod: raw.lod === true,
    completion: raw.completion === true,
  };
  if (enhance.repair) enhance.repairSettings = normalizeRepairSettings(raw.repairSettings || {});
  if (enhance.confidence) enhance.confidenceSettings = normalizeConfidenceSettings(raw.confidenceSettings || {});
  if (enhance.lod) enhance.lodSettings = normalizeLodSettings(raw.lodSettings || {});
  return enhance;
}

const OPERATIONS = {
  'POST repair': { operation: 'repair', settings: normalizeRepairSettings },
  'POST confidence/analyze': { operation: 'confidence', settings: normalizeConfidenceSettings },
  'POST lod/generate': { operation: 'lod', settings: normalizeLodSettings },
  'POST completion': { operation: 'completion', settings: () => ({}) },
  'POST mesh-quality': { operation: 'analyze', settings: () => ({}) },
};

export function createApi({ root, uploads, tasks, jobs, send, streamFile, commands, hardware, openTaskStream }) {
  const stale = (response, state, what) => send(response, state.exists ? 409 : 404, {
    error: state.exists ? `${what} is out of date: ${state.reason}. Generate it again.` : `${what} has not been generated for this job.`,
    stale: state.exists,
    reason: state.reason,
  });
  const file = (request, response, path, type = 'application/octet-stream') => {
    if (!path || !existsSync(path)) {
      send(response, 404, { error: 'That output does not exist for this job.' });
      return;
    }
    streamFile(response, path, type, request);
  };

  async function uploadRoutes(request, response, url) {
    const match = url.pathname.match(/^\/api\/uploads(?:\/([^/]+)(?:\/(videos)(?:\/([^/]+)(?:\/(thumbnail))?)?)?)?$/);
    if (!match) return false;
    const [, id, videos, videoId, thumbnail] = match;
    const method = request.method;
    const describe = async (session) => ({
      ...session,
      limits: uploads.limits(),
      estimate: estimateRequirements(session.videos, {
        quality: ['fast', 'balanced', 'detailed'].includes(url.searchParams.get('quality')) ? url.searchParams.get('quality') : 'balanced',
        hardware: await hardware(),
        maxFrames: url.searchParams.get('frames'),
      }),
    });
    if (!id) {
      if (method !== 'POST') return false;
      await uploads.cleanupStale().catch(() => {});
      send(response, 201, await describe(await uploads.create()));
      return true;
    }
    if (!videos) {
      if (method === 'GET') send(response, 200, await describe(await uploads.get(id)));
      else if (method === 'PATCH') send(response, 200, await describe(await uploads.update(id, await readBody(request))));
      else if (method === 'DELETE') { await uploads.remove(id); send(response, 200, { id, removed: true }); } else return false;
      return true;
    }
    if (!videoId) {
      if (method !== 'POST') return false;
      let fileName = String(request.headers['x-file-name'] || url.searchParams.get('name') || 'video.mp4');
      try { fileName = decodeURIComponent(fileName); } catch { /* keep as sent */ }
      const entry = await uploads.receive(id, request, { fileName, replace: url.searchParams.get('replace') });
      send(response, 201, entry);
      return true;
    }
    if (thumbnail) {
      file(request, response, uploads.thumbnailPath(id, videoId), 'image/jpeg');
      return true;
    }
    if (method === 'DELETE') {
      send(response, 200, await describe(await uploads.removeVideo(id, videoId)));
      return true;
    }
    return false;
  }

  async function confidenceFor(work, requested) {
    const which = requested === 'original' || requested === 'repaired' ? requested : (await activeModel(work)).which;
    const state = await artifactState(work, which === 'repaired' ? 'confidenceRepaired' : 'confidence');
    const dir = join(productsDir(work), 'confidence');
    const names = which === 'repaired'
      ? { report: 'repaired.json', data: 'repaired.bin', regions: 'repaired-regions.json' }
      : { report: 'confidence.json', data: 'confidence.bin', regions: 'low-confidence-regions.json' };
    return { which, state, dir, names };
  }

  async function jobRoutes(request, response, url) {
    const match = url.pathname.match(/^\/api\/reconstruct\/([^/]+)\/(.+)$/);
    if (!match) return false;
    const [, id, rest] = match;
    const method = request.method === 'HEAD' ? 'GET' : request.method;
    const route = `${method} ${rest}`;
    const known = /^(views|videos(\/video_\d{3,}\/thumbnail)?|connectivity|frames|artifacts|active-model|mesh-quality|repair|repair-report|completion(\/model)?|provenance|lod(\/generate|\/[0-4](\/(confidence|provenance))?)?|confidence(\/(analyze|regions|data|coverage))?|tasks(\/(stream|cancel))?|components\/\d+\/model)$/;
    if (!known.test(rest)) return false;
    if (!isSafeId(id)) { send(response, 400, { error: 'Invalid job id.' }); return true; }
    const work = join(root, id);
    if (!existsSync(work)) { send(response, 404, { error: 'Job not found.' }); return true; }
    const running = jobs.get(id)?.status === 'processing';
    const products = productsDir(work);

    if (OPERATIONS[route]) {
      if (running) { send(response, 409, { error: 'This job is still reconstructing. Run the tool when it finishes.' }); return true; }
      if (!originalModelPath(work)) { send(response, 409, { error: 'This job has no 3D mesh to process.' }); return true; }
      const { operation, settings } = OPERATIONS[route];
      const task = tasks.start(id, operation, work, settings(await readBody(request)));
      send(response, 202, { task });
      return true;
    }
    if (route === 'GET tasks') { send(response, 200, { task: tasks.get(id) || null }); return true; }
    if (route === 'GET tasks/stream') { await openTaskStream(id, request, response); return true; }
    if (route === 'POST tasks/cancel') {
      const task = tasks.cancel(id);
      send(response, task ? 200 : 409, task ? { task } : { error: 'No task is running for this job.' });
      return true;
    }

    if (route === 'GET views') {
      if (running) { send(response, 409, { error: 'Camera views are available when the reconstruction finishes.' }); return true; }
      const views = await jobViews(work).catch(() => null);
      send(response, views ? 200 : 404, views || { error: 'This job has no solved cameras to view from.' });
      return true;
    }
    if (route === 'GET videos') {
      const inputs = await readJson(inputsPath(work));
      const frames = await readJson(framesPath(work));
      const registration = await readJson(registrationPath(work));
      const record = await readJson(join(work, 'job.json'));
      send(response, 200, {
        mode: inputs ? 'multi' : 'single',
        videos: inputs
          ? inputs.videos.map((v) => ({
            ...v,
            frames: frames?.perVideo?.find((f) => f.id === v.id) || null,
            registration: registration?.videos?.find((r) => r.id === v.id) || null,
          }))
          : [{ id: 'video_001', originalName: record?.fileName || null }],
      });
      return true;
    }
    if (method === 'GET' && rest.startsWith('videos/')) {
      file(request, response, join(work, 'manifests', 'thumbs', `${rest.split('/')[1]}.jpg`), 'image/jpeg');
      return true;
    }
    if (route === 'GET connectivity') {
      const connectivity = await readJson(connectivityPath(work));
      if (!connectivity) { send(response, 404, { error: 'Connectivity is only computed for multi-video jobs.' }); return true; }
      send(response, 200, {
        connectivity,
        registration: await readJson(registrationPath(work)),
        altitude: (await readJson(join(products, 'transform.json')))?.altitudeReport || null,
        report: (await readJson(join(products, 'report.json')))?.multiVideo || null,
      });
      return true;
    }
    if (route === 'GET frames') { file(request, response, framesPath(work), 'application/json'); return true; }
    if (route === 'GET artifacts') {
      send(response, 200, {
        ...(await allStates(work)),
        task: tasks.get(id) || null,
        running,
        capabilities: { compression: compressionSupport(commands.ffmpeg) },
        defaults: { weights: DEFAULT_WEIGHTS, bands: BANDS, palettes: PALETTES, provenance: PROVENANCE_INFO, provenanceCaps: PROVENANCE_CAPS },
      });
      return true;
    }
    if (route === 'POST active-model') {
      const body = await readBody(request);
      if (body.model !== 'original' && body.model !== 'repaired') { send(response, 400, { error: 'model must be "original" or "repaired".' }); return true; }
      if (body.model === 'repaired') {
        const state = await artifactState(work, 'repair');
        if (!state.current) { stale(response, state, 'The repaired model'); return true; }
      }
      await setActiveModel(work, body.model);
      send(response, 200, await allStates(work));
      return true;
    }
    if (route === 'GET mesh-quality') {
      const state = await artifactState(work, 'analysis');
      if (state.current) {
        send(response, 200, await readJson(join(products, 'repair', 'mesh-quality.json')));
      } else if (running || !originalModelPath(work)) {
        stale(response, state, 'Mesh analysis');
      } else {
        // First request for it: measure now, on a worker, and let the caller poll.
        const task = tasks.isBusy(id) ? tasks.get(id) : tasks.start(id, 'analyze', work, {});
        send(response, 202, { task });
      }
      return true;
    }
    if (route === 'GET repair-report') {
      const state = await artifactState(work, 'repair');
      if (!state.current) { stale(response, state, 'The repair report'); return true; }
      send(response, 200, await readJson(join(products, 'repair', 'repair-report.json')));
      return true;
    }
    if (route === 'GET repair') {
      const state = await artifactState(work, 'repair');
      if (!state.current) { stale(response, state, 'The repaired model'); return true; }
      file(request, response, repairedModelPath(work), 'model/gltf-binary');
      return true;
    }
    if (route === 'GET completion' || route === 'GET completion/model') {
      const state = await artifactState(work, 'completion');
      if (!state.current) { stale(response, state, 'The filled model'); return true; }
      if (route === 'GET completion') send(response, 200, await readJson(join(products, 'completion.json')));
      else file(request, response, completedModelPath(work), 'model/gltf-binary');
      return true;
    }
    if (route === 'GET provenance') {
      const state = await artifactState(work, 'repair');
      if (!state.current) { stale(response, state, 'Provenance'); return true; }
      file(request, response, join(products, 'repair', 'provenance.bin'));
      return true;
    }
    if (method === 'GET' && (rest === 'lod' || /^lod\/[0-4]/.test(rest))) {
      const state = await artifactState(work, 'lod');
      if (!state.current) { stale(response, state, 'The level-of-detail set'); return true; }
      const manifest = await readJson(join(products, 'lod', 'manifest.json'));
      if (rest === 'lod') { send(response, 200, manifest); return true; }
      const [, number, sidecar] = rest.split('/');
      const level = manifest?.levels.find((l) => l.level === Number(number));
      const name = level && (sidecar ? level[sidecar] : level.file);
      // Names come from the manifest this server wrote, never from the request.
      file(request, response, name ? join(products, 'lod', name) : null, sidecar ? 'application/octet-stream' : 'model/gltf-binary');
      return true;
    }
    if (method === 'GET' && rest.startsWith('confidence')) {
      const { which, state, dir, names } = await confidenceFor(work, url.searchParams.get('model'));
      if (!state.current) { stale(response, state, `Confidence for the ${which} model`); return true; }
      if (rest === 'confidence') send(response, 200, { which, ...(await readJson(join(dir, names.report))) });
      else if (rest === 'confidence/regions') send(response, 200, { which, ...(await readJson(join(dir, names.regions))) });
      else if (rest === 'confidence/data') file(request, response, join(dir, names.data));
      else if (rest === 'confidence/coverage') file(request, response, join(dir, 'coverage.json'), 'application/json');
      else return false;
      return true;
    }
    const component = rest.match(/^components\/(\d+)\/model$/);
    if (method === 'GET' && component) {
      const path = join(products, 'components', `component_${Number(component[1])}`, 'model.glb');
      if (existsSync(path) && statSync(path).size > 0) streamFile(response, path, 'model/gltf-binary', request);
      else send(response, 404, { error: 'That component has no model.' });
      return true;
    }
    return false;
  }

  return async (request, response, url) => (await uploadRoutes(request, response, url)) || jobRoutes(request, response, url);
}
