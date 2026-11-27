/**
 * Thin client for the Skyforge job API.
 *
 * The server takes the video as a raw request body (not multipart) with the
 * options in headers, returns 202 + {id}, and streams progress over SSE with a
 * polling fallback.
 */

export const API = '';

export const fileUrl = (id, kind) => `${API}/api/reconstruct/${id}/file?kind=${kind}`;

/**
 * Same file, with a cache-buster.
 *
 * Products are rewritten in place when a job is re-run or when a turntable is
 * rendered on demand, and a <video> that has already seen a 404 or an older
 * copy will happily keep showing it. The server sends `Cache-Control:
 * no-store`, but the element's own in-memory copy survives a re-render, so the
 * URL has to change for it to reload.
 */
export const fileUrlFresh = (id, kind, token = Date.now()) => `${fileUrl(id, kind)}&t=${token}`;

/** Ask the server which outputs this job actually has on disk. */
export async function getJobFiles(id) {
  const response = await fetch(`${API}/api/reconstruct/${id}/files`);
  if (!response.ok) throw new Error(`Could not list job files: ${response.status}`);
  return response.json();
}

/** True if the file exists, without downloading it. */
export async function hasFile(id, kind) {
  try {
    const response = await fetch(fileUrl(id, kind), { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Render a turntable for a job that has none.
 *
 * This blocks for as long as the render takes (tens of seconds), because the
 * server does the work synchronously — the caller wants the video, not another
 * job to poll.
 */
export async function generateTurntable(id) {
  const response = await fetch(`${API}/api/reconstruct/${id}/turntable`, { method: 'POST' });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Turntable render failed: ${response.status}`);
  return body;
}

export async function getHealth() {
  const response = await fetch(`${API}/api/health`);
  if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
  return response.json();
}

export async function listJobs() {
  const response = await fetch(`${API}/api/jobs`);
  if (!response.ok) throw new Error(`Could not list jobs: ${response.status}`);
  return response.json();
}

export async function getJob(id) {
  const response = await fetch(`${API}/api/reconstruct/${id}`);
  if (!response.ok) throw new Error(`Could not read job: ${response.status}`);
  return response.json();
}

export async function cancelJob(id) {
  const response = await fetch(`${API}/api/reconstruct/${id}/cancel`, { method: 'POST' });
  if (!response.ok) throw new Error(`Could not cancel: ${response.status}`);
  return response.json();
}

/**
 * Start a reconstruction.
 *
 * The body is the raw file so a multi-gigabyte upload streams straight to disk
 * instead of being base64-inflated through a multipart form.
 */
export async function startReconstruction(file, options) {
  const headers = {
    'Content-Type': 'application/octet-stream',
    'X-File-Name': encodeURIComponent(file.name),
    'X-Quality': options.quality,
    'X-Altitude': String(options.altitude),
  };
  if (options.frames) headers['X-Frames'] = String(options.frames);
  if (options.threads) headers['X-Threads'] = String(options.threads);
  if (options.splat) headers['X-Splat'] = '1';
  if (options.dsm) headers['X-Dsm'] = '1';
  if (options.maskDynamic) headers['X-Mask-Dynamic'] = '1';
  if (options.enhance) headers['X-Enhance'] = encodeURIComponent(JSON.stringify(options.enhance));

  const response = await fetch(`${API}/api/reconstruct`, { method: 'POST', headers, body: file });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).error || detail; } catch { /* keep status */ }
    throw new Error(detail);
  }
  return response.json();
}

/**
 * Follow a job. Returns an unsubscribe function.
 *
 * SSE first; on error it degrades to 1s polling rather than going silent,
 * because a stalled progress bar is indistinguishable from a hung engine.
 */
export function watchJob(id, { onUpdate, onDone, onError, onLog }) {
  let stopped = false;
  let source = null;
  let timer = null;
  let logSeq = 0;

  const takeLines = (lines) => {
    if (!lines?.length) return;
    logSeq = Math.max(logSeq, lines[lines.length - 1].seq || 0);
    onLog?.(lines);
  };

  const finish = (job) => {
    if (stopped) return;
    stopped = true;
    cleanup();
    if (job?.status === 'error') onError?.(job);
    else onDone?.(job);
  };

  const cleanup = () => {
    source?.close();
    source = null;
    if (timer) clearInterval(timer);
    timer = null;
  };

  const startPolling = () => {
    if (stopped || timer) return;
    timer = setInterval(async () => {
      try {
        const job = await getJob(id);
        onUpdate?.(job);
        // Pull whatever log lines appeared since the last poll, so the fallback
        // path is not silent either.
        try {
          const response = await fetch(`${API}/api/reconstruct/${id}/log?since=${logSeq}`);
          if (response.ok) takeLines((await response.json()).lines);
        } catch { /* log is best-effort */ }
        if (job.status === 'complete' || job.status === 'error' || job.status === 'cancelled') finish(job);
      } catch { /* keep polling; the server may be restarting */ }
    }, 1000);
  };

  try {
    source = new EventSource(`${API}/api/reconstruct/${id}/stream`);
    source.addEventListener('progress', (event) => onUpdate?.(JSON.parse(event.data)));
    source.addEventListener('log', (event) => takeLines(JSON.parse(event.data).lines));
    source.addEventListener('done', (event) => finish(JSON.parse(event.data)));
    source.addEventListener('error', (event) => {
      // An SSE `error` event is both "the server sent an error payload" and
      // "the socket dropped". Only the first carries data.
      if (event.data) { try { finish(JSON.parse(event.data)); return; } catch { /* fall through */ } }
      source?.close();
      source = null;
      startPolling();
    });
  } catch {
    startPolling();
  }

  return () => { stopped = true; cleanup(); };
}

// ---------------------------------------------------------------------------
// Multi-video uploads
// ---------------------------------------------------------------------------

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error || `${response.status} ${response.statusText}`);
    error.status = response.status;
    error.stale = Boolean(body.stale);
    throw error;
  }
  return body;
}

const json = (method, value) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value || {}) });

const uploadQuery = ({ quality, frames } = {}) => {
  const query = new URLSearchParams();
  if (quality) query.set('quality', quality);
  if (frames) query.set('frames', String(frames));
  const text = query.toString();
  return text ? `?${text}` : '';
};

export const createUpload = (options) => request(`/api/uploads${uploadQuery(options)}`, { method: 'POST' });
export const getUpload = (id, options) => request(`/api/uploads/${id}${uploadQuery(options)}`);
export const updateUpload = (id, changes, options) => request(`/api/uploads/${id}${uploadQuery(options)}`, json('PATCH', changes));
export const deleteUpload = (id) => request(`/api/uploads/${id}`, { method: 'DELETE' });
export const deleteUploadVideo = (id, videoId, options) => request(`/api/uploads/${id}/videos/${videoId}${uploadQuery(options)}`, { method: 'DELETE' });
export const uploadThumbnailUrl = (id, video) => `${API}/api/uploads/${id}/videos/${video.id}/thumbnail?v=${video.sha256?.slice(0, 8)}`;

/**
 * Upload one video. XMLHttpRequest rather than fetch, because only it reports
 * upload progress. Returns the promise and an abort function.
 */
export function uploadVideo(sessionId, file, { replace = null, onProgress = null } = {}) {
  const xhr = new XMLHttpRequest();
  const promise = new Promise((resolve, reject) => {
    xhr.open('POST', `${API}/api/uploads/${sessionId}/videos${replace ? `?replace=${replace}` : ''}`);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.setRequestHeader('X-File-Name', encodeURIComponent(file.name));
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total);
    };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.error || `Upload failed: ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error('The upload connection failed.'));
    xhr.onabort = () => reject(Object.assign(new Error('Upload cancelled.'), { aborted: true }));
    xhr.send(file);
  });
  return { promise, abort: () => xhr.abort() };
}

export const startMultiReconstruction = (uploadId, options) => request('/api/reconstruct', json('POST', { uploadId, options }));

// ---------------------------------------------------------------------------
// Job products: videos, repair, levels of detail, confidence
// ---------------------------------------------------------------------------

const job = (id, path) => `/api/reconstruct/${id}/${path}`;

export const getArtifacts = (id) => request(job(id, 'artifacts'));
export const setActiveModel = (id, model) => request(job(id, 'active-model'), json('POST', { model }));
export const getJobVideos = (id) => request(job(id, 'videos'));
export const getViews = (id) => request(job(id, 'views'));
export const jobThumbnailUrl = (id, videoId) => `${API}${job(id, `videos/${videoId}/thumbnail`)}`;
export const getConnectivity = (id) => request(job(id, 'connectivity'));
export const getMeshQuality = async (id) => {
  const response = await fetch(`${API}${job(id, 'mesh-quality')}`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Mesh analysis unavailable: ${response.status}`);
  // 202: the analysis has just been started as a task.
  return response.status === 202 ? { pending: true, task: body.task } : body;
};
export const startRepair = (id, settings) => request(job(id, 'repair'), json('POST', settings));
export const getRepairReport = (id) => request(job(id, 'repair-report'));
export const startConfidence = (id, settings) => request(job(id, 'confidence/analyze'), json('POST', settings));
export const getConfidence = (id, model) => request(job(id, `confidence${model ? `?model=${model}` : ''}`));
export const getConfidenceRegions = (id, model) => request(job(id, `confidence/regions${model ? `?model=${model}` : ''}`));
export const startLod = (id, settings) => request(job(id, 'lod/generate'), json('POST', settings));
export const getLodManifest = (id) => request(job(id, 'lod'));
export const lodUrl = (id, level, token = '') => `${API}${job(id, `lod/${level}`)}?t=${encodeURIComponent(token)}`;
export const componentUrl = (id, number) => `${API}${job(id, `components/${number}/model`)}`;
export const cancelTask = (id) => request(job(id, 'tasks/cancel'), { method: 'POST' });
export const getTask = (id) => request(job(id, 'tasks'));

/** A binary sidecar (confidence or provenance records). */
export async function getBinary(path) {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `${response.status} ${response.statusText}`);
  }
  return response.arrayBuffer();
}

export const confidenceDataPath = (id, model, level = 0) => (level
  ? job(id, `lod/${level}/confidence`)
  : job(id, `confidence/data?model=${model}`));
export const provenanceDataPath = (id, level = 0) => (level ? job(id, `lod/${level}/provenance`) : job(id, 'provenance'));

/** Follow the enhancement task of a job: SSE first, polling if it drops. */
export function watchTask(id, onUpdate) {
  let stopped = false;
  let source = null;
  let timer = null;
  const stop = () => {
    stopped = true;
    source?.close();
    if (timer) clearInterval(timer);
  };
  const take = (task) => {
    if (stopped) return;
    onUpdate(task);
    if (!task || (task.status !== 'queued' && task.status !== 'running')) stop();
  };
  const poll = () => {
    if (stopped || timer) return;
    timer = setInterval(async () => {
      try { take((await getTask(id)).task); } catch { /* server restarting */ }
    }, 1000);
  };
  try {
    source = new EventSource(`${API}${job(id, 'tasks/stream')}`);
    const read = (event) => take(JSON.parse(event.data).task);
    source.addEventListener('progress', read);
    source.addEventListener('done', read);
    source.onerror = () => { source?.close(); source = null; poll(); };
  } catch {
    poll();
  }
  return stop;
}
