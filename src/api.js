/**
 * Thin client for the Skyforge job API.
 *
 * The server takes the video as a raw request body (not multipart) with the
 * options in headers, returns 202 + {id}, and streams progress over SSE with a
 * polling fallback.
 */

export const API = '';

export const fileUrl = (id, kind) => `${API}/api/reconstruct/${id}/file?kind=${kind}`;

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
export function watchJob(id, { onUpdate, onDone, onError }) {
  let stopped = false;
  let source = null;
  let timer = null;

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
        if (job.status === 'complete' || job.status === 'error' || job.status === 'cancelled') finish(job);
      } catch { /* keep polling; the server may be restarting */ }
    }, 1000);
  };

  try {
    source = new EventSource(`${API}/api/reconstruct/${id}/stream`);
    source.addEventListener('progress', (event) => onUpdate?.(JSON.parse(event.data)));
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
