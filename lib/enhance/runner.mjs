/** Run an enhancement in a worker thread, with cancellation and a time limit. */

import { Worker } from 'node:worker_threads';
import { totalmem } from 'node:os';

/** JavaScript heap ceiling for a worker; geometry buffers live outside it. */
function heapLimitMb() {
  const forced = Number(process.env.SKYFORGE_WORKER_HEAP_MB);
  if (forced > 0) return forced;
  return Math.min(8192, Math.max(1536, Math.round((totalmem() / 1024 / 1024) * 0.4)));
}

/**
 * @param {object} [options]
 * @param {object} [options.job]  anything with a `cancelHooks` set
 * @returns {Promise<{summary:string}>}
 */
export function runEnhancement(operation, work, settings = {}, { job = null, onProgress = null } = {}) {
  return new Promise((resolve, reject) => {
    if (job?.cancelled) { reject(new Error(`${operation} was cancelled.`)); return; }
    const limit = heapLimitMb();
    const worker = new Worker(new URL('./worker.mjs', import.meta.url), {
      workerData: { operation, work, settings },
      resourceLimits: { maxOldGenerationSizeMb: limit },
    });
    let settled = false;
    const finish = (error, result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      job?.cancelHooks?.delete(cancel);
      worker.terminate();
      if (error) reject(error);
      else resolve(result);
    };
    const cancel = () => finish(new Error(`${operation} was cancelled.`));
    job?.cancelHooks?.add(cancel);
    const minutes = Number(process.env.SKYFORGE_ENHANCE_TIMEOUT_MIN) || 120;
    const timer = setTimeout(() => finish(new Error(`${operation} exceeded the ${minutes} minute processing timeout.`)), minutes * 60 * 1000);
    worker.on('message', (message) => {
      if (message.type === 'progress') onProgress?.(message.fraction, message.message);
      else if (message.type === 'done') finish(null, message.result);
      else if (message.type === 'error') {
        const text = /Array buffer allocation failed|Invalid (typed )?array length/i.test(message.message)
          ? `Ran out of memory (${message.message}). Close other applications or reconstruct at a lower quality.`
          : message.message;
        finish(new Error(text));
      }
    });
    worker.on('error', (error) => finish(error.code === 'ERR_WORKER_OUT_OF_MEMORY'
      ? new Error(`Ran out of memory (heap limit ${limit} MB). Raise SKYFORGE_WORKER_HEAP_MB or use a smaller mesh.`)
      : error));
    worker.on('exit', (code) => finish(new Error(`The ${operation} worker stopped unexpectedly (code ${code}).`)));
  });
}
