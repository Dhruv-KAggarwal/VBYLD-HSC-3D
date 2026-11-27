/** Worker-thread entry: runs one enhancement off the server's event loop. */

import { parentPort, workerData } from 'node:worker_threads';
import { runOperation } from './index.mjs';

const { operation, work, settings } = workerData;
let last = 0;
try {
  const result = await runOperation(operation, work, settings, (fraction, message) => {
    const now = Date.now();
    // Throttled: some loops report thousands of times.
    if (!message && fraction < 1 && now - last < 250) return;
    last = now;
    parentPort.postMessage({ type: 'progress', fraction, message: message || null });
  });
  parentPort.postMessage({ type: 'done', result });
} catch (error) {
  parentPort.postMessage({ type: 'error', message: error?.message || String(error), stack: error?.stack || null });
}
