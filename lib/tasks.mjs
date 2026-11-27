/**
 * Background enhancement tasks started against finished jobs.
 *
 * One runs at a time: each loads a whole mesh and often the dense cloud, and two
 * at once is how a 16 GB machine starts swapping.
 */

import { runEnhancement } from './enhance/runner.mjs';

export function createTaskManager() {
  const tasks = new Map(); // job id -> latest task
  let chain = Promise.resolve();
  let sequence = 0;

  const view = (task) => task && ({
    id: task.id,
    jobId: task.jobId,
    operation: task.operation,
    status: task.status,
    progress: task.progress,
    message: task.message,
    error: task.error,
    summary: task.summary,
    startedAt: task.startedAt,
    endedAt: task.endedAt,
  });
  const active = (task) => task && (task.status === 'queued' || task.status === 'running');

  return {
    get: (jobId) => view(tasks.get(jobId)),
    isBusy: (jobId) => Boolean(active(tasks.get(jobId))),
    start(jobId, operation, work, settings) {
      const current = tasks.get(jobId);
      if (active(current)) {
        const error = new Error(`A ${current.operation} task is already ${current.status} for this job.`);
        error.status = 409;
        throw error;
      }
      sequence += 1;
      const task = {
        id: `task_${Date.now().toString(36)}_${sequence}`,
        jobId,
        operation,
        status: 'queued',
        progress: 0,
        message: 'Waiting for the previous task',
        error: null,
        summary: null,
        startedAt: null,
        endedAt: null,
        cancelled: false,
        cancelHooks: new Set(),
      };
      tasks.set(jobId, task);
      const run = async () => {
        if (task.cancelled) return;
        task.status = 'running';
        task.startedAt = new Date().toISOString();
        task.message = 'Starting';
        try {
          const result = await runEnhancement(operation, work, settings, {
            job: task,
            onProgress: (fraction, message) => {
              task.progress = Math.max(task.progress, Math.round(fraction * 1000) / 10);
              if (message) task.message = message;
            },
          });
          task.status = 'complete';
          task.progress = 100;
          task.summary = result?.summary || null;
          task.message = 'Complete';
        } catch (error) {
          task.status = task.cancelled ? 'cancelled' : 'error';
          task.error = task.cancelled ? null : error.message;
          task.message = task.cancelled ? 'Cancelled' : 'Failed';
        }
        task.endedAt = new Date().toISOString();
      };
      chain = chain.then(run, run);
      return view(task);
    },
    cancel(jobId) {
      const task = tasks.get(jobId);
      if (!active(task)) return null;
      task.cancelled = true;
      if (task.status === 'queued') {
        task.status = 'cancelled';
        task.message = 'Cancelled';
        task.endedAt = new Date().toISOString();
      }
      for (const hook of [...task.cancelHooks]) hook();
      return view(task);
    },
    cancelAll() {
      for (const jobId of tasks.keys()) this.cancel(jobId);
    },
  };
}
