/** Peak CPU, RAM and GPU use over the life of a job, sampled every few seconds. */

import { sampleTelemetry } from './telemetry.mjs';

export function trackPeaks(intervalMs = 5000) {
  const peaks = {
    samples: 0,
    cpuPercent: null,
    ramUsedMb: null,
    ramPercent: null,
    gpuPercent: null,
    vramUsedMb: null,
    // These are machine-wide readings, not per-process ones.
    scope: 'whole machine',
  };
  const raise = (key, value) => {
    if (Number.isFinite(value) && (peaks[key] == null || value > peaks[key])) peaks[key] = value;
  };
  let stopped = false;
  const sample = async () => {
    try {
      const t = await sampleTelemetry();
      if (stopped) return;
      peaks.samples += 1;
      raise('cpuPercent', t.cpu?.loadPercent);
      raise('ramUsedMb', t.memory?.usedMb);
      raise('ramPercent', t.memory?.usedPercent);
      raise('gpuPercent', t.gpu?.utilisationPercent);
      raise('vramUsedMb', t.gpu?.memoryUsedMb);
    } catch { /* telemetry is optional */ }
  };
  const timer = setInterval(sample, intervalMs);
  timer.unref?.();
  sample();
  return {
    current: () => ({ ...peaks }),
    stop() {
      stopped = true;
      clearInterval(timer);
      return { ...peaks };
    },
  };
}
