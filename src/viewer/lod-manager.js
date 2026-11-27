/**
 * Progressive loading and automatic switching between levels of detail.
 *
 * Balanced shows full detail whenever the model fits the memory budget, and only
 * steps down while frames are genuinely slow. Performance chooses by screen-space
 * error: measured geometric error and lost texture resolution, projected to
 * pixels at the viewing distance. A light preview is shown first; if a later load
 * fails, whatever is on screen stays.
 */

import { ModelCache } from './model-cache.js';

export const LOD_MODES = Object.freeze({
  performance: { label: 'Performance', pixels: 3, budget: 256 * 1024 * 1024 },
  balanced: { label: 'Balanced', pixels: null, budget: 768 * 1024 * 1024 },
  quality: { label: 'Maximum quality', pixels: 0, budget: 1536 * 1024 * 1024 },
  manual: { label: 'Manual', pixels: null, budget: 1536 * 1024 * 1024 },
});

const DWELL_MS = 1200;
const RETRY_MS = 30000;
/** Automatic modes never show less than this; below it buildings turn to blobs. */
const MIN_AUTO_TRIANGLES = { performance: 40000, balanced: 150000 };
/** Preview shown while the chosen level downloads. */
const PREVIEW_TRIANGLES = 150000;
const SLOW_FPS = 20;
const FAST_FPS = 45;
const RECOVER_MS = 30000;
const bytesOf = (level) => (level.geometryGpuBytes || 0) + (level.textureGpuBytes || 0);

/** A conservative default for machines that report modest capabilities. */
export function defaultMode(renderer) {
  const memory = navigator.deviceMemory;
  if ((memory && memory <= 4) || renderer.capabilities.maxTextureSize < 8192) return 'performance';
  return 'balanced';
}

export class LodManager {
  constructor(viewer, { onState = null } = {}) {
    this.viewer = viewer;
    this.onState = onState;
    this.mode = defaultMode(viewer.renderer);
    this.manualLevel = 0;
    this.cache = new ModelCache(LOD_MODES[this.mode].budget, (model) => viewer.disposeObject(model));
    this.levels = [];
    this.current = null;
    this.pending = null;
    this.measurePending = null;
    this.failed = new Map();
    // How many levels Balanced has stepped down because frames were slow.
    this.penalty = 0;
    this.slowChecks = 0;
    this.penaltyAt = 0;
    this.lastSwitch = 0;
    this.lastCheck = 0;
    this.error = null;
    this.disposed = false;
    this.measuring = false;
  }

  /** Show a light but recognisable level immediately, then refine. */
  async open(manifest, urlFor) {
    this.manifest = manifest;
    this.urlFor = urlFor;
    this.levels = [...manifest.levels].sort((a, b) => a.level - b.level);
    const preview = [...this.levels].reverse().find((l) => l.triangleCount >= PREVIEW_TRIANGLES) || this.levels[0];
    await this.show(preview.level, { fit: true });
    if (this.current == null) await this.show(0, { fit: true });
    this.lastSwitch = 0;
  }

  level(number) { return this.levels.find((l) => l.level === number) || null; }

  setMode(mode) {
    if (!LOD_MODES[mode]) return;
    this.mode = mode;
    this.cache.budget = LOD_MODES[mode].budget;
    this.penalty = 0;
    this.slowChecks = 0;
    this.lastSwitch = 0;
    this.lastCheck = 0;
    if (mode === 'manual' && this.current != null) this.manualLevel = this.current;
    this.failed.clear();
    this.update(performance.now(), true);
    this.publish();
  }

  setManualLevel(number) {
    this.manualLevel = number;
    this.failed.delete(number);
    if (this.mode !== 'manual') this.setMode('manual');
    this.manualLevel = number;
    this.show(number);
  }

  /** Metres covered by one texel, from the manifest or estimated from it. */
  texelSize(level) {
    if (level.texelSizeM) return level.texelSizeM;
    if (!level.textureResolution) return 0;
    const area = (this.manifest?.sceneDiagonalM || 0) ** 2 / 4;
    return Math.sqrt(area / (level.textureResolution ** 2 * Math.max(1, level.textureCount || 1)));
  }

  /** Projected error of a level in pixels at the current view. */
  screenError(level) {
    const { camera, renderer } = this.viewer;
    const height = renderer.domElement.clientHeight || 1;
    const perMetre = height / (2 * Math.tan((camera.fov * Math.PI) / 360));
    // Blur from a smaller texture counts as much as a moved surface.
    const original = this.level(0);
    const blur = original ? Math.max(0, this.texelSize(level) - this.texelSize(original)) : 0;
    return (Math.max(level.geometricError, blur) * perMetre) / Math.max(this.viewer.viewDistance(), 0.01);
  }

  choose() {
    if (this.mode === 'manual') return this.manualLevel;
    if (this.mode === 'quality') return 0;
    const budget = LOD_MODES[this.mode].budget;
    const floor = MIN_AUTO_TRIANGLES[this.mode] || 0;
    const fits = this.levels.filter((l) => bytesOf(l) <= budget);
    const detailed = fits.filter((l) => l.level === 0 || l.triangleCount >= floor);
    const usable = detailed.length ? detailed : [fits[0] || this.levels[this.levels.length - 1]];
    if (this.mode === 'balanced') return usable[Math.min(this.penalty, usable.length - 1)].level;
    const threshold = LOD_MODES[this.mode].pixels;
    // The coarsest level whose error is still invisible enough.
    let choice = usable[0];
    for (const level of usable) {
      const pixels = this.screenError(level);
      const coarser = this.current != null && level.level > this.current;
      // Moving coarser needs a margin; staying or refining does not.
      if (pixels <= (coarser ? threshold * 0.75 : threshold)) choice = level;
    }
    const now = this.level(this.current);
    if (now && usable.includes(now) && choice.level < now.level && this.screenError(now) <= threshold * 1.15) return now.level;
    return choice.level;
  }

  update(time, force = false) {
    if (this.disposed || !this.levels.length || (!force && time - this.lastCheck < 300)) return;
    this.lastCheck = time;
    const { fps, age } = this.viewer.performanceStats();
    // Only a frame rate measured just now says anything: the viewer idles.
    if (this.mode === 'balanced' && fps != null && age < 500) {
      this.slowChecks = fps < SLOW_FPS ? this.slowChecks + 1 : 0;
      if (this.slowChecks >= 6 && this.penalty < this.levels.length - 1) {
        this.penalty += 1;
        this.penaltyAt = time;
        this.slowChecks = 0;
      } else if (this.penalty && fps > FAST_FPS && time - this.penaltyAt > RECOVER_MS) {
        this.penalty -= 1;
        this.penaltyAt = time;
      }
    }
    const target = this.choose();
    if (target === this.current) {
      if (this.pending && this.pending.level !== target) this.cancelPending();
    } else if (force || time - this.lastSwitch >= DWELL_MS) {
      const failedAt = this.failed.get(target);
      if (!failedAt || time - failedAt > RETRY_MS) this.show(target);
    }
    if (force) this.publish();
  }

  cancelPending() {
    this.pending?.controller.abort();
    this.pending = null;
  }

  async fetchLevel(number, controller, quiet = false) {
    const cached = this.cache.get(number);
    if (cached) return cached;
    const level = this.level(number);
    // Only the very first level reports download progress; refinements are silent.
    const loaded = await this.viewer.loadModelData(this.urlFor(level), { signal: controller.signal, quiet });
    if (this.disposed) {
      this.viewer.disposeObject(loaded.model);
      throw new DOMException('disposed', 'AbortError');
    }
    // A measurement load may have raced this one.
    const raced = this.cache.get(number);
    if (raced) {
      this.viewer.disposeObject(loaded.model);
      return raced;
    }
    const entry = { ...loaded, bytes: bytesOf(level) };
    this.cache.put(number, entry);
    return entry;
  }

  async show(number, { fit = false } = {}) {
    if (this.disposed || !this.level(number)) return;
    if (this.pending?.level === number) return;
    this.cancelPending();
    const controller = new AbortController();
    const request = { level: number, controller };
    this.pending = request;
    this.publish();
    try {
      const entry = await this.fetchLevel(number, controller, this.current != null);
      if (this.pending !== request) return;
      this.viewer.showModel(entry.model, { fit, owned: false });
      this.current = number;
      this.lastSwitch = performance.now();
      this.error = null;
      this.viewer.onLodChange?.(number);
    } catch (error) {
      if (error.name !== 'AbortError') {
        this.failed.set(number, performance.now());
        this.error = `Level ${number} failed to load (${error.message}); showing the previous level.`;
      }
    } finally {
      if (this.pending === request) this.pending = null;
      this.cache.trim(this.pinned());
      this.publish();
    }
  }

  pinned() {
    const keep = new Set();
    if (this.current != null) keep.add(this.current);
    if (this.measuring) keep.add(0);
    return keep;
  }

  /** Load full-resolution geometry for measuring, without displaying it. */
  setMeasuring(on) {
    this.measuring = on;
    if (!on) { this.cache.trim(this.pinned()); this.publish(); return; }
    if (this.cache.has(0) || this.measurePending) { this.publish(); return; }
    const controller = new AbortController();
    this.measurePending = this.fetchLevel(0, controller, true)
      .catch((error) => { this.error = `Full-resolution geometry could not be loaded: ${error.message}`; })
      .finally(() => { this.measurePending = null; this.publish(); });
    this.publish();
  }

  /** @returns the model to measure against, and whether it is full resolution */
  measurementModel() {
    const original = this.cache.get(0);
    if (original) return { model: original.model, full: true };
    const shown = this.cache.get(this.current);
    return shown ? { model: shown.model, full: false, level: this.current, loading: Boolean(this.measurePending) } : null;
  }

  snapshot() {
    const level = this.level(this.current);
    return {
      mode: this.mode,
      manualLevel: this.manualLevel,
      current: this.current,
      loading: this.pending?.level ?? null,
      error: this.error,
      level,
      original: this.level(0),
      levels: this.levels,
      screenErrorPixels: level ? this.screenError(level) : null,
      cacheBytes: this.cache.bytes(),
      cacheBudget: this.cache.budget,
      cached: [...this.cache.entries.keys()],
      measurementReady: this.cache.has(0),
      steppedDown: this.penalty,
    };
  }

  publish() { if (!this.disposed) this.onState?.(this.snapshot()); }

  dispose() {
    this.disposed = true;
    this.cancelPending();
    const shown = this.cache.get(this.current);
    if (shown) this.viewer.releaseModel(shown.model);
    this.cache.clear();
    this.levels = [];
  }
}
