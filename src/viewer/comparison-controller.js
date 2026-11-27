/**
 * Original-versus-repaired comparison in the single viewer.
 *
 * Both models share one camera, so navigation is synchronised by construction.
 * "side" renders each into half of the canvas; "slider" splits one view with a
 * scissor test at a draggable position.
 */

import { FaceOverlay } from './overlay-renderer.js';

export class ComparisonController {
  constructor(viewer) {
    this.viewer = viewer;
    this.active = false;
    this.mode = 'slider';
    this.split = 0.5;
    this.showRemoved = false;
    this.showRepaired = false;
    this.models = null;
    this.overlay = new FaceOverlay();
    this.generation = 0;
  }

  async enable({ originalUrl, repairedUrl, removedUrl, provenance }) {
    this.disable();
    const generation = ++this.generation;
    const load = (url) => this.viewer.loadModelData(url);
    const [original, repaired] = await Promise.all([load(originalUrl), load(repairedUrl)]);
    const removed = removedUrl ? await load(removedUrl).catch(() => null) : null;
    if (generation !== this.generation) {
      for (const entry of [original, repaired, removed]) if (entry) this.viewer.disposeObject(entry.model);
      return;
    }
    this.models = { original: original.model, repaired: repaired.model, removed: removed?.model || null };
    if (this.models.removed) {
      this.models.removed.traverse((node) => {
        if (!node.isMesh) return;
        node.material.polygonOffset = true;
        node.material.polygonOffsetFactor = -3;
        node.material.polygonOffsetUnits = -6;
      });
    }
    for (const model of Object.values(this.models)) if (model) this.viewer.root.add(model);
    this.provenance = provenance;
    this.active = true;
    if (provenance) {
      this.overlay.attach(this.models.repaired, { count: provenance.count, provenance: (f) => provenance.codes[f] });
      this.overlay.update({ mode: this.showRepaired ? 'provenance' : 'off', opacity: 0.8 });
    }
    this.viewer.setComparisonActive(true);
  }

  set({ mode, split, showRemoved, showRepaired }) {
    if (mode) this.mode = mode;
    if (split != null) this.split = Math.min(0.98, Math.max(0.02, split));
    if (showRemoved != null) this.showRemoved = showRemoved;
    if (showRepaired != null) {
      this.showRepaired = showRepaired;
      this.overlay.update({ mode: showRepaired ? 'provenance' : 'off' });
    }
    this.viewer.invalidate();
  }

  /** Models the measuring and wireframe tools should act on. */
  list() {
    return this.models ? [this.models.original, this.models.repaired] : [];
  }

  render(renderer, scene, camera) {
    const { original, repaired, removed } = this.models;
    if (removed) removed.visible = this.showRemoved;
    const show = (which) => {
      original.visible = which === original;
      repaired.visible = which === repaired;
    };
    if (this.mode === 'original' || this.mode === 'repaired') {
      show(this.mode === 'original' ? original : repaired);
      renderer.render(scene, camera);
      return;
    }
    const width = renderer.domElement.clientWidth;
    const height = renderer.domElement.clientHeight;
    const side = this.mode === 'side';
    const cut = Math.round(width * (side ? 0.5 : this.split));
    const aspect = camera.aspect;
    renderer.setScissorTest(true);
    for (const [model, x, w] of [[original, 0, cut], [repaired, cut, width - cut]]) {
      show(model);
      if (side) {
        renderer.setViewport(x, 0, w, height);
        camera.aspect = w / height;
        camera.updateProjectionMatrix();
      }
      renderer.setScissor(x + (x ? 1 : 0), 0, Math.max(0, w - (x ? 1 : 0)), height);
      renderer.render(scene, camera);
    }
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, width, height);
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
  }

  disable() {
    this.generation += 1;
    this.overlay.detach();
    if (this.models) for (const model of Object.values(this.models)) if (model) this.viewer.disposeObject(model);
    this.models = null;
    if (this.active) {
      this.active = false;
      this.viewer.setComparisonActive(false);
    }
  }
}
