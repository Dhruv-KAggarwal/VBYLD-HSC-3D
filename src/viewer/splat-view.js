/**
 * Gaussian splats on their own canvas.
 *
 * three.js draws `GaussianSplat` only through its WebGPURenderer (WebGPU, or
 * that renderer's WebGL fallback); the mesh viewer's WebGLRenderer cannot. So
 * the splat gets a second canvas laid over the mesh canvas, driven by the same
 * camera. It ignores pointer events, so the existing controls keep working.
 * Everything is imported on first use, to keep the ordinary viewer light.
 */

/** Splats farther than this many filmed-area radii from its centre are dropped. */
const KEEP_RADII = 10;
/** As are single splats wider than this share of the radius. */
const MAX_WIDTH_RADII = 0.5;

/**
 * Remove stray splats before they reach the renderer.
 *
 * Training leaves a few splats at enormous distances (sky has no depth). three.js
 * sorts by depth into a fixed number of bins across the whole bounding sphere, so
 * one such stray collapses the entire scene into a single bin and it draws
 * unsorted, as a translucent haze.
 *
 * @returns {{geometry: object, removed: number}}
 */
function dropStrays(THREE, geometry, matrix, focus) {
  const position = geometry.getAttribute('position').array;
  const covariance = geometry.getAttribute('covariance').array;
  const count = position.length / 3;
  const m = matrix || [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const scale = Math.hypot(m[0], m[1], m[2]);
  let centre = focus?.center;
  let radius = focus?.radius;
  if (!centre) {
    // No camera information: use the bulk of the splats themselves.
    const step = Math.max(1, Math.floor(count / 20000));
    const axes = [[], [], []];
    for (let i = 0; i < count; i += step) {
      for (let k = 0; k < 3; k += 1) axes[k].push(m[k] * position[i * 3] + m[4 + k] * position[i * 3 + 1] + m[8 + k] * position[i * 3 + 2] + m[12 + k]);
    }
    for (const axis of axes) axis.sort((a, b) => a - b);
    centre = axes.map((axis) => axis[axis.length >> 1]);
    radius = Math.max(...axes.map((axis) => axis[Math.floor(axis.length * 0.85)] - axis[Math.floor(axis.length * 0.15)])) || 1;
  }
  const keep = new Uint32Array(count);
  let kept = 0;
  for (let i = 0; i < count; i += 1) {
    const x = position[i * 3];
    const y = position[i * 3 + 1];
    const z = position[i * 3 + 2];
    const dx = m[0] * x + m[4] * y + m[8] * z + m[12] - centre[0];
    const dy = m[1] * x + m[5] * y + m[9] * z + m[13] - centre[1];
    const dz = m[2] * x + m[6] * y + m[10] * z + m[14] - centre[2];
    // The covariance diagonal (xx, yy, zz) bounds the splat's widest axis.
    const spread = Math.sqrt(Math.max(covariance[i * 6], covariance[i * 6 + 3], covariance[i * 6 + 5])) * scale;
    if (dx * dx + dy * dy + dz * dz <= (KEEP_RADII * radius) ** 2 && spread <= MAX_WIDTH_RADII * radius) {
      keep[kept] = i;
      kept += 1;
    }
  }
  if (kept === count) return { geometry, removed: 0 };
  const filtered = new THREE.BufferGeometry();
  for (const [name, attribute] of Object.entries(geometry.attributes)) {
    const size = attribute.itemSize;
    const values = new attribute.array.constructor(kept * size);
    for (let j = 0; j < kept; j += 1) {
      for (let k = 0; k < size; k += 1) values[j * size + k] = attribute.array[keep[j] * size + k];
    }
    filtered.setAttribute(name, new THREE.BufferAttribute(values, size, attribute.normalized));
  }
  geometry.dispose();
  return { geometry: filtered, removed: count - kept };
}

export class SplatView {
  constructor(hostCanvas) {
    this.host = hostCanvas;
    this.canvas = null;
    this.renderer = null;
    this.scene = null;
    this.splat = null;
    this.ready = null;
  }

  init() {
    this.ready ||= (async () => {
      const [THREE, { GaussianSplat }, { GaussianSplatPLYLoader }] = await Promise.all([
        import('three/webgpu'),
        import('three/addons/objects/GaussianSplat.js'),
        import('three/addons/loaders/GaussianSplatPLYLoader.js'),
      ]);
      const canvas = document.createElement('canvas');
      canvas.className = 'scene-canvas';
      canvas.style.cssText = 'position:absolute;inset:0;pointer-events:none;display:none';
      this.host.insertAdjacentElement('afterend', canvas);
      const renderer = new THREE.WebGPURenderer({
        canvas, antialias: false, alpha: false, forceWebGL: !navigator.gpu, powerPreference: 'high-performance',
      });
      await renderer.init();
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      // Splat colours are fitted to the photographs and are already sRGB. Left at
      // its default the renderer encodes them a second time and washes them out.
      renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
      this.canvas = canvas;
      this.renderer = renderer;
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color().setHex(0x0f1821, THREE.LinearSRGBColorSpace);
      this.THREE = THREE;
      this.GaussianSplat = GaussianSplat;
      this.loader = new GaussianSplatPLYLoader();
    })().catch((error) => {
      this.ready = null;
      throw new Error(`This browser could not start the splat renderer: ${error.message}`);
    });
    return this.ready;
  }

  /**
   * @param {ArrayBuffer} buffer  3DGS PLY in the reconstruction frame
   * @param {number[]|null} matrix  column-major transform into viewer coordinates
   * @param {{center:number[], radius:number}|null} focus  the filmed area
   * @returns {Promise<{count:number, removed:number}>}
   */
  async load(buffer, matrix, focus = null) {
    await this.init();
    const { geometry, removed } = dropStrays(this.THREE, this.loader.parse(buffer), matrix, focus);
    this.clear();
    const splat = new this.GaussianSplat(geometry);
    splat.name = 'GaussianSplat';
    this.splat = splat;
    this.align(matrix);
    this.scene.add(splat);
    return { count: geometry.getAttribute('position').count, removed };
  }

  align(matrix) {
    if (!this.splat || !matrix) return;
    this.splat.matrixAutoUpdate = false;
    this.splat.matrix.fromArray(matrix);
    this.splat.matrixWorldNeedsUpdate = true;
  }

  setVisible(visible) {
    if (this.canvas) this.canvas.style.display = visible && this.splat ? 'block' : 'none';
  }

  get visible() { return Boolean(this.splat) && this.canvas?.style.display !== 'none'; }

  render(camera) {
    if (!this.visible) return;
    const width = Math.max(2, this.host.clientWidth);
    const height = Math.max(2, this.host.clientHeight);
    if (width !== this.width || height !== this.height) {
      this.width = width;
      this.height = height;
      this.renderer.setSize(width, height, false);
    }
    this.renderer.render(this.scene, camera);
  }

  clear() {
    if (!this.splat) return;
    this.scene.remove(this.splat);
    this.splat.dispose?.();
    this.splat.geometry?.dispose?.();
    this.splat.material?.dispose?.();
    this.splat = null;
    this.setVisible(false);
  }
}
