/**
 * Per-face colour overlays (confidence heatmap and geometry provenance).
 *
 * The overlay is a second mesh drawn over each model mesh with its own shader,
 * so the authored materials and textures are never modified. Filters are shader
 * uniforms: changing them costs nothing.
 */

import * as THREE from 'three';

export const OVERLAY_DEFAULTS = Object.freeze({
  mode: 'off', // off | confidence | provenance
  opacity: 0.75,
  palette: 'standard',
  min: 0,
  max: 100,
  threshold: 40,
  showUnknown: true,
  onlyLow: false,
  onlyUnknown: false,
  onlyRepaired: false,
  isolate: false,
  hiddenProvenance: [],
});

export const PALETTE_COLORS = Object.freeze({
  // Order: very low, low, moderate, good, high, unknown.
  standard: ['#dc2626', '#f97316', '#facc15', '#a3e635', '#22c55e', '#808080'],
  accessible: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725', '#969696'],
});

export const PROVENANCE_COLORS = Object.freeze(['#808080', '#3b82f6', '#22d3ee', '#facc15', '#fb923c', '#ef4444']);
export const PROVENANCE_LABELS = Object.freeze(['Original', 'Cleaned', 'Smoothed', 'Interpolated', 'Hole filled', 'Unsupported']);

// Raw sRGB components: the shader writes straight to the framebuffer.
const rgb = (hex) => new THREE.Vector3(
  parseInt(hex.slice(1, 3), 16) / 255, parseInt(hex.slice(3, 5), 16) / 255, parseInt(hex.slice(5, 7), 16) / 255,
);

const vertexShader = /* glsl */`
  attribute float aScore;
  attribute float aProvenance;
  varying float vScore;
  varying float vProvenance;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  #include <clipping_planes_pars_vertex>
  void main() {
    vScore = aScore;
    vProvenance = aProvenance;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    #include <logdepthbuf_vertex>
    #include <clipping_planes_vertex>
  }
`;

const fragmentShader = /* glsl */`
  uniform int uMode;
  uniform float uOpacity;
  uniform vec3 uBands[6];
  uniform vec3 uProvenance[6];
  uniform float uVisible[6];
  uniform float uMin;
  uniform float uMax;
  uniform float uThreshold;
  uniform float uShowUnknown;
  uniform float uOnlyLow;
  uniform float uOnlyUnknown;
  uniform float uOnlyRepaired;
  varying float vScore;
  varying float vProvenance;
  #include <logdepthbuf_pars_fragment>
  #include <clipping_planes_pars_fragment>
  void main() {
    #include <clipping_planes_fragment>
    #include <logdepthbuf_fragment>
    int code = int(vProvenance + 0.5);
    vec3 color;
    if (uMode == 2) {
      // Original geometry keeps its normal appearance.
      if (code == 0 || uVisible[code] < 0.5) discard;
      color = uProvenance[code];
    } else {
      float score = floor(vScore + 0.5);
      bool unknown = score > 100.5;
      if (uOnlyRepaired > 0.5 && code == 0) discard;
      if (uOnlyUnknown > 0.5 && !unknown) discard;
      if (uOnlyLow > 0.5 && (unknown || score >= uThreshold)) discard;
      if (unknown) {
        if (uShowUnknown < 0.5) discard;
        color = uBands[5];
      } else {
        if (score < uMin || score > uMax) discard;
        color = score >= 80.0 ? uBands[4] : score >= 60.0 ? uBands[3] : score >= 40.0 ? uBands[2]
          : score >= 20.0 ? uBands[1] : uBands[0];
      }
    }
    gl_FragColor = vec4(color, uOpacity);
  }
`;

export class FaceOverlay {
  constructor() {
    this.settings = { ...OVERLAY_DEFAULTS };
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uMode: { value: 1 },
        uOpacity: { value: 0.75 },
        uBands: { value: PALETTE_COLORS.standard.map(rgb) },
        uProvenance: { value: PROVENANCE_COLORS.map(rgb) },
        uVisible: { value: [1, 1, 1, 1, 1, 1] },
        uMin: { value: 0 },
        uMax: { value: 100 },
        uThreshold: { value: 40 },
        uShowUnknown: { value: 1 },
        uOnlyLow: { value: 0 },
        uOnlyUnknown: { value: 0 },
        uOnlyRepaired: { value: 0 },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      clipping: true,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -4,
    });
    this.attached = [];
  }

  /**
   * @param {THREE.Object3D} model
   * @param {{score?:(f:number)=>number, provenance?:(f:number)=>number, count:number}} data
   *   per-face values indexed in the model file's face order
   */
  attach(model, data) {
    this.detach();
    if (!model || !data) return;
    const meshes = [];
    model.traverse((node) => { if (node.isMesh && !node.userData.isOverlay) meshes.push(node); });
    let running = 0;
    const plan = meshes.map((node) => {
      const { geometry } = node;
      const faces = (geometry.index ? geometry.index.count : geometry.attributes.position.count) / 3;
      const offset = geometry.userData.skyforgeFaceOffset ?? running;
      running += faces;
      return { node, faces, offset };
    });
    if (running !== data.count) {
      throw new Error(`The overlay data describes ${data.count.toLocaleString()} faces but the model has ${running.toLocaleString()}.`);
    }
    for (const { node, faces, offset } of plan) {
      const source = node.geometry.attributes.position;
      const { index } = node.geometry;
      const positions = new Float32Array(faces * 9);
      const score = new Uint8Array(faces * 3);
      const provenance = new Uint8Array(faces * 3);
      for (let f = 0; f < faces; f += 1) {
        const value = data.score ? data.score(offset + f) : 255;
        const code = data.provenance ? data.provenance(offset + f) : 0;
        for (let k = 0; k < 3; k += 1) {
          const v = index ? index.getX(f * 3 + k) : f * 3 + k;
          positions[(f * 3 + k) * 3] = source.getX(v);
          positions[(f * 3 + k) * 3 + 1] = source.getY(v);
          positions[(f * 3 + k) * 3 + 2] = source.getZ(v);
          score[f * 3 + k] = value;
          provenance[f * 3 + k] = code;
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('aScore', new THREE.BufferAttribute(score, 1));
      geometry.setAttribute('aProvenance', new THREE.BufferAttribute(provenance, 1));
      const overlay = new THREE.Mesh(geometry, this.material);
      overlay.userData.isOverlay = true;
      overlay.renderOrder = 10;
      overlay.raycast = () => {};
      node.add(overlay);
      this.attached.push({ node, overlay });
    }
    this.update();
  }

  update(changes = {}) {
    this.settings = { ...this.settings, ...changes };
    const s = this.settings;
    const u = this.material.uniforms;
    u.uMode.value = s.mode === 'provenance' ? 2 : 1;
    u.uOpacity.value = s.opacity;
    u.uBands.value = (PALETTE_COLORS[s.palette] || PALETTE_COLORS.standard).map(rgb);
    u.uVisible.value = [0, 1, 2, 3, 4, 5].map((c) => (s.hiddenProvenance.includes(c) ? 0 : 1));
    u.uMin.value = s.min;
    u.uMax.value = s.max;
    u.uThreshold.value = s.threshold;
    u.uShowUnknown.value = s.showUnknown ? 1 : 0;
    u.uOnlyLow.value = s.onlyLow ? 1 : 0;
    u.uOnlyUnknown.value = s.onlyUnknown ? 1 : 0;
    u.uOnlyRepaired.value = s.onlyRepaired ? 1 : 0;
    for (const { node, overlay } of this.attached) {
      overlay.visible = s.mode !== 'off';
      // "Isolate" hides the model itself so only the filtered faces remain.
      for (const material of [].concat(node.material)) material.colorWrite = !(s.isolate && s.mode !== 'off');
    }
  }

  detach() {
    for (const { node, overlay } of this.attached) {
      node.remove(overlay);
      overlay.geometry.dispose();
      for (const material of [].concat(node.material)) material.colorWrite = true;
    }
    this.attached = [];
  }

  dispose() {
    this.detach();
    this.material.dispose();
  }
}
