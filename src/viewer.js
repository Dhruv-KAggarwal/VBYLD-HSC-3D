/**
 * Skyforge 3D viewer.
 * ---------------------------------------------------------------------------
 * Three display modes over one scene:
 *
 *   mesh   — the textured OpenMVS mesh (GLB). Measurable, exportable.
 *   splat  — a 3D Gaussian splat (PLY). Photoreal, the one you can fly down a
 *            street in and read the front of a building.
 *   cloud  — the dense point cloud (PLY). Diagnostic: shows what was actually
 *            measured, with no surface interpolation hiding the gaps.
 *
 * Three navigation modes, because orbit alone is why the old viewer "did
 * nothing useful on rotating":
 *
 *   orbit  — turntable around the scene. Good for an overview, useless at
 *            ground level.
 *   fly    — WASD + mouse-look, no collision. Gets you between buildings.
 *   walk   — fly with the camera pinned to a height above the surface below,
 *            so you travel the streets at eye level.
 *
 * The old viewer actively prevented street level: maxPolarAngle locked the
 * camera above the horizon and minDistance 8 stopped it approaching the
 * surface. Both are gone.
 *
 * It also replaced every material with an unlit MeshBasicMaterial. That hid a
 * real bug (the GLB exporter emitted no KHR_materials_unlit, so the model
 * rendered black in every *other* viewer) and threw away normals. Now the GLB
 * is written correctly, so materials are kept as authored and shading is a
 * user-facing toggle.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { GaussianSplatPLYLoader } from 'three/addons/loaders/GaussianSplatPLYLoader.js';
import { GaussianSplat } from 'three/addons/objects/GaussianSplat.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

const UP = new THREE.Vector3(0, 1, 0);

export class ModelViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f1821);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100000);
    this.camera.position.set(140, 90, 160);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    // Reconstructed photography already contains the real lighting of the
    // scene. Cinematic tone mapping re-grades it and the colours stop matching
    // the source video, which is exactly the complaint this viewer exists to
    // answer. Neutral by default; the user can opt into filmic.
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));

    this.root = new THREE.Group();
    this.scene.add(this.root);

    // Lights matter again now that materials are kept as authored. A textured
    // mesh exported with KHR_materials_unlit ignores them (correct: the photo
    // already has the light baked in); a vertex-coloured or shaded view uses
    // them.
    this.hemi = new THREE.HemisphereLight(0xd7ecff, 0x3f4a32, 0.9);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfff1d2, 1.4);
    this.sun.position.set(80, 140, 50);
    this.scene.add(this.sun);
    this.fill = new THREE.DirectionalLight(0x8fb7ff, 0.3);
    this.fill.position.set(-90, 40, -70);
    this.scene.add(this.fill);

    this.grid = new THREE.GridHelper(400, 24, 0x4d6f7a, 0x2a4a52);
    this.scene.add(this.grid);

    // --- Navigation ---------------------------------------------------------
    this.orbit = new OrbitControls(this.camera, canvas);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.07;
    this.orbit.screenSpacePanning = false;
    // Deliberately NO maxPolarAngle and NO minDistance. Going below the horizon
    // and right up to a wall is the entire point.
    this.orbit.minDistance = 0.01;
    this.orbit.addEventListener('change', () => this.invalidate());

    this.pointerLock = new PointerLockControls(this.camera, canvas);
    this.pointerLock.addEventListener('change', () => this.invalidate());
    this.scene.add(this.pointerLock.object);

    this.navMode = 'orbit';
    this.keys = new Set();
    this.moveSpeed = 1;
    this.walkHeight = 1.7; // metres: eye level, since the model is metric
    this.boundsRadius = 100;

    // --- Measurement --------------------------------------------------------
    this.measureMode = 'off'; // off | distance | height | area
    this.measurePoints = [];
    this.measureGroup = new THREE.Group();
    this.scene.add(this.measureGroup);
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();

    // --- Display ------------------------------------------------------------
    this.layers = { mesh: true, splat: true, points: false, grid: true };
    this.shading = 'authored'; // authored | shaded | flat
    this.objects = { mesh: null, splat: null, cloud: null };
    this.pointSize = 0.08;

    this.onMeasure = null;
    this.onProgress = null;
    this.onNavChange = null;

    // --- Loaders ------------------------------------------------------------
    // Compressed-geometry and compressed-texture support, so a large mesh
    // streams instead of arriving as one uncompressed blob.
    this.dracoLoader = new DRACOLoader().setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
    this.ktx2Loader = new KTX2Loader().setTranscoderPath('https://cdn.jsdelivr.net/npm/three@0.186.0/examples/jsm/libs/basis/');
    try { this.ktx2Loader.detectSupport(this.renderer); } catch { /* optional */ }
    this.gltfLoader = new GLTFLoader()
      .setDRACOLoader(this.dracoLoader)
      .setKTX2Loader(this.ktx2Loader)
      .setMeshoptDecoder(MeshoptDecoder);
    this.plyLoader = new PLYLoader();
    this.splatLoader = new GaussianSplatPLYLoader();

    this.needsRender = true;
    this.clock = new THREE.Clock();
    this.resize();
    this.bindInput();
    window.addEventListener('resize', () => this.resize());
    this.loop();
  }

  /** Request one more frame. The loop is otherwise idle, not a busy spinner. */
  invalidate() { this.needsRender = true; }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(2, rect.width);
    const height = Math.max(2, rect.height);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  // -------------------------------------------------------------------------
  // Input
  // -------------------------------------------------------------------------

  bindInput() {
    this.canvas.addEventListener('pointerdown', (event) => this.onPointerDown(event));
    window.addEventListener('keydown', (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      this.keys.add(event.code);
      // Shift sprints; a town is large and walking it at 1x is tedious.
      if (event.code === 'Escape' && this.navMode !== 'orbit') this.setNavMode('orbit');
    });
    window.addEventListener('keyup', (event) => this.keys.delete(event.code));
    window.addEventListener('blur', () => this.keys.clear());
    this.canvas.addEventListener('wheel', (event) => {
      if (this.navMode === 'orbit') return;
      // In fly/walk the wheel changes travel speed rather than zooming, which
      // would fight the mouse-look.
      event.preventDefault();
      this.moveSpeed = Math.min(200, Math.max(0.05, this.moveSpeed * (event.deltaY < 0 ? 1.15 : 0.87)));
      this.onNavChange?.({ mode: this.navMode, speed: this.moveSpeed });
    }, { passive: false });
  }

  /**
   * orbit | fly | walk.
   *
   * fly and walk take a pointer lock, so they can only start from a real user
   * gesture; the caller wires them to a button.
   */
  setNavMode(mode) {
    if (mode === this.navMode) return;
    this.navMode = mode;
    if (mode === 'orbit') {
      this.orbit.enabled = true;
      if (this.pointerLock.isLocked) this.pointerLock.unlock();
      // Keep looking at whatever was in front, so the switch is not jarring.
      const ahead = new THREE.Vector3();
      this.camera.getWorldDirection(ahead);
      this.orbit.target.copy(this.camera.position).addScaledVector(ahead, this.boundsRadius * 0.3);
      this.orbit.update();
    } else {
      this.orbit.enabled = false;
      this.pointerLock.lock();
      if (mode === 'walk') this.snapToGround();
    }
    this.onNavChange?.({ mode, speed: this.moveSpeed });
    this.invalidate();
  }

  /** Drop the camera to eye level above whatever surface is below it. */
  snapToGround() {
    const target = this.objects.mesh || this.objects.cloud;
    if (!target) return;
    const from = new THREE.Vector3(this.camera.position.x, this.camera.position.y + this.boundsRadius, this.camera.position.z);
    this.raycaster.set(from, new THREE.Vector3(0, -1, 0));
    this.raycaster.far = this.boundsRadius * 4;
    const hit = this.raycaster.intersectObject(target, true)[0];
    if (hit) this.camera.position.y = hit.point.y + this.walkHeight;
    this.invalidate();
  }

  updateMovement(delta) {
    if (this.navMode === 'orbit' || !this.keys.size) return;
    const sprint = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 4 : 1;
    const step = this.moveSpeed * sprint * delta * 10;

    const forward = new THREE.Vector3();
    this.camera.getWorldDirection(forward);
    if (this.navMode === 'walk') { forward.y = 0; forward.normalize(); }
    const right = new THREE.Vector3().crossVectors(forward, UP).normalize();

    const move = new THREE.Vector3();
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) move.add(forward);
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) move.sub(forward);
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) move.add(right);
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) move.sub(right);
    if (this.navMode === 'fly') {
      if (this.keys.has('KeyE') || this.keys.has('Space')) move.add(UP);
      if (this.keys.has('KeyQ')) move.sub(UP);
    }
    if (move.lengthSq() === 0) return;

    this.camera.position.addScaledVector(move.normalize(), step);
    if (this.navMode === 'walk') this.snapToGround();
    this.invalidate();
  }

  // -------------------------------------------------------------------------
  // Loading
  // -------------------------------------------------------------------------

  /** Fetch with byte progress, since a textured town mesh is tens of MB. */
  async fetchWithProgress(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const total = Number(response.headers.get('content-length')) || 0;
    if (!response.body) return new Uint8Array(await response.arrayBuffer());

    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      this.onProgress?.(total ? loaded / total : 0, loaded, total);
    }
    const packed = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) { packed.set(chunk, offset); offset += chunk.length; }
    return packed;
  }

  async loadGlb(url) {
    const packed = await this.fetchWithProgress(url);
    const gltf = await new Promise((resolve, reject) => {
      this.gltfLoader.parse(packed.buffer, '', resolve, reject);
    });

    this.disposeObject(this.objects.mesh);
    const model = gltf.scene;
    let hasTexture = false;
    let hasVertexColor = false;
    model.traverse((node) => {
      if (node.isPoints) {
        node.material.size = this.pointSize;
        node.material.sizeAttenuation = true;
        node.material.vertexColors = Boolean(node.geometry.getAttribute('color'));
      }
      if (!node.isMesh) return;
      const material = node.material;
      if (material?.map) {
        hasTexture = true;
        material.map.colorSpace = THREE.SRGBColorSpace;
        material.map.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
        material.map.needsUpdate = true;
      }
      if (node.geometry?.getAttribute('color')) hasVertexColor = true;
      if (material) {
        material.side = THREE.DoubleSide;
        material.needsUpdate = true;
      }
    });

    this.objects.mesh = model;
    this.root.add(model);
    this.applyShading();
    this.fit(model);
    this.applyLayers();
    return { gltf, hasTexture, hasVertexColor };
  }

  /** Dense point cloud, so the raw measurement is inspectable on its own. */
  async loadPointCloud(url) {
    const packed = await this.fetchWithProgress(url);
    const geometry = this.plyLoader.parse(packed.buffer);
    geometry.computeBoundingBox();
    this.disposeObject(this.objects.cloud);
    const material = new THREE.PointsMaterial({
      size: this.pointSize,
      sizeAttenuation: true,
      vertexColors: Boolean(geometry.getAttribute('color')),
      color: geometry.getAttribute('color') ? 0xffffff : 0xaab6c2,
      toneMapped: false,
    });
    const points = new THREE.Points(geometry, material);
    points.name = 'DenseCloud';
    this.objects.cloud = points;
    this.root.add(points);
    if (!this.objects.mesh) this.fit(points);
    this.applyLayers();
    return { count: geometry.getAttribute('position').count };
  }

  /**
   * 3D Gaussian splat.
   *
   * three 0.186 renders splats natively (GaussianSplat extends Mesh), so this
   * needs no extra renderer and no third-party library. There is no LOD here,
   * so a very large scene is limited by VRAM rather than streamed.
   */
  async loadSplat(url) {
    const packed = await this.fetchWithProgress(url);
    const geometry = this.splatLoader.parse(packed.buffer);
    this.disposeObject(this.objects.splat);
    const splat = new GaussianSplat(geometry);
    splat.name = 'GaussianSplat';
    this.objects.splat = splat;
    this.root.add(splat);
    if (!this.objects.mesh) this.fit(splat);
    this.applyLayers();
    return { count: geometry.getAttribute('position')?.count || 0 };
  }

  disposeObject(object) {
    if (!object) return;
    object.traverse?.((node) => {
      node.geometry?.dispose?.();
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of materials) {
        if (!material) continue;
        for (const key of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'emissiveMap', 'aoMap']) {
          material[key]?.dispose?.();
        }
        material.dispose?.();
      }
    });
    object.removeFromParent?.();
  }

  clearModel() {
    for (const key of Object.keys(this.objects)) {
      this.disposeObject(this.objects[key]);
      this.objects[key] = null;
    }
    this.clearMeasure();
    this.invalidate();
  }

  // -------------------------------------------------------------------------
  // Framing and display
  // -------------------------------------------------------------------------

  fit(object) {
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1;
    this.boundsRadius = radius;

    // A town model spans hundreds of metres but we want to stand next to a
    // wall, so the near plane has to be small in absolute terms, not relative.
    this.camera.near = Math.min(0.05, radius / 5000);
    this.camera.far = radius * 60;
    this.camera.updateProjectionMatrix();

    this.camera.position.set(
      centre.x + radius * 1.5,
      centre.y + radius * 1.1,
      centre.z + radius * 1.5,
    );
    this.orbit.target.copy(centre);
    this.orbit.update();

    this.sun.position.set(centre.x + radius, centre.y + radius * 2, centre.z + radius * 0.6);
    this.sun.target.position.copy(centre);
    this.sun.target.updateMatrixWorld();

    this.grid.position.set(centre.x, box.min.y, centre.z);
    this.grid.scale.setScalar(Math.max(0.1, radius / 200));

    // Default travel speed scaled to the scene, so WASD is usable immediately
    // on both a courtyard and a town.
    this.moveSpeed = Math.max(0.1, radius / 60);
    this.pointSize = Math.max(0.01, radius / 900);
    this.invalidate();
  }

  setLayers(layers) {
    this.layers = { ...this.layers, ...layers };
    this.applyLayers();
  }

  applyLayers() {
    if (this.objects.mesh) this.objects.mesh.visible = this.layers.mesh !== false;
    if (this.objects.splat) this.objects.splat.visible = this.layers.splat !== false;
    if (this.objects.cloud) this.objects.cloud.visible = this.layers.points === true;
    this.grid.visible = this.layers.grid !== false;
    this.invalidate();
  }

  /**
   * authored — use the material the exporter wrote (unlit for a baked atlas).
   * shaded   — force lit PBR, which shows geometry relief the flat photo hides.
   * flat     — force unlit, for comparing colour against the source video.
   */
  setShading(mode) {
    this.shading = mode;
    this.applyShading();
  }

  applyShading() {
    const model = this.objects.mesh;
    if (!model) return;
    model.traverse((node) => {
      if (!node.isMesh || !node.material) return;
      if (!node.userData.authoredMaterial) node.userData.authoredMaterial = node.material;
      const authored = node.userData.authoredMaterial;

      if (this.shading === 'authored') {
        node.material = authored;
      } else if (this.shading === 'flat') {
        node.material = node.userData.flatMaterial ||= new THREE.MeshBasicMaterial({
          map: authored.map || null,
          color: authored.map ? 0xffffff : (authored.color || new THREE.Color(0xffffff)),
          vertexColors: Boolean(node.geometry.getAttribute('color')) && !authored.map,
          side: THREE.DoubleSide,
          toneMapped: false,
        });
      } else {
        node.material = node.userData.shadedMaterial ||= new THREE.MeshStandardMaterial({
          map: authored.map || null,
          color: authored.map ? 0xffffff : (authored.color || new THREE.Color(0xffffff)),
          vertexColors: Boolean(node.geometry.getAttribute('color')) && !authored.map,
          roughness: 0.95,
          metalness: 0,
          side: THREE.DoubleSide,
        });
      }
      node.material.needsUpdate = true;
    });
    this.invalidate();
  }

  // -------------------------------------------------------------------------
  // Measurement
  // -------------------------------------------------------------------------

  setMeasureMode(mode) {
    this.measureMode = mode;
    this.clearMeasure();
    this.onMeasure?.({ mode, message: mode === 'off' ? '' : measurePrompt(mode) });
  }

  onPointerDown(event) {
    if (this.measureMode === 'off' || event.button !== 0) return;
    if (this.navMode !== 'orbit' && !this.pointerLock.isLocked) return;

    const rect = this.canvas.getBoundingClientRect();
    if (this.navMode === 'orbit') {
      this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      this.raycaster.setFromCamera(this.pointer, this.camera);
    } else {
      // In fly/walk the cursor is locked to the centre of the view.
      this.raycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);
    }
    this.raycaster.far = this.boundsRadius * 60;

    const targets = [this.objects.mesh, this.objects.cloud].filter(Boolean);
    if (!targets.length) return;
    const hit = this.raycaster.intersectObjects(targets, true)[0];
    if (!hit) return;

    this.measurePoints.push(hit.point.clone());
    this.addMarker(hit.point);
    this.evaluateMeasurement();
  }

  evaluateMeasurement() {
    const pts = this.measurePoints;
    if (this.measureMode === 'distance' && pts.length >= 2) {
      const [a, b] = pts.slice(-2);
      this.addLine(a, b);
      const metres = a.distanceTo(b);
      this.onMeasure?.({
        mode: 'distance',
        metres,
        message: `Distance ${formatMetres(metres)} (horizontal ${formatMetres(Math.hypot(b.x - a.x, b.z - a.z))}, vertical ${formatMetres(Math.abs(b.y - a.y))})`,
      });
      this.measurePoints = [];
    } else if (this.measureMode === 'height' && pts.length >= 2) {
      const [a, b] = pts.slice(-2);
      // Draw the actual vertical, not the slant, so what is measured is visible.
      const foot = new THREE.Vector3(b.x, a.y, b.z);
      this.addLine(a, foot);
      this.addLine(foot, b);
      const metres = Math.abs(b.y - a.y);
      this.onMeasure?.({ mode: 'height', metres, message: `Height ${formatMetres(metres)}` });
      this.measurePoints = [];
    } else if (this.measureMode === 'area' && pts.length >= 2) {
      this.addLine(pts[pts.length - 2], pts[pts.length - 1]);
      if (pts.length >= 3) {
        const metres = polygonAreaXZ(pts);
        this.onMeasure?.({
          mode: 'area',
          metres,
          message: `Area ${metres.toFixed(metres < 100 ? 2 : 0)} m² over ${pts.length} points — click to add, double-click to finish`,
        });
      }
    } else {
      this.onMeasure?.({ mode: this.measureMode, message: measurePrompt(this.measureMode) });
    }
    this.invalidate();
  }

  addMarker(point) {
    const scale = Math.max(0.05, this.boundsRadius / 320);
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(scale, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0x5ef2b0, toneMapped: false, depthTest: false }),
    );
    marker.renderOrder = 999;
    marker.position.copy(point);
    this.measureGroup.add(marker);
    this.invalidate();
  }

  addLine(a, b) {
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([a, b]),
      new THREE.LineBasicMaterial({ color: 0x5ef2b0, toneMapped: false, depthTest: false }),
    );
    line.renderOrder = 998;
    this.measureGroup.add(line);
    this.invalidate();
  }

  clearMeasure() {
    this.measurePoints = [];
    while (this.measureGroup.children.length) {
      const child = this.measureGroup.children[0];
      child.geometry?.dispose?.();
      child.material?.dispose?.();
      this.measureGroup.remove(child);
    }
    this.invalidate();
  }

  // -------------------------------------------------------------------------

  loop() {
    requestAnimationFrame(() => this.loop());
    const delta = Math.min(0.1, this.clock.getDelta());
    this.updateMovement(delta);
    if (this.navMode === 'orbit' && this.orbit.enabled) {
      // Damping keeps producing motion for a few frames after input stops.
      if (this.orbit.update()) this.invalidate();
    }
    if (!this.needsRender) return;
    this.needsRender = false;
    this.renderer.render(this.scene, this.camera);
  }
}

function measurePrompt(mode) {
  if (mode === 'distance') return 'Click two points to measure the distance between them';
  if (mode === 'height') return 'Click the base then the top of a structure';
  if (mode === 'area') return 'Click around a footprint; area is the horizontal projection';
  return '';
}

function formatMetres(metres) {
  if (metres < 1) return `${(metres * 100).toFixed(1)} cm`;
  if (metres < 1000) return `${metres.toFixed(2)} m`;
  return `${(metres / 1000).toFixed(3)} km`;
}

/** Shoelace area of the horizontal projection. */
function polygonAreaXZ(points) {
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    sum += a.x * b.z - b.x * a.z;
  }
  return Math.abs(sum) / 2;
}
