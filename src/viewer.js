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
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { MeshBVH } from '../lib/bvh.mjs';
import { FaceOverlay } from './viewer/overlay-renderer.js';
import { ComparisonController } from './viewer/comparison-controller.js';
import { LodManager } from './viewer/lod-manager.js';
import { SplatView } from './viewer/splat-view.js';

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
    // The browser can take the mouse back on its own (Esc, switching window) or
    // refuse to give it. Either way fly and walk cannot continue.
    this.pointerLock.addEventListener('unlock', () => this.setNavMode('orbit'));
    document.addEventListener('pointerlockerror', () => this.setNavMode('orbit'));
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
    this.layers = { mesh: true, splat: false, points: false, grid: true };
    this.shading = 'authored'; // authored | shaded | flat
    this.objects = { mesh: null, cloud: null };
    this.pointSize = 0.08;

    this.onMeasure = null;
    this.onProgress = null;
    this.onNavChange = null;
    this.onInspect = null;
    this.onLodState = null;
    this.onLodChange = null;
    this.onModelShown = null;

    // --- Enhancements -------------------------------------------------------
    this.overlay = new FaceOverlay();
    this.comparison = new ComparisonController(this);
    this.lod = null;
    this.inspect = false;
    this.wireframe = false;
    this.scaleReliable = true;
    this.highlight = null;
    this.perf = { fps: null, frameMs: null, last: 0 };
    // Camera positions of the source video and the area they filmed.
    this.views = null;
    this.focusClip = false;
    this.defaultFov = 55;

    // --- Loaders ------------------------------------------------------------
    // Compressed-geometry and compressed-texture support, so a large mesh
    // streams instead of arriving as one uncompressed blob.
    // Decoders are served by this app (copied from three at build time), so a
    // compressed asset opens without a network connection.
    this.dracoLoader = new DRACOLoader().setDecoderPath('/decoders/draco/');
    this.ktx2Loader = new KTX2Loader().setTranscoderPath('/decoders/basis/');
    try { this.ktx2Loader.detectSupport(this.renderer); } catch { /* optional */ }
    this.gltfLoader = new GLTFLoader()
      .setDRACOLoader(this.dracoLoader)
      .setKTX2Loader(this.ktx2Loader)
      .setMeshoptDecoder(MeshoptDecoder);
    this.plyLoader = new PLYLoader();
    this.splatView = new SplatView(canvas);

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
    this.canvas.addEventListener('pointerdown', (event) => {
      this.pressed = { x: event.clientX, y: event.clientY, button: event.button };
      this.onPointerDown(event);
    });
    // Inspection acts on a click, not on the start of an orbit drag.
    this.canvas.addEventListener('pointerup', (event) => {
      const start = this.pressed;
      this.pressed = null;
      if (!this.inspect || this.measureMode !== 'off' || !start || start.button !== 0) return;
      if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return;
      this.inspectAt(event);
    });
    window.addEventListener('keydown', (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      this.keys.add(event.code);
      // Space climbs in fly mode; it must not also press a focused button.
      if (this.navMode !== 'orbit' && event.code === 'Space') event.preventDefault();
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
      // A focused slider would otherwise swallow the movement keys.
      document.activeElement?.blur?.();
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
    const hit = target.isPoints
      ? this.raycaster.intersectObject(target, true)[0]
      : this.pickModel(this.raycaster, target);
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
  async fetchWithProgress(url, { signal = null, quiet = false } = {}) {
    const response = await fetch(url, signal ? { signal } : undefined);
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
      if (!quiet) this.onProgress?.(total ? loaded / total : 0, loaded, total);
    }
    const packed = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) { packed.set(chunk, offset); offset += chunk.length; }
    return packed;
  }

  /** Download and parse a GLB without putting it in the scene. */
  async loadModelData(url, { signal = null, quiet = true } = {}) {
    const packed = await this.fetchWithProgress(url, { signal, quiet });
    const gltf = await new Promise((resolve, reject) => {
      this.gltfLoader.parse(packed.buffer, '', resolve, reject);
    });
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
        // Estimated ground is one-sided: from below it must not hide the model.
        material.side = node.geometry?.userData.skyforgeGround ? THREE.FrontSide : THREE.DoubleSide;
        material.needsUpdate = true;
      }
    });
    model.updateMatrixWorld(true);
    return { model, gltf, hasTexture, hasVertexColor };
  }

  async loadGlb(url) {
    const { model, gltf, hasTexture, hasVertexColor } = await this.loadModelData(url);
    this.showModel(model, { fit: true });
    return { gltf, hasTexture, hasVertexColor };
  }

  /**
   * Put a model on screen. The camera, navigation mode and measurements are
   * untouched unless `fit` is set. `owned: false` models belong to a cache and
   * are not disposed when replaced.
   */
  showModel(model, { fit = false, owned = true } = {}) {
    const previous = this.objects.mesh;
    this.overlay.detach();
    if (previous && previous !== model) {
      if (previous.userData.owned) this.disposeObject(previous);
      else previous.removeFromParent();
    }
    model.userData.owned = owned;
    this.objects.mesh = model;
    this.root.add(model);
    model.updateMatrixWorld(true);
    this.applyShading();
    if (fit) this.fit(model);
    this.applyLayers();
    this.onModelShown?.();
  }

  releaseModel(model) {
    if (this.objects.mesh !== model) return;
    this.overlay.detach();
    model.removeFromParent();
    this.objects.mesh = null;
    this.invalidate();
  }

  cameraState() {
    return { position: this.camera.position.clone(), quaternion: this.camera.quaternion.clone(), target: this.orbit.target.clone() };
  }

  restoreCamera(saved) {
    this.camera.position.copy(saved.position);
    this.camera.quaternion.copy(saved.quaternion);
    this.orbit.target.copy(saved.target);
    if (this.navMode === 'orbit') this.orbit.update();
    this.invalidate();
  }

  /** Switch to automatic level-of-detail display for the current job. */
  async openLod(manifest, urlFor) {
    this.closeLod();
    this.lod = new LodManager(this, { onState: (state) => this.onLodState?.(state) });
    await this.lod.open(manifest, urlFor);
    if (this.measureMode !== 'off') this.lod.setMeasuring(true);
  }

  closeLod() {
    if (!this.lod) return;
    this.lod.dispose();
    this.lod = null;
    this.onLodState?.(null);
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
   * 3D Gaussian splat, drawn on its own canvas (see viewer/splat-view.js).
   * It is stored in the reconstruction frame, so it needs the job's transform
   * (from setViews) to overlay the mesh.
   */
  async loadSplat(url) {
    const generation = this.generation;
    const packed = await this.fetchWithProgress(url);
    const loaded = await this.splatView.load(packed.buffer, this.views?.reconstructionToViewer, this.views?.focus);
    // The operator may have opened another job while this downloaded.
    if (generation !== this.generation) { this.splatView.clear(); throw new Error('Superseded by another model.'); }
    this.applyLayers();
    return loaded;
  }

  disposeObject(object) {
    if (!object) return;
    object.traverse?.((node) => {
      node.geometry?.dispose?.();
      if (node.geometry?.userData) delete node.geometry.userData.bvh;
      const { authoredMaterial, flatMaterial, shadedMaterial } = node.userData || {};
      const materials = new Set([].concat(node.material, authoredMaterial, flatMaterial, shadedMaterial));
      // The overlay material is shared and outlives any one model.
      if (node.userData?.isOverlay) materials.clear();
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

  clearModel({ keepMeasure = false } = {}) {
    this.comparison.disable();
    this.closeLod();
    this.overlay.detach();
    this.clearHighlight();
    this.generation = (this.generation || 0) + 1;
    this.splatView.clear();
    for (const key of Object.keys(this.objects)) {
      this.disposeObject(this.objects[key]);
      this.objects[key] = null;
    }
    if (!keepMeasure) this.clearMeasure();
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
    // Near/far ratio kept modest, or distant surfaces flicker into each other.
    this.camera.near = Math.min(0.5, Math.max(0.05, radius / 5000));
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

  // -------------------------------------------------------------------------
  // Drone views and the filmed area
  // -------------------------------------------------------------------------

  /** Adopt the source cameras of the loaded job (null clears them). */
  setViews(data, { jump = true } = {}) {
    this.views = data;
    if (!data) { this.setFocusClip(false); return; }
    this.splatView.align(data.reconstructionToViewer);
    // Speeds, markers and walking scale with the filmed area, not the horizon.
    this.boundsRadius = Math.max(10, data.focus.radius);
    this.moveSpeed = Math.max(0.1, this.boundsRadius / 60);
    const y = data.focus.center[1];
    this.grid.position.set(data.focus.center[0], Math.min(this.grid.position.y, y), data.focus.center[2]);
    this.grid.scale.setScalar(Math.max(0.1, this.boundsRadius / 200));
    if (jump) this.showDroneView(data.start);
  }

  /** Look through the camera that took one video frame. */
  showDroneView(index) {
    const view = this.views?.views[index];
    if (!view) return;
    this.setNavMode('orbit');
    this.clearHighlight();
    // Match the video's framing: widen if the window is narrower than 16:9.
    const widen = Math.max(1, (16 / 9) / Math.max(0.2, this.camera.aspect));
    this.camera.fov = Math.min(90, (2 * Math.atan(Math.tan((view.fov * Math.PI) / 360) * widen) * 180) / Math.PI);
    this.camera.updateProjectionMatrix();
    this.camera.position.set(...view.position);
    this.orbit.target.set(...view.target);
    this.orbit.update();
    this.invalidate();
  }

  /** The whole filmed area from above and to one side. */
  showOverview() {
    if (!this.views) { if (this.objects.mesh) this.fit(this.objects.mesh); return; }
    const { focus, views, start } = this.views;
    this.setNavMode('orbit');
    const target = new THREE.Vector3(...focus.center);
    const from = new THREE.Vector3(...views[start].position).sub(target);
    from.y = 0;
    if (from.lengthSq() < 1e-6) from.set(1, 0, 1);
    from.normalize().multiplyScalar(focus.radius * 1.5);
    from.y = focus.radius * 1.1;
    this.camera.fov = this.defaultFov;
    this.camera.updateProjectionMatrix();
    this.camera.position.copy(target).add(from);
    this.orbit.target.copy(target);
    this.orbit.update();
    this.invalidate();
  }

  /** Hide everything outside the filmed area. Display only; nothing is removed. */
  setFocusClip(on) {
    this.focusClip = Boolean(on && this.views);
    const planes = [];
    if (this.focusClip) {
      const { min, max } = this.views.focus;
      planes.push(
        new THREE.Plane(new THREE.Vector3(1, 0, 0), -min[0]),
        new THREE.Plane(new THREE.Vector3(-1, 0, 0), max[0]),
        new THREE.Plane(new THREE.Vector3(0, 0, 1), -min[1]),
        new THREE.Plane(new THREE.Vector3(0, 0, -1), max[1]),
      );
    }
    this.renderer.clippingPlanes = planes;
    this.invalidate();
  }

  isVisiblePoint(point) {
    return this.renderer.clippingPlanes.every((plane) => plane.distanceToPoint(point) >= 0);
  }

  setLayers(layers) {
    this.layers = { ...this.layers, ...layers };
    this.applyLayers();
  }

  applyLayers() {
    const comparing = this.comparison.active;
    if (this.objects.mesh) this.objects.mesh.visible = !comparing && this.layers.mesh !== false;
    this.splatView.setVisible(!comparing && this.layers.splat === true);
    if (this.objects.cloud) this.objects.cloud.visible = !comparing && this.layers.points === true;
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
      if (!node.isMesh || !node.material || node.userData.isOverlay) return;
      if (!node.userData.authoredMaterial) node.userData.authoredMaterial = node.material;
      const authored = node.userData.authoredMaterial;

      if (this.shading === 'authored') {
        node.material = authored;
      } else if (this.shading === 'flat') {
        node.material = node.userData.flatMaterial ||= new THREE.MeshBasicMaterial({
          map: authored.map || null,
          color: authored.map ? 0xffffff : (authored.color || new THREE.Color(0xffffff)),
          vertexColors: Boolean(node.geometry.getAttribute('color')) && !authored.map,
          side: authored.side,
          toneMapped: false,
        });
      } else {
        node.material = node.userData.shadedMaterial ||= new THREE.MeshStandardMaterial({
          map: authored.map || null,
          color: authored.map ? 0xffffff : (authored.color || new THREE.Color(0xffffff)),
          vertexColors: Boolean(node.geometry.getAttribute('color')) && !authored.map,
          roughness: 0.95,
          metalness: 0,
          side: authored.side,
        });
      }
      node.material.wireframe = this.wireframe;
      node.material.needsUpdate = true;
    });
    // Materials were swapped, so re-apply what the overlay does to them.
    this.overlay.update();
    this.invalidate();
  }

  setWireframe(on) {
    this.wireframe = on;
    this.applyShading();
    for (const model of this.comparison.list()) {
      model.traverse((node) => { if (node.isMesh && !node.userData.isOverlay) node.material.wireframe = on; });
    }
    this.invalidate();
  }

  // -------------------------------------------------------------------------
  // Overlays, comparison, inspection
  // -------------------------------------------------------------------------

  /** Attach per-face data (confidence and/or provenance) to the shown model. */
  setOverlayData(data) {
    if (!data || !this.objects.mesh) this.overlay.detach();
    else this.overlay.attach(this.objects.mesh, data);
    this.invalidate();
  }

  setOverlay(settings) {
    this.overlay.update(settings);
    this.invalidate();
  }

  setComparisonActive(active) {
    if (active) {
      this.overlay.detach();
      this.setWireframe(this.wireframe);
    }
    this.applyLayers();
  }

  setInspect(on) { this.inspect = on; }

  /** Nearest hit on a model's triangles, using a BVH built on first use. */
  pickModel(raycaster, model) {
    if (!this.focusClip) return this.pickAnywhere(raycaster, model);
    // Step past surfaces the focus filter hides.
    const origin = raycaster.ray.origin.clone();
    let travelled = 0;
    let found = null;
    for (let i = 0; i < 12; i += 1) {
      const hit = this.pickAnywhere(raycaster, model);
      if (!hit) break;
      if (this.isVisiblePoint(hit.point)) { found = { ...hit, distance: hit.distance + travelled }; break; }
      const step = hit.distance + Math.max(0.01, hit.distance * 1e-5);
      travelled += step;
      raycaster.ray.origin.addScaledVector(raycaster.ray.direction, step);
    }
    raycaster.ray.origin.copy(origin);
    return found;
  }

  pickAnywhere(raycaster, model) {
    let best = null;
    let running = 0;
    model.traverse((node) => {
      if (!node.isMesh || node.userData.isOverlay) return;
      const { geometry } = node;
      const count = (geometry.index ? geometry.index.count : geometry.attributes.position.count) / 3;
      const offset = geometry.userData.skyforgeFaceOffset ?? running;
      running += count;
      if (!geometry.userData.bvh) {
        const source = geometry.attributes.position;
        const positions = new Float32Array(source.count * 3);
        for (let i = 0; i < source.count; i += 1) {
          positions[i * 3] = source.getX(i);
          positions[i * 3 + 1] = source.getY(i);
          positions[i * 3 + 2] = source.getZ(i);
        }
        const indices = geometry.index
          ? geometry.index.array
          : Uint32Array.from({ length: source.count }, (_, i) => i);
        geometry.userData.bvh = new MeshBVH(positions, indices);
      }
      // The ray in the mesh's local frame; leaving the direction unnormalised
      // keeps the hit parameter equal to the world-space one.
      const inverse = node.matrixWorld.clone().invert();
      const origin = raycaster.ray.origin.clone().applyMatrix4(inverse);
      const tip = raycaster.ray.origin.clone().add(raycaster.ray.direction).applyMatrix4(inverse);
      const direction = tip.sub(origin);
      const hit = geometry.userData.bvh.raycast(
        origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, best ? best.distance : raycaster.far,
      );
      if (hit) {
        best = {
          distance: hit.t,
          point: raycaster.ray.at(hit.t, new THREE.Vector3()),
          node,
          face: offset + hit.face,
        };
      }
    });
    return best;
  }

  aimAt(event) {
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
  }

  inspectAt(event) {
    if (!this.objects.mesh || this.comparison.active) return;
    this.aimAt(event);
    const hit = this.pickModel(this.raycaster, this.objects.mesh);
    if (hit) this.onInspect?.({ face: hit.face, point: hit.point });
  }

  /** Frame a low-confidence region from the side its surface faces. */
  focusRegion({ boundingBox, centre, normal }) {
    this.setNavMode('orbit');
    const box = new THREE.Box3(new THREE.Vector3(...boundingBox.min), new THREE.Vector3(...boundingBox.max));
    const size = box.getSize(new THREE.Vector3()).length() || this.boundsRadius * 0.1;
    const target = new THREE.Vector3(...centre);
    const direction = new THREE.Vector3(...normal);
    if (direction.y < 0.15) direction.y = 0.15 + Math.max(0, -direction.y);
    direction.normalize();
    this.camera.position.copy(target).addScaledVector(direction, Math.max(size * 1.4, this.boundsRadius * 0.05));
    this.orbit.target.copy(target);
    this.orbit.update();
    this.clearHighlight();
    this.highlight = new THREE.Box3Helper(box, 0xffffff);
    this.highlight.material.depthTest = false;
    this.scene.add(this.highlight);
    this.invalidate();
  }

  clearHighlight() {
    if (!this.highlight) return;
    this.scene.remove(this.highlight);
    this.highlight.geometry.dispose();
    this.highlight.material.dispose();
    this.highlight = null;
    this.invalidate();
  }

  /** Distance that the LOD screen-space error is projected over. */
  viewDistance() {
    if (this.navMode === 'orbit') return this.camera.position.distanceTo(this.orbit.target);
    // Flying: assume surfaces are near, scaled to the scene.
    return this.boundsRadius * 0.08;
  }

  performanceStats() {
    return { fps: this.perf.fps, frameMs: this.perf.frameMs, age: performance.now() - this.perf.last };
  }

  // -------------------------------------------------------------------------
  // Measurement
  // -------------------------------------------------------------------------

  setMeasureMode(mode) {
    this.measureMode = mode;
    this.clearMeasure();
    this.lod?.setMeasuring(mode !== 'off');
    this.onMeasure?.({ mode, message: mode === 'off' ? '' : measurePrompt(mode) });
  }

  /** Report a measurement together with anything that limits its accuracy. */
  reportMeasurement(result) {
    const notes = [];
    if (this.measuredOn && !this.measuredOn.full) {
      notes.push(`measured on simplified level ${this.measuredOn.level}${this.measuredOn.loading ? ', full resolution still loading' : ''}`);
    }
    if (!this.scaleReliable) notes.push('scene scale is approximate');
    this.onMeasure?.({ ...result, message: notes.length ? `${result.message} · ${notes.join('; ')}` : result.message });
  }

  onPointerDown(event) {
    if (this.measureMode === 'off' || event.button !== 0) return;
    if (this.navMode !== 'orbit' && !this.pointerLock.isLocked) return;

    this.aimAt(event);

    // Measure against full-resolution geometry whatever level is displayed.
    let model = this.objects.mesh;
    this.measuredOn = null;
    if (this.comparison.active) {
      const [original, repaired] = this.comparison.list();
      model = this.comparison.mode === 'original' ? original : repaired;
    } else if (this.lod) {
      this.measuredOn = this.lod.measurementModel();
      model = this.measuredOn?.model || model;
    }
    const hits = [];
    if (model) hits.push(this.pickModel(this.raycaster, model));
    if (this.objects.cloud?.visible) hits.push(this.raycaster.intersectObject(this.objects.cloud, true)[0]);
    const hit = hits.filter(Boolean).sort((a, b) => a.distance - b.distance)[0];
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
      this.reportMeasurement({
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
      this.reportMeasurement({ mode: 'height', metres, message: `Height ${formatMetres(metres)}` });
      this.measurePoints = [];
    } else if (this.measureMode === 'area' && pts.length >= 2) {
      this.addLine(pts[pts.length - 2], pts[pts.length - 1]);
      if (pts.length >= 3) {
        const metres = polygonAreaXZ(pts);
        this.reportMeasurement({
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
    const now = performance.now();
    this.lod?.update(now);
    if (!this.needsRender) return;
    this.needsRender = false;
    // Frame rate is only meaningful across consecutive frames; the loop idles
    // when nothing changes.
    const gap = now - this.perf.last;
    if (gap > 0 && gap < 250) {
      const fps = 1000 / gap;
      this.perf.fps = this.perf.fps == null ? fps : this.perf.fps * 0.9 + fps * 0.1;
    }
    this.perf.last = now;
    if (this.splatView.visible) this.splatView.render(this.camera);
    else if (this.comparison.active) this.comparison.render(this.renderer, this.scene, this.camera);
    else this.renderer.render(this.scene, this.camera);
    const spent = performance.now() - now;
    this.perf.frameMs = this.perf.frameMs == null ? spent : this.perf.frameMs * 0.9 + spent * 0.1;
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
