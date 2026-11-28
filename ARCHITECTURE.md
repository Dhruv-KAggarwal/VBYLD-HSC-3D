# Skyforge architecture

A guided tour of how one drone video becomes a textured 3-D mesh. Read this
top-to-bottom once and the code in `lib/` will read like prose.

---

## 1. The pipeline at a glance

A reconstruction is a fixed sequence of **stages**. Each stage is one native
engine invocation (FFmpeg / COLMAP / OpenMVS) wrapped by `run()`, which streams
the engine's progress into a shared tracker and its full output into a log file.

```text
            stage key   engine            what it produces
 ┌────────┐
 │ video  │
 └───┬────┘
     │   extract      FFmpeg            runtime/<id>/frames/frame_*.jpg
     │   features     COLMAP            database.db (SIFT keypoints, on GPU)
     │   match        COLMAP            database.db (sequential matches, on GPU)
     │   mapper       COLMAP            sparse/0 (camera poses — Structure-from-Motion)
     │   undistort    COLMAP            dense/ (undistorted images + pinhole sparse model)
     │
     │   ── primary path: OpenMVS ─────────────────────────────────────────
     │   densify      DensifyPointCloud dense depth maps (CUDA) → scene_dense.ply
     │   mesh         ReconstructMesh   mesh.ply  (Delaunay + graph-cut surface)
     │                RefineMesh*       mesh_refined.ply  (*optional, detailed only)
     │   texture      TextureMesh       model.glb (per-face texture atlas → GLB)
     │
     │   ── fallback path: COLMAP (if OpenMVS absent / OOM) ───────────────
     │   densify      patch_match_stereo + stereo_fusion → fused.ply
     │
     │   products     (pure JS)         terrain.glb (DSM) + ortho + metadata
 ┌───┴────┐
 │ GLBs   │          model.glb (headline) · terrain.glb (DSM) · pointcloud.ply
 └────────┘
```

The stage list lives in `PIPELINE_STAGES` (`lib/reconstruct.mjs`). Weights on
each stage drive the progress bar and the self-correcting ETA.

---

## 2. Why two dense paths?

COLMAP solves **where each photo was taken** (poses). That is all the DSM needs.
But "a detailed 3-D texture" means a real triangle mesh coloured from the photos,
viewable from any angle — not a top-down height-field. **OpenMVS** is the engine
that produces that, so it is the primary path. The COLMAP dense-stereo path is
kept as a fallback for machines without OpenMVS or without enough VRAM, and it
still yields the DSM so a run is never empty-handed.

| | OpenMVS path (primary) | COLMAP path (fallback) |
|---|---|---|
| Dense cloud | `DensifyPointCloud` (CUDA) | `patch_match_stereo` + `stereo_fusion` |
| Surface | `ReconstructMesh` (+ `RefineMesh`) | — (DSM only) |
| Texture | `TextureMesh` → `model.glb` | — (vertex colour on DSM) |
| Headline output | `model.glb` (textured mesh) | `terrain.glb` (DSM) |

Both always build the gravity-aligned DSM in `export3d.mjs` from the dense cloud.

---

## 3. Module map (`lib/`)

| Module | Responsibility |
|---|---|
| `reconstruct.mjs` | Orchestrator. Owns the stage list, the progress tracker, `run()` (the spawn/timeout/cancel/log helper), quality presets (`qualitySettings`, `mvsFor`), and the top-level `reconstruct(job)` flow with the OpenMVS-primary / COLMAP-fallback branch. |
| `openmvs.mjs` | Drives the four OpenMVS tools (Interface → Densify → ReconstructMesh → RefineMesh → TextureMesh). Dependency-injected (`run`, `commands`) to avoid a circular import. Refinement failures degrade to a warning. |
| `glbtransform.mjs` | Re-orients the OpenMVS GLB (COLMAP frame → Y-up, metric scale) by injecting **one** parent-node transform matrix. The heavy BIN chunk is copied untouched; only the tiny JSON chunk is edited. |
| `export3d.mjs` | Turns the dense cloud into the DSM products: densify → trim footprint → build terrain grid → bake orthophoto → write `terrain.glb`, `terrain.ply`, `pointcloud.ply`, `orthomosaic.png`, `metadata.json`. Returns the gravity rotation + scale so the GLB can be aligned the same way. |
| `colmap.mjs` | Reads COLMAP's binary/text models: camera poses (`images.bin`), intrinsics (`cameras.bin`), and derives gravity-up from the solved cameras. |
| `terrain.mjs` | Builds the 2.5-D digital surface model: an occupancy/height grid from the point cloud with slope-aware despiking. |
| `texture.mjs` | Bakes the orthophoto by reprojecting undistorted frames onto the DSM grid. |
| `geometry.mjs` | Point-cloud maths: PCA, density filtering, rotation to Y-up, subsampling, normals. |
| `ply.mjs` / `glb.mjs` / `png.mjs` | Binary writers/readers for each output format. |
| `hardware.mjs` | Classifies the machine (tier `cpu`/`low`/`mid`/`high`) from a telemetry sample and caps dense resolution, thread count and GPU cache accordingly. |
| `telemetry.mjs` | Best-effort live CPU/RAM/GPU sampling (`nvidia-smi`), cached and failure-backed-off so it can never take the API down. |
| `logger.mjs` | The structured logger: timestamped, levelled terminal output + append-only log files. Every stage boundary, warning and error flows through it. |

---

## 4. How resources are used

The machine is profiled once per run (`hardwareFromTelemetry`) and the profile
caps everything downstream:

- **GPU (RTX 4060, 8 GB).** COLMAP SIFT + matching run on CUDA. OpenMVS
  `DensifyPointCloud` and `TextureMesh` run on CUDA (`--cuda-device -1`).
  Depth-map resolution is capped (`resolution-level`, `max-resolution`) so a
  frame never exceeds VRAM; `RefineMesh` falls back to the unrefined mesh if the
  card runs out of room.
- **CPU (i7-14700HX, 28 threads).** COLMAP bundle adjustment is pinned to the
  machine's `workerThreads` (16 on a 16 GB box — bundle adjustment is
  memory-bound and faster on the P-cores). OpenMVS mesh reconstruction and
  refinement use **all** cores (`--max-threads 0`), which is where "use full
  computational resources" actually pays off.
- **RAM (16 GB).** Caps the COLMAP thread budget and the GPU cache size; OpenMVS
  runs with `--reduce-memory 1` during refinement.

Quality presets (`fast` / `balanced` / `detailed`) scale frame count, feature
count, dense resolution level, mesh point density, texture size, and whether the
expensive photo-refinement stage runs. See `qualitySettings()` and `mvsFor()`.

---

## 5. Progress, logging and lifecycle

- **Progress.** `createProgressTracker` turns per-engine line parsing
  (`parseEngineLine`) into a 0–1 fraction per stage and a weighted overall
  percentage. ETA is pure measurement (elapsed ÷ fraction − elapsed), so it
  self-corrects when a stage runs longer than its reference weight.
- **Logging.** The tracker is handed the per-run logger, so every stage
  transition, completion, skip and failure is one timestamped line on the
  terminal *and* in `runtime/<id>/pipeline.log`. Raw engine output is mirrored
  into the same file (terminal stays readable). The CLI adds an intra-stage
  heartbeat (percentage + ETA).
- **Lifecycle.** `queueExclusive` serialises GPU-heavy runs so two jobs cannot
  thrash the card or corrupt a shared dense workspace. `killTree` tears down the
  whole engine process tree on cancel / timeout / shutdown (a bare `child.kill()`
  leaves COLMAP holding the GPU on Windows).

---

## 6. Serving and viewing

`server.mjs` exposes the console and a small JSON/SSE API on port 8787:

- `POST /api/reconstruct` — upload a video, returns a job id.
- `GET /api/reconstruct/:id/stream` — Server-Sent Events: live progress + telemetry.
- `GET /api/reconstruct/:id/file?kind=glb|model|terrain|cloud|mesh|ortho|meta` —
  stream a product. `kind=glb` resolves to the textured `model.glb` when present,
  otherwise the DSM `terrain.glb`.

`src/viewer.js` (Three.js) loads the GLB, detects whether it carries a texture
map (OpenMVS mesh) or only vertex colours (DSM) and shades it accordingly, then
frames the camera to the model's bounds.

---

## 7. Multi-video and enhancement stages

```text
videos ─► multivideo/  sample · dedup · per-video features · intra + cross matching
              │        connectivity ─► recovery rounds
              ▼
       mapper ─► merge on shared cameras ─► undistort ─► dense ─► mesh ─► texture
              │                                                   (shared code)
              ▼
       products/model.glb ─► analysis ─► confidence ─► repair ─► final attribution ─► LOD
                              (each in a worker thread; failure keeps the model)
```

| Module | Responsibility |
|---|---|
| `multivideo/` | Probe and validate inputs, frame selection and deduplication, VLAD retrieval, pair selection, connectivity, sub-model merging, altitude consistency, separate components. |
| `meshdata.mjs`, `glbmesh.mjs` | One indexed mesh over all primitives with welded topology; GLB geometry read/write that preserves the alignment matrix. |
| `spatial.mjs`, `bvh.mjs` | Hashed point grid and triangle BVH (the latter shared with the browser). |
| `meshrepair/` | Analysis, cleanup, floating components, hole classification and filling, smoothing, appearance, provenance. |
| `confidence/` | Visibility depth buffers, viewing angles, reprojection, point support, depth agreement, scoring, regions, exports. |
| `lod/` | meshoptimizer simplification, atlas re-baking, error measurement, attribute transfer. |
| `gravity.mjs` | Which way is up: camera right axes when the flight turns, roof and ground normals when it holds one heading. `tools/relevel.mjs` applies it to finished jobs. |
| `completion/` | Fills unseen sides after repair: removes grey and edge-on faces, levels the scene, estimates ground and walls, copies photographed facades onto them. Writes `model-completed.glb`; estimates are for viewing, not measuring. |
| `artifacts.mjs` | Content fingerprints so a derived product is never served after its source changed. |
| `enhance/` | Job context in the mesh frame, worker thread and runner. |
| `uploads.mjs`, `tasks.mjs`, `api.mjs` | Staged uploads, background tasks, the added routes. |

Face order is the contract between these: per-face binary sidecars are indexed by
the face order of the matching GLB.

---

## 8. Where to start reading

1. `lib/reconstruct.mjs` → `reconstruct(job)` — the whole flow in one function.
2. `lib/openmvs.mjs` → `buildTexturedMesh` — the four-tool textured-mesh chain.
3. `lib/export3d.mjs` → `buildProducts` — the dense cloud → DSM products.
4. `lib/hardware.mjs` + `qualitySettings`/`mvsFor` — how the knobs are chosen.
