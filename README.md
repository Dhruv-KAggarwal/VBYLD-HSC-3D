# Skyforge — drone video to a measurable, inspectable 3D scene

Takes one drone video, or several videos of the same place, and produces a
textured 3D mesh you can measure, a dense point cloud, and optionally a Gaussian
splat you can fly down a street in. It can then repair the mesh, build lighter
levels of detail and show how well each surface is supported by the footage.

Everything runs locally. No cloud, no account, no upload.

---

## What you get

| Output | File | What it is |
|---|---|---|
| Textured 3D mesh | `model.glb` | Delaunay + graph-cut surface with a photographic texture atlas. The headline product. |
| Dense point cloud | `pointcloud.ply` | What was actually measured, with no surface interpolation. |
| Raw mesh | `mesh.ply` | Untextured geometry. |
| Gaussian splat | `splat.ply` | Optional. Photoreal novel views — this is the one for street-level inspection. |
| Terrain model | `terrain.glb`, `terrain.ply` | Optional. A 2.5D heightfield plus orthophoto, for elevation and ortho work. |
| Orthophoto | `orthomosaic.png` | Optional, with the terrain model. |
| Report | `report.json`, `report.md` | Stage timings, measured quality figures, and explicit limitations. |
| Repaired mesh | `model-repaired.glb`, `repair/` | Optional. Cleaned copy with per-face provenance; the original is kept. |
| Levels of detail | `lod/` | Optional. Up to four lighter models with measured error. |
| Confidence | `confidence/` | Optional. Per-face evidence score, regions, heatmap exports. |

Products land in `runtime/<job-id>/products/` and are copied to `output/latest/`.

---

## Requirements

**Hardware.** An NVIDIA GPU is effectively required: COLMAP feature extraction,
OpenMVS depth maps and splat training are all CUDA. The reference machine is an
RTX 4060 Laptop (8 GB), i7-14700HX (20C/28T), 16 GB RAM. 16 GB of system RAM is
the real constraint — the texturing stage peaked at 4.1 GB on that machine and
was paging.

**Software.** Node 22+, and Python 3.13 only if you want the Gaussian splat
branch. The FFmpeg, COLMAP and OpenMVS binaries are not in the repository; see
`engines/README.md` for where they go.

---

## Install

```bash
npm install
```

Then put the engines in place (`engines/README.md` has the download links):

```
engines/
  ffmpeg/bin/ffmpeg.exe
  ffmpeg/bin/ffprobe.exe
  colmap/bin/colmap.exe      # must be a CUDA build
  openmvs/                   # TextureMesh.exe, DensifyPointCloud.exe, ...
```

### Optional: learned features (recommended)

COLMAP 4.2 can run ALIKED detection and LightGlue matching on the GPU, which is
markedly more robust than SIFT on the repetitive facades and large flat roofs a
drone pass is full of. It needs cuDNN 9, which the CUDA Toolkit does not ship
(the toolkit has cuDNN 8):

```bash
npm run setup:cudnn      # ~2 GB into engines/cudnn
```

Without it the pipeline logs a warning and falls back to SIFT. It does not fail.

### Optional: Gaussian splatting

```bash
npm run setup:splat      # ~3 GB: a .venv-splat with torch cu128 + gsplat
```

This uses a prebuilt gsplat wheel so no Visual Studio C++ compiler is needed.
The official gsplat wheels compile CUDA kernels on first import and do require
one.

---

## Run

```bash
npm start            # builds the UI and serves it on http://localhost:8787
```

For UI development with hot reload:

```bash
npm run api          # API on 8787
npm run dev          # Vite on 5173, proxying /api to 8787
```

Then: drop a video, set the flight altitude, pick a profile, press
**Reconstruct scene**.

### Flight altitude matters

Consumer drone video usually carries no GPS. Without it the reconstruction has
no absolute scale, so the altitude you type is what sets metres. Get it wrong
and every measurement is wrong by the same ratio. The report states which
method set the scale.

### Profiles

| Profile | Frame width | Mesh | Use for |
|---|---|---|---|
| Rapid | up to 1280px | up to 250k triangles | Checking a flight is usable |
| Balanced | up to 1920px | up to 400k triangles | Most work |
| Detailed | up to 2560px | up to 1.8M triangles, 8192px texture (6 GB+ GPU) | Final deliverable; sharpest buildings, much slower |

Photo-consistent mesh refinement is off in every profile; set
`SKYFORGE_MESH_REFINE=1` to try it.

Frames are never upscaled past the source resolution. A 1080p clip on
*Detailed* is extracted at 1920px, not 2560px — upscaling adds no detail and
costs quadratically in the dense and texture stages.

---

## Multi-video, repair, levels of detail, confidence

| Feature | In the console | Command line |
|---|---|---|
| Several videos into one model | *Footage → Multi-video* | `node reconstruct-cli.mjs a.mp4 b.mp4 c.mp4` |
| Mesh repair | *After reconstruction*, or the **Repair** tab | `--repair`, or `--tool=repair --reuse=runtime/<job>` |
| Levels of detail | *After reconstruction*, or **Optimize** | `--lod`, or `--tool=lod --reuse=…` |
| Confidence heatmap | *After reconstruction*, or **Confidence** | `--confidence`, or `--tool=confidence --reuse=…` |

How each works, what it can and cannot claim, and every setting:
[`docs/ENHANCEMENTS.md`](docs/ENHANCEMENTS.md). Endpoints and binary formats:
[`docs/API.md`](docs/API.md).

All four run offline with no new dependencies: simplification uses the
meshoptimizer build shipped inside three.js, and `npm run build` copies three's
Draco and Basis decoders into the app so the viewer never calls a CDN.

---

## Using the viewer

Three navigation modes, because orbit alone is useless at ground level:

- **Orbit** — drag to turn, scroll to zoom, right-drag to pan.
- **Fly** — WASD to move, Q/E for down/up, mouse to look, scroll to change
  speed, Esc to exit.
- **Walk** — street level, camera pinned to eye height above the surface.

**Splat view.** *Splat* in the layer panel downloads and shows the Gaussian
splat on the same camera (WebGPU, or its WebGL fallback). It looks close to the
video near the flight path and breaks into streaks from angles the drone never
flew, such as straight down; switch to *Mesh* there, and for measuring. Splats
left at extreme distances by training are hidden on load.

**Drone views.** A reconstruction opens looking through one of the video's own
cameras. The slider steps along the flight, *Overview* frames the filmed area,
and *Filmed area only* hides the thinly observed surroundings out to the horizon
(display only; nothing is deleted).

Display modes: *authored* uses the material as exported (unlit for a baked
atlas, which is correct — the photo already contains the scene's lighting),
*shaded* forces lit PBR so geometric relief is visible, *flat* forces unlit for
comparing colour against the source video.

Measurement: distance, vertical height, and horizontal area. Click points in
the scene; lines and a readout are drawn.

---

## How it works

```
video
  └─ ffmpeg (NVDEC)              extract + sharpen frames, never upscaled
  └─ blur filter                 drop the blurriest frames
  └─ COLMAP feature_extractor    ALIKED on GPU (SIFT fallback)
  └─ COLMAP sequential_matcher   LightGlue + loop closure
  └─ COLMAP global_mapper        global SfM (incremental fallback)
  └─ COLMAP image_undistorter    pinhole dense workspace
  └─ OpenMVS DensifyPointCloud   CUDA depth maps + fusion
  └─ OpenMVS ReconstructMesh     Delaunay tetrahedralisation + graph-cut
  └─ OpenMVS RefineMesh          photo-consistent vertex optimisation (Detailed)
  └─ OpenMVS TextureMesh         per-face texture atlas
  └─ exports                     oriented, metric GLB + cloud + report
  └─ gsplat                      optional Gaussian splat on the same poses
```

The splat branch reuses the COLMAP poses, so enabling it costs only training
time, not a second solve.

---

## Known limitations

These are properties of single-pass capture, not bugs:

- **A single pass sees each surface from few angles.** Facades viewed edge-on
  and surfaces never seen at all are reconstructed from little or no evidence.
  Faces no camera saw are filled with neutral grey, not invented detail.
- **To get real vertical geometry, fly oblique** (camera tilted 30–45° below
  horizontal) and ideally orbit each structure. A nadir or straight-forward
  pass caps how much of a wall any method can recover.
- **No GPS means no absolute georeferencing.** Output is a local tangent plane
  in metres, scaled from your stated altitude.
- **Moving vehicles and people** smear unless masking is enabled, and even then
  may leave artefacts.
- **A Gaussian splat is not a measurement.** Use the mesh and cloud for
  distances; use the splat for looking.

---

## Troubleshooting

**The model renders black.** This was a real bug and is fixed; see
`tools/validate-glb.mjs`, which checks a GLB for every cause we found:

```bash
node tools/validate-glb.mjs output/latest/model.glb
```

**Progress sits at 90% for a long time.** Texturing is the longest stage by far.
The stage weights in `lib/reconstruct.mjs` are calibrated so this should no
longer happen; if it does, re-measure and update `referenceSeconds`.

**"cuDNN 9 not found".** Expected until you run `npm run setup:cudnn`. The
pipeline falls back to SIFT.

**No OpenMVS?** The COLMAP fallback meshes the fused cloud with COLMAP's
Delaunay mesher and colours it per vertex. That step is single-threaded and can
take 15 minutes or more on a cloud of a few million points.

**Reconstruction produced no viewable model.** The dense mesh could not be
built or textured. The log names the stage. This is deliberately a hard error:
previously it silently shipped the 2.5D heightfield instead, which has no
building facades and is what made the output look like a rotating photograph.

---

## Tests

```bash
npm test             # GLB/texture regression tests + pipeline selftests
npm run test:texture # the texture and colour-space tests alone
```

## Environment overrides

| Variable | Effect |
|---|---|
| `SKYFORGE_FEATURES=sift` | Force SIFT instead of ALIKED + LightGlue |
| `SKYFORGE_MAPPER=incremental` | Force the incremental mapper |
| `SKYFORGE_NO_NVDEC=1` | Software video decode |
| `SKYFORGE_MESH_REFINE=0/1` | Force mesh refinement off/on |
| `SKYFORGE_DSM=1` | Always build the 2.5D terrain model |
| `SKYFORGE_SPLAT=1` | Always train a Gaussian splat |
| `SKYFORGE_NO_MVS=1` | Skip OpenMVS, use the COLMAP dense fallback |
| `PORT` | API port (default 8787) |
