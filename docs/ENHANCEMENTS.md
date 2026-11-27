# Multi-video, mesh repair, levels of detail and confidence

## Multi-video reconstruction

**Use it.** Choose *Multi-video*, add the clips, reorder or trim them, optionally
give each its own altitude or camera model, and press *Reconstruct scene*. From
the command line:

```bash
node reconstruct-cli.mjs front.mp4 rear.mov roof.mkv --balanced --altitude=60 --confidence --lod
```

**Capture advice.** Videos merge only where they see the same surfaces from
viewpoints that features can be matched across (roughly under 30–45° apart). A
rooftop-only nadir pass and a ground-level facade pass may share nothing; add an
oblique orbit that links them.

**What happens.**

1. Each video is sampled separately. Thumbnails give sharpness and frame-to-frame
   motion; frames are spaced evenly in visual change, the sharpest in each slot
   is kept, and near-identical neighbours are dropped only if hash, correlation
   and zero image shift all agree.
2. Features are extracted per video, so each has its own intrinsics.
3. Within a video: sequential window (wider while the camera barely moves),
   power-of-two skips and retrieval-based loop closures.
4. Between videos: a VLAD descriptor built from the job's own local features
   ranks candidate pairs. Every pair is verified by COLMAP's RANSAC two-view
   geometry. Jobs of 50 frames or fewer are matched exhaustively.
5. Connectivity is computed from verified pairs. Disconnected groups get, in
   order: more retrieval candidates, exhaustive search between the groups, then
   the fallback matcher.
6. SfM runs globally (incrementally when groups are known to be separate). Extra
   models are merged only through shared registered images (`model_merger`, a
   Sim(3), then bundle adjustment). Unmergeable groups are reported and, by
   default, reconstructed as separate models in `products/components/`.
7. Dense reconstruction onwards is the normal single-video code.

**Scale.** Per-video altitudes are each converted to a scale; if they disagree by
more than 15% the median is applied, the inconsistency is reported and
measurements are marked approximate. With only the job-wide altitude, videos
flown at clearly different heights are flagged the same way.

**Layout** in `runtime/<job>/`: `inputs/video_00N.*`, `frames/` (flat, named
`video_00N_frame_000123.jpg` so every engine keeps the source identity),
`manifests/{inputs,frames,connectivity,registration}.json`.

## Mesh repair

Open a finished job → **Repair**. No COLMAP or OpenMVS run is needed.

```bash
node reconstruct-cli.mjs --tool=repair --reuse=runtime/<job> --settings="{\"preset\":\"balanced\"}"
```

`products/model.glb` is never modified. Output is `model-repaired.glb` plus
`repair/{provenance.bin, removed.glb, repair-report.json, mesh-quality.json}`.

Steps: invalid-face removal; welding of true duplicate vertices (never across a
UV, colour or material difference); degenerate and duplicate faces; removal of
non-manifold fins; winding consistency; floating components (small **and** either
far from structure or unsupported by dense points); boundary-loop
classification; hole filling (planar ear clipping, minimum-area/bend
triangulation, or refinement with bi-Laplacian fairing); Taubin smoothing with
crease protection and a displacement clamp.

Left open: the rim of the mapped area, gaps above the size limit, and — with
*preserve openings* on — rectangular unsupported gaps in vertical surfaces and
gaps with geometry visible behind them. This classification is a heuristic.
Generated faces are neutral grey unless dense points support the patch and one
camera sees every vertex, in which case vertex colour is sampled from that photo.

## Levels of detail

Open a finished job → **Optimize**, or `--tool=lod`.

Simplification uses meshoptimizer (the build shipped inside three.js). Levels
keep the original UVs where seam-preserving collapse can reach the target; where
a fragmented atlas stops it, the level is unwrapped into new charts and its atlas
is baked from the original model. Each level is accepted only if its measured
95th-percentile surface error is within 0.2 / 0.4 / 1 / 2.5% of the scene
diagonal (× *error tolerance*); otherwise it is retried with more triangles or
skipped, and the manifest says why. Level 0 is the source model itself.

Viewer modes: *Balanced* (default) shows level 0 whenever it fits the memory
budget and steps down only while frames stay below 20 fps. *Performance* picks
the lightest level whose geometric error, or lost texture resolution, projects
below 3 pixels. Neither goes below 150 000 / 40 000 triangles. *Maximum quality*
shows level 0, *Manual* shows what you choose. No level under 30 000 triangles is
generated: on a wide aerial scene it turns buildings into blobs.
Measurement always loads and uses level 0, and says so if it is not yet there.

Compression available: JPEG/PNG/WebP textures and `KHR_mesh_quantization`.
Meshopt, Draco and KTX2 *encoding* are not available (no encoder is bundled).

## Confidence

Open a finished job → **Confidence**, or `--tool=confidence`.

For every face of the original mesh:

| Component | Weight | Definition |
|---|---|---|
| Camera coverage | 0.25 | observers (in image, in front, within 80° of the normal, unoccluded in a per-camera depth buffer); `min(1, Σ min(1, cos/0.5) / 6)` |
| Viewpoint diversity | 0.20 | angular spread of viewing rays over 20°, discounted when all views are oblique |
| Reprojection | 0.20 | median error of nearby sparse points: 1 at ≤0.3 px, 0 at ≥2 px |
| Point support | 0.15 | distance to and density of dense points, relative to the cloud's own spacing |
| Image quality | 0.10 | sharpness of observing frames relative to the job's 90th percentile |
| Depth consistency | 0.10 | agreement with COLMAP depth maps within 2% |

`score = 100 × Σ wᵢ sᵢ / Σ wᵢ` over **available** components; unknown when the
available weight is below 0.5. Depth consistency is unavailable for OpenMVS runs,
which delete their depth maps. OpenMVS per-point visibility is not read.

Repaired meshes inherit scores face by face through the provenance map; smoothed
faces are scaled by 0.95, unsupported ones capped at 25, and generated faces are
unknown. Levels of detail take the mean of four sampled source faces (unknown if
most are unknown) and the worst provenance.

The score is a relative engineering heuristic, **not** a probability or an
accuracy certificate.

## Settings (environment)

| Variable | Default | Effect |
|---|---|---|
| `SKYFORGE_MAX_VIDEOS` | 64 | Videos per job |
| `SKYFORGE_MAX_VIDEO_GB` / `_MAX_TOTAL_GB` | 16 / 64 | Upload limits |
| `SKYFORGE_DISK_RESERVE_GB` | 5 | Free space that must remain |
| `SKYFORGE_UPLOAD_TTL_HOURS` | 24 | Abandoned upload lifetime |
| `SKYFORGE_MAX_FRAMES` | by profile (180 / 300 / 420) | Total multi-video frames |
| `SKYFORGE_WORKER_HEAP_MB` | 40% of RAM, ≤ 8192 | Enhancement worker heap |
| `SKYFORGE_ENHANCE_TIMEOUT_MIN` | 120 | Enhancement time limit |
| `SKYFORGE_RUNTIME` | `./runtime` | Job workspace directory |

## Compatibility

Single-video upload (raw body and headers), existing output files, measurement,
Gaussian splat, turntable, monitoring and cancellation are unchanged. New output
files sit beside the old ones. Without OpenMVS, the COLMAP fallback now meshes the
fused cloud with COLMAP's Delaunay mesher (vertex-coloured); previously such a run
ended without a 3D model unless the 2.5D terrain model was requested.

## Known limitations

- Retrieval is VLAD over local features, not a learned place-recognition network;
  very large viewpoint changes between videos may be missed.
- The hloc path is used for single-video jobs only.
- `node:sqlite` (Node 22.13+) is needed for pre-solve connectivity.
- Self-intersections and surface noise are sampled estimates.
- Hole and opening classification is heuristic; review the *left open* list.
- Smoothing marks every moved face, so a light pass can label most of a noisy
  mesh as smoothed.
- Rebaked LOD atlases are less sharp than the original at the same pixel size.
- Mesh stages hold the whole mesh and up to 4 M cloud points in memory.
