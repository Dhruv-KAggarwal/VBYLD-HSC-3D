# SkyForge — End-to-End Upgrade Plan (best drone-video → 3D)

**Goal:** highest-quality, metrically-accurate, photoreal 3D from a single drone pass — on a laptop where possible, top-class where it counts.

**Your pipeline today:** FFmpeg → COLMAP (SIFT + sequential match + PatchMatch MVS) → custom gravity-aligned DSM → GLB/PLY → Three.js. Auto-tuned to GPU. Good baseline — here's how to make it best-in-class.

---

## 1. The 7-stage pipeline — current vs upgraded

| Stage | You have now | Upgrade to (best) | Why it's better |
| --- | --- | --- | --- |
| **1. Frames** | FFmpeg, 720p sample | FFmpeg + **sharpness/blur keyframe selection** | Drop blurry frames → cleaner matches |
| **2. Masking** | — | **SAM 2 + GroundingDINO** auto-mask cars/people/sky | Removes ghost geometry & dynamic objects |
| **3. Features + Matching** | COLMAP SIFT + sequential | **hloc + LightGlue (SuperPoint)** | Learned matching = far more robust to blur, low texture, repetition |
| **4. Pose / SfM** | COLMAP incremental Mapper | **GLOMAP** (global SfM) | Much faster + more complete reconstructions, same team as COLMAP |
| **5. Georeference** | GPS telemetry fuse | COLMAP **model_aligner** with GPS (+ optional RTK/PPK) | True metric scale & geo-coordinates |
| **6a. Measurable surface** | custom DSM + GLB | **OpenMVS** dense + mesh + **texture refine** | Clean textured mesh for measurement/inspection |
| **6b. Photoreal branch (NEW)** | — | **3D Gaussian Splatting** (Splatfacto) + **2DGS/SuGaR** for mesh | The "wow" photoreal demo + novel views |
| **7. View / Export** | Three.js + GLB/PLY/ortho | Keep Three.js; add **GS web viewer** | Photoreal in-browser |

---

## 2. Tech-stack tiers — pick your lane

### Tier 1 — Best OPEN pipeline (laptop-friendly, your main track) ⭐ recommended

`FFmpeg` + `hloc/LightGlue` + `GLOMAP` + `COLMAP MVS / OpenMVS` + `Three.js`

- All free, offline, GPU-accelerated. This alone is a big jump in quality.

### Tier 2 — Photoreal "wow" branch (for the demo)

Add `Nerfstudio → Splatfacto (3D Gaussian Splatting)` and `2D Gaussian Splatting / SuGaR` for mesh extraction.

- Gives jaw-dropping photoreal renders + smooth novel-view flythroughs. Runs on your RTX 4060.

### Tier 3 — Top-class COMMERCIAL (benchmark / fallback, not for submission code)

`RealityCapture` (now free via Epic for most users) · `Agisoft Metashape` · `Pix4D` · `DJI Terra`

- Use ONE of these to **benchmark** your output against industry quality, and as a safety net for a guaranteed clean result. Don't build your submission on them — build on Tier 1/2.

---

## 3. GitHub repos to use (copy-paste shortlist)

**SfM / matching (core quality win):**

- [COLMAP](https://github.com/colmap/colmap)
- [GLOMAP (global SfM)](https://github.com/colmap/glomap)
- [hloc (hierarchical localization)](https://github.com/cvg/Hierarchical-Localization)
- [LightGlue (learned matcher)](https://github.com/cvg/LightGlue)

**Dense / mesh / texture:**

- [OpenMVS](https://github.com/cdcseacave/openMVS)

**Photoreal / Gaussian Splatting:**

- [Nerfstudio (Splatfacto)](https://github.com/nerfstudio-project/nerfstudio)
- [3D Gaussian Splatting (original)](https://github.com/graphdeco-inria/gaussian-splatting)
- [2D Gaussian Splatting (clean meshes)](https://github.com/hbb1/2d-gaussian-splatting)
- [SuGaR (surface from GS)](https://github.com/Anttwo/SuGaR)

**Preprocessing (masking dynamic objects):**

- [SAM 2](https://github.com/facebookresearch/sam2)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

**Reference pipeline (study their UX / video support):**

- [OpenDroneMap](https://github.com/OpenDroneMap/ODM)

---

## 4. Recommended go-ahead (the decision)

**Do a hybrid, two-branch pipeline:**

1. **Accuracy branch** (your deliverable): LightGlue matching → GLOMAP poses → GPS georeference → OpenMVS textured mesh + DSM. → *measurable, metric, inspection-grade.*
2. **Photoreal branch** (your demo): same poses → Splatfacto (3DGS) → 2DGS/SuGaR mesh. → *the stunning flythrough that wins the room.*

Both share stages 1–5, so you only fork after poses. This gives you **"measurable AND beautiful"** — exactly what beats SfM-only competitors and the LIDAR "colourless cloud."

---

## 5. Next steps — end to end (ordered, do in this sequence)

1. **Capture** a proper Chitkara clip (see the shoot checklist): slow, oblique ~45°, single pass, 70%+ overlap, 4K, low wind, even light.
2. **Preprocess:** extract frames + auto-drop blurry ones; run SAM2+GroundingDINO to mask cars/people.
3. **Match + pose:** integrate hloc+LightGlue for features, run GLOMAP (fallback to COLMAP). Verify all cameras registered.
4. **Georeference:** feed GPS into `model_aligner` → confirm metric scale on a known distance (e.g., a building width you can measure on campus).
5. **Dense deliverable:** run OpenMVS (densify → mesh → refine → texture); export GLB/PLY + orthomosaic + metadata into your existing Three.js viewer.
6. **Photoreal demo:** train Splatfacto on the same poses; extract mesh via 2DGS/SuGaR; record a flythrough for the pitch.
7. **Validate:** measure 2–3 known distances in-viewer vs real tape measurements → report error % (your metric-accuracy proof).
8. **Benchmark (optional):** run the same clip through RealityCapture/Metashape once, screenshot side-by-side to show you're near commercial quality at zero cost.

---

## 6. Quick wins if short on time (biggest quality per hour)

1. **Blur/keyframe filtering** (1 hr) — instantly cleaner results.
2. **LightGlue matching** via hloc (half day) — the single biggest robustness jump.
3. **Dynamic-object masking** with SAM2 (half day) — removes the ugliest artifacts.
4. **Splatfacto demo** (half day) — the visual that wins the pitch.

---

*Pairs with: SkyForge_Pitch_Script.md (how to present) and the shoot checklist (how to capture).*
