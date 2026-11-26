# SkyForge optional AI upgrades (Python side)

The Node pipeline (FFmpeg + COLMAP + OpenMVS) runs with **zero** Python. These
scripts add the higher-quality engines from `SkyForge_Upgrade_Plan.md`. Each is
**optional** — the pipeline auto-detects what's installed (`lib/external.mjs`)
and falls back to the bundled path if a tool is missing. Install only what you
want.

All are driven automatically during `node reconstruct-cli.mjs ...`; you normally
don't call them by hand. Point the pipeline at a specific Python with the
`PYTHON_PATH` env var if you use a venv/conda env.

| Upgrade | What it does | Script | Turn off with |
|---|---|---|---|
| Blur filter | Drops blurry frames (Node, always on) | `lib/preprocess.mjs` | `SKYFORGE_NO_BLURFILTER=1` |
| Sky/dynamic masking | Removes sky + cars/people at the source | `mask_frames.py` | `SKYFORGE_NO_MASK=1` |
| hloc + LightGlue | Learned features + matching (robust) | `hloc_match.py` | `SKYFORGE_NO_HLOC=1` |
| GLOMAP | Global SfM (faster, more complete poses) | *(binary)* | `SKYFORGE_NO_GLOMAP=1` |
| Splatfacto (3DGS) | Photoreal Gaussian-splat branch | `train_splat.py` | `SKYFORGE_NO_SPLAT=1` |

---

## 1. Sky / dynamic masking (lightest — do this first)

Removes the white sky/cloud "ceiling" and moving vehicles so they never become
ghost geometry. The default back-end is a fast colour+connectivity sky segmenter
needing only OpenCV/numpy (no GPU):

```bash
pip install -r tools/python/requirements-mask.txt
```

That's it — the next reconstruction auto-detects it and masks the sky.
(The learned SAM 2 + GroundingDINO back-end is scaffolded in `mask_frames.py`;
wire in your model weights there if you want prompt-driven object masking.)

## 2. hloc + LightGlue (biggest robustness jump)

Learned SuperPoint features + LightGlue matching → a COLMAP database. Needs a
CUDA PyTorch.

```bash
# in a venv/conda env with CUDA torch already installed:
pip install git+https://github.com/cvg/Hierarchical-Localization.git
pip install pycolmap
```

Verify: `python -c "import hloc, pycolmap"` should not error. The pipeline then
prints `Features + matching: hloc + LightGlue`.

## 3. GLOMAP (global SfM)

Download a Windows build from https://github.com/colmap/glomap (releases) and
either drop `glomap.exe` in `engines/glomap/` or set `GLOMAP_PATH=C:\path\glomap.exe`.
Runs after features+matching; needs the COLMAP database to already have matches
(COLMAP SIFT or hloc both populate it).

## 4. Splatfacto — photoreal Gaussian Splatting (the "wow" demo)

```bash
pip install nerfstudio            # pulls torch/CUDA deps; see their install guide
```

Then, after a normal reconstruction, train the splat on the SAME poses:

```bash
python tools/python/train_splat.py \
  --work "runtime/<job-id>" \
  --out  "runtime/<job-id>/products/splat" \
  --iters 15000
```

Output: a Gaussian-splat `.ply` in `products/splat/`. View it with any web GS
viewer (e.g. antimatter15/splat, or a three.js `.splat` loader) for photoreal
novel views. This branch is separate from the measurable mesh — it reuses the
poses, so no re-solve.

---

### Recommended order (quality per hour)
1. **Masking** (5 min install) — kills the sky artefacts.
2. **hloc + LightGlue** (½ day) — the single biggest completeness/robustness win.
3. **GLOMAP** (drop-in binary) — faster, more complete poses.
4. **Splatfacto** (½ day) — the photoreal pitch demo.

Blur filtering is already on with no install.
