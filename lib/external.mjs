/**
 * Detection of OPTIONAL upgrade engines.
 * ---------------------------------------------------------------------------
 * SkyForge's baseline (FFmpeg + COLMAP + OpenMVS, all bundled) always works.
 * The SkyForge_Upgrade_Plan adds learned matching (hloc + LightGlue), global SfM
 * (GLOMAP), dynamic-object/sky masking (SAM 2 + GroundingDINO) and a photoreal
 * Gaussian-Splatting branch (Nerfstudio / Splatfacto). Those are heavier Python
 * / CUDA installs, so every one is OPTIONAL: this module probes for each and the
 * pipeline lights the upgrade up only when it is present, otherwise it falls back
 * to the bundled path. Nothing here throws — a missing tool is just `available:
 * false` with a human-readable reason.
 *
 * Paths can always be forced with env vars (handy on Windows):
 *   GLOMAP_PATH, PYTHON_PATH, HLOC_PY, MASK_PY, SPLAT_PY,
 *   SKYFORGE_NO_HLOC / _NO_GLOMAP / _NO_MASK / _NO_BLURFILTER / _NO_SPLAT = "1"
 */

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const off = (name) => process.env[`SKYFORGE_NO_${name}`] === '1';

/** Resolve an executable: explicit env path, bundled engine, or PATH lookup. */
function resolveExe(envVar, bundled, probeName) {
  const fromEnv = process.env[envVar];
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  for (const candidate of bundled) {
    if (candidate && existsSync(candidate)) return candidate;
  }
  // Last resort: is it on PATH? (spawnSync with --version / -h is cheap.)
  try {
    const probe = spawnSync(probeName, ['--help'], { windowsHide: true, timeout: 8000 });
    if (probe.status === 0 || probe.status === 1) return probeName;
  } catch { /* not on PATH */ }
  return null;
}

/** The Python interpreter to use for the learned-vision tools. */
export function pythonCommand() {
  const forced = process.env.PYTHON_PATH;
  if (forced) return forced;
  for (const name of ['python', 'python3', 'py']) {
    try {
      const probe = spawnSync(name, ['--version'], { windowsHide: true, timeout: 8000 });
      if (probe.status === 0) return name;
    } catch { /* keep trying */ }
  }
  return null;
}

/**
 * Does `import <module>` succeed in the chosen interpreter?
 *
 * Memoised, because this is spawnSync: importing torch takes seconds, and
 * `/api/health` calls the detectors on every request. Un-memoised it blocked
 * the event loop long enough that the health check timed out and the UI
 * reported the whole API as offline.
 */
const moduleProbeCache = new Map();
function pythonHasModule(python, moduleName) {
  if (!python) return false;
  const key = `${python}::${moduleName}`;
  if (moduleProbeCache.has(key)) return moduleProbeCache.get(key);
  let ok = false;
  try {
    const probe = spawnSync(python, ['-c', `import ${moduleName}`], { windowsHide: true, timeout: 20000 });
    ok = probe.status === 0;
  } catch {
    ok = false;
  }
  moduleProbeCache.set(key, ok);
  return ok;
}

/** GLOMAP global-SfM binary (drop-in for `colmap mapper`, same team). */
export function detectGlomap() {
  if (off('GLOMAP')) return { available: false, reason: 'disabled via SKYFORGE_NO_GLOMAP' };
  const exe = resolveExe('GLOMAP_PATH', [resolve('engines/glomap/glomap.exe'), resolve('engines/glomap/bin/glomap.exe')], 'glomap');
  return exe ? { available: true, exe } : { available: false, reason: 'glomap binary not found (set GLOMAP_PATH)' };
}

/** hloc + LightGlue learned matching (Python). Needs our script + the modules. */
export function detectHloc(scriptPath) {
  if (off('HLOC')) return { available: false, reason: 'disabled via SKYFORGE_NO_HLOC' };
  const python = pythonCommand();
  if (!python) return { available: false, reason: 'no python interpreter found (set PYTHON_PATH)' };
  const script = process.env.HLOC_PY || scriptPath;
  if (!script || !existsSync(script)) return { available: false, reason: 'hloc script missing' };
  if (!pythonHasModule(python, 'hloc')) return { available: false, reason: 'python module "hloc" not installed' };
  return { available: true, python, script };
}

/** SAM 2 / GroundingDINO masking (Python). Falls back to a heuristic sky mask. */
export function detectMasking(scriptPath) {
  if (off('MASK')) return { available: false, reason: 'disabled via SKYFORGE_NO_MASK' };
  const python = pythonCommand();
  if (!python) return { available: false, reason: 'no python interpreter found' };
  const script = process.env.MASK_PY || scriptPath;
  if (!script || !existsSync(script)) return { available: false, reason: 'mask script missing' };
  // The script self-selects SAM2+GroundingDINO when importable, else a fast
  // colour/position sky heuristic (only needs numpy + opencv or PIL), so we only
  // require a python with at least one of those.
  const heuristic = pythonHasModule(python, 'cv2') || pythonHasModule(python, 'PIL');
  const learned = pythonHasModule(python, 'sam2');
  if (!heuristic && !learned) return { available: false, reason: 'need python cv2/PIL (heuristic) or sam2 (learned)' };
  return { available: true, python, script, mode: learned ? 'sam2' : 'heuristic' };
}

/**
 * Gaussian-splat trainer (gsplat).
 *
 * Prefers the project-local `.venv-splat`, which `npm run setup:splat`
 * creates. That venv is deliberately separate from whatever Python the machine
 * uses for masking and hloc: the splat branch pins torch and gsplat to matching
 * CUDA builds, and sharing an interpreter with other tools is how those pins
 * get broken.
 *
 * Probes for `gsplat` rather than `nerfstudio`. The trainer used to shell out
 * to Nerfstudio's CLI; it now drives gsplat directly, so Nerfstudio is no
 * longer a dependency.
 */
export function detectSplat(scriptPath) {
  if (off('SPLAT')) return { available: false, reason: 'disabled via SKYFORGE_NO_SPLAT' };
  const script = process.env.SPLAT_PY || scriptPath || resolve('tools/python/train_splat.py');
  if (!script || !existsSync(script)) return { available: false, reason: 'splat trainer script missing' };

  const venvPython = resolve('.venv-splat/Scripts/python.exe');
  const venvPosix = resolve('.venv-splat/bin/python');
  const python = process.env.SPLAT_PYTHON
    || (existsSync(venvPython) ? venvPython : existsSync(venvPosix) ? venvPosix : pythonCommand());
  if (!python) return { available: false, reason: 'no python interpreter found' };

  if (!pythonHasModule(python, 'gsplat')) {
    return { available: false, reason: 'python module "gsplat" not installed (run: npm run setup:splat)' };
  }
  if (!pythonHasModule(python, 'torch')) {
    return { available: false, reason: 'python module "torch" not installed' };
  }
  return { available: true, python, script };
}

/** One-shot summary for the startup log / report. */
export function detectAll({ hlocScript, maskScript, splatScript } = {}) {
  return {
    glomap: detectGlomap(),
    hloc: detectHloc(hlocScript),
    masking: detectMasking(maskScript),
    splat: detectSplat(splatScript),
    blurFilter: off('BLURFILTER') ? { available: false, reason: 'disabled' } : { available: true },
  };
}
