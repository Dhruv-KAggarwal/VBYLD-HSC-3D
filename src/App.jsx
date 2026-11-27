import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ModelViewer } from './viewer.js';
import {
  cancelJob, fileUrl, fileUrlFresh, getArtifacts, getHealth, getJobFiles, getLodManifest, getViews, listJobs, lodUrl,
  startMultiReconstruction, startReconstruction, watchJob,
} from './api.js';
import { TurntablePlayer } from './TurntablePlayer.jsx';
import { MultiVideoUpload } from './components/MultiVideoUpload.jsx';
import { DRAWER_TABS, EnhancementDrawer } from './components/EnhancementDrawer.jsx';
import { REPAIR_DEFAULTS, RepairSettings } from './components/MeshRepairPanel.jsx';
import { DroneViews } from './components/DroneViews.jsx';
import { Collapsible, Segmented, Toggle, formatBytes, formatDuration } from './components/ui.jsx';

const QUALITY = [
  { id: 'Rapid', title: 'Rapid', note: 'Fastest. 1280px frames, no mesh refinement.' },
  { id: 'Balanced', title: 'Balanced', note: '1920px frames, full texture atlas.' },
  { id: 'Detailed', title: 'Detailed', note: 'Most building detail: up to 1.8M triangles and an 8192px texture. Much slower.' },
];

const NAV_MODES = [
  { id: 'orbit', label: 'Orbit', hint: 'Drag to turn, scroll to zoom, right-drag to pan' },
  { id: 'fly', label: 'Fly', hint: 'WASD to move, Q/E down-up, mouse to look, Esc to exit' },
  { id: 'walk', label: 'Walk', hint: 'Street level. WASD to walk, mouse to look, Esc to exit' },
];

const MEASURE_MODES = [
  { id: 'off', label: 'Off' },
  { id: 'distance', label: 'Distance' },
  { id: 'height', label: 'Height' },
  { id: 'area', label: 'Area' },
];

function clockOf(at) {
  if (!at) return '--:--:--';
  const d = new Date(at);
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, '0')).join(':');
}

const DOWNLOADS = [
  ['glb', 'Textured mesh (GLB)'],
  ['repaired', 'Repaired mesh (GLB)'],
  ['video', 'Turntable video (MP4)'],
  ['cloud', 'Point cloud (PLY)'],
  ['mesh', 'Raw mesh (PLY)'],
  ['splat', 'Gaussian splat (PLY)'],
  ['terrain', 'Terrain DSM (GLB)'],
  ['ortho', 'Orthophoto (PNG)'],
  ['meta', 'Metadata (JSON)'],
  ['report', 'Report (Markdown)'],
  ['confidence-glb', 'Confidence heatmap (GLB)'],
  ['confidence-ply', 'Confidence (PLY)'],
  ['confidence-report', 'Confidence report'],
  ['lod-report', 'LOD report'],
];

export default function App() {
  const canvasRef = useRef(null);
  const viewerRef = useRef(null);
  const unwatchRef = useRef(null);
  const logRef = useRef(null);

  const [health, setHealth] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [file, setFile] = useState(null);
  const [inputMode, setInputMode] = useState('single');
  const [session, setSession] = useState(null);
  const [pendingUploads, setPendingUploads] = useState(0);
  const [drawer, setDrawer] = useState(null);
  const [lod, setLod] = useState(null);
  const [modelVersion, setModelVersion] = useState(0);
  const [viewer, setViewer] = useState(null);
  const [downloads, setDownloads] = useState(null);
  const [views, setViews] = useState(null);
  const loadSequence = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  // Which view the scene area shows. The turntable is a flat MP4 and needs no
  // GPU, so it is the fallback when WebGL is unavailable as well as the
  // quickest way to judge a finished model.
  const [sceneView, setSceneView] = useState('viewer');
  const [log, setLog] = useState([]);
  const [loadPct, setLoadPct] = useState(null);
  const [sceneInfo, setSceneInfo] = useState(null);
  const [measureText, setMeasureText] = useState('');
  const [navHint, setNavHint] = useState(NAV_MODES[0].hint);
  // Raw engine lines are hidden by default: COLMAP and OpenMVS emit hundreds
  // per second and bury the narration. The toggle shows everything.
  const [showEngineLines, setShowEngineLines] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);

  const [options, setOptions] = useState({
    quality: 'Balanced',
    altitude: 82,
    frames: '',
    threads: '',
    splat: false,
    dsm: false,
    maskDynamic: false,
    matching: 'auto',
    separateComponents: true,
    // Optional stages. Repair is opt-in; the original model is kept either way.
    confidence: true,
    lod: true,
    repair: false,
    repairSettings: { ...REPAIR_DEFAULTS },
  });

  const [display, setDisplay] = useState({
    nav: 'orbit',
    measure: 'off',
    shading: 'authored',
    layers: { mesh: true, splat: true, points: false, grid: true },
  });

  const pushLog = useCallback((text, kind = 'info') => {
    setLog((prev) => [...prev.slice(-400), { text, kind, at: Date.now() }]);
  }, []);

  // --- Viewer bootstrap ----------------------------------------------------
  useEffect(() => {
    if (!canvasRef.current || viewerRef.current) return;
    const viewer = new ModelViewer(canvasRef.current);
    viewerRef.current = viewer;
    setViewer(viewer);
    viewer.onLodState = setLod;
    viewer.onModelShown = () => setModelVersion((v) => v + 1);
    viewer.onProgress = (pct, loaded, total) => {
      setLoadPct({ pct: Math.round(pct * 100), loaded, total });
    };
    viewer.onMeasure = ({ message }) => setMeasureText(message || '');
    viewer.onNavChange = ({ mode, speed }) => {
      const found = NAV_MODES.find((m) => m.id === mode);
      setNavHint(`${found?.hint || ''} · speed ${speed.toFixed(2)}`);
    };
    return () => { viewer.clearModel(); };
  }, []);

  // Keep the canvas sized to its container, not just the window.
  useEffect(() => {
    if (!canvasRef.current) return undefined;
    const observer = new ResizeObserver(() => viewerRef.current?.resize());
    observer.observe(canvasRef.current.parentElement);
    return () => observer.disconnect();
  }, []);

  useEffect(() => { viewerRef.current?.setLayers(display.layers); }, [display.layers]);
  useEffect(() => { viewerRef.current?.setShading(display.shading); }, [display.shading]);
  useEffect(() => { viewerRef.current?.setMeasureMode(display.measure); }, [display.measure]);

  const visibleLog = useMemo(
    () => (showEngineLines ? log : log.filter((entry) => entry.kind !== 'engine')),
    [log, showEngineLines],
  );

  useEffect(() => {
    if (autoScroll && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [visibleLog, autoScroll]);

  // --- Server state --------------------------------------------------------
  const refreshJobs = useCallback(async () => {
    try { setJobs(await listJobs()); } catch { /* the library is not critical */ }
  }, []);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ offline: true }));
    refreshJobs();
  }, [refreshJobs]);

  // --- Loading results into the viewer -------------------------------------
  const loadResult = useCallback(async (target, { keepCamera = false } = {}) => {
    const viewer = viewerRef.current;
    if (!viewer || !target?.id) return;
    const sequence = ++loadSequence.current;
    const current = () => sequence === loadSequence.current;
    const camera = keepCamera ? viewer.cameraState() : null;
    setLoadPct({ pct: 0, loaded: 0, total: 0 });
    try {
      viewer.clearModel({ keepMeasure: keepCamera });
      if (!keepCamera) { viewer.setViews(null); setViews(null); }
      viewer.scaleReliable = target.metrics
        ? (target.metrics.scaleReliable ?? target.metrics.scaleSource === 'flight-altitude')
        : true;
      // Older jobs, and jobs still running, may have no derived products at all.
      const artifacts = await getArtifacts(target.id).catch(() => null);
      let hasTexture = false;
      let hasVertexColor = false;
      let levels = false;
      if (artifacts?.artifacts.lod.current) {
        try {
          const manifest = await getLodManifest(target.id);
          if (!current()) return;
          await viewer.openLod(manifest, (level) => lodUrl(target.id, level.level, manifest.generatedAt));
          // A newer load owns the viewer now; leave its session alone.
          if (!current()) return;
          levels = viewer.lod?.current != null;
          hasTexture = manifest.levels[0].textureCount > 0;
          hasVertexColor = !hasTexture;
          if (levels) pushLog(`Loaded progressively: ${manifest.levels.length} levels of detail`, 'ok');
          else viewer.closeLod();
        } catch (lodError) {
          if (!current()) return;
          viewer.closeLod();
          pushLog(`Levels of detail unavailable (${lodError.message}); loading the full model`, 'warn');
        }
      }
      if (!levels) {
        const url = artifacts?.activeModel === 'repaired'
          ? fileUrlFresh(target.id, 'repaired', artifacts.artifacts.repair.createdAt)
          : fileUrlFresh(target.id, 'glb', artifacts?.artifacts.repair.createdAt || target.id);
        const loaded = await viewer.loadModelData(url, { quiet: false });
        if (!current()) { viewer.disposeObject(loaded.model); return; }
        viewer.showModel(loaded.model, { fit: true });
        ({ hasTexture, hasVertexColor } = loaded);
        pushLog(hasTexture
          ? 'Loaded textured 3D mesh (photographic atlas)'
          : 'Loaded vertex-coloured 3D mesh (no texture atlas in this build)', 'ok');
      }
      if (!current()) return;
      if (camera) viewer.restoreCamera(camera);
      // Open from where the drone actually looked, on the area it filmed.
      const found = await getViews(target.id).catch(() => null);
      if (!current()) return;
      setViews(found ? { id: target.id, ...found } : null);
      viewer.setViews(found, { jump: !keepCamera });
      if (found && !keepCamera) viewer.setFocusClip(true);
      const info = {
        textured: hasTexture,
        vertexColoured: hasVertexColor && !hasTexture,
        splat: false,
        cloud: false,
      };

      // The splat is optional and may not exist for this job.
      try {
        const splat = await viewer.loadSplat(fileUrl(target.id, 'splat'));
        info.splat = true;
        pushLog(`Loaded Gaussian splat (${splat.count.toLocaleString()} splats)`, 'ok');
      } catch { /* no splat for this job */ }

      if (current()) setSceneInfo(info);
    } catch (loadError) {
      if (!current()) return;
      pushLog(`Could not load the model: ${loadError.message}`, 'error');
      setError(`Could not load the model: ${loadError.message}`);
    } finally {
      if (current()) setLoadPct(null);
    }
  }, [pushLog]);

  const loadCloud = useCallback(async () => {
    const viewer = viewerRef.current;
    const id = job?.id;
    if (!viewer || !id) return;
    try {
      setLoadPct({ pct: 0, loaded: 0, total: 0 });
      const { count } = await viewer.loadPointCloud(fileUrl(id, 'cloud'));
      pushLog(`Loaded dense point cloud (${count.toLocaleString()} points)`, 'ok');
      setSceneInfo((prev) => ({ ...(prev || {}), cloud: true }));
      setDisplay((prev) => ({ ...prev, layers: { ...prev.layers, points: true } }));
    } catch (cloudError) {
      pushLog(`Could not load the point cloud: ${cloudError.message}`, 'error');
    } finally {
      setLoadPct(null);
    }
  }, [job?.id, pushLog]);

  // --- Running a job -------------------------------------------------------
  const enabledVideos = session?.videos?.filter((v) => v.settings.enabled).length || 0;
  const multi = inputMode === 'multi';
  const canStart = multi ? enabledVideos > 0 && pendingUploads === 0 : Boolean(file);

  const start = useCallback(async () => {
    if (!canStart || busy) return;
    setBusy(true);
    setError(null);
    setLog([]);
    setSceneInfo(null);
    pushLog(multi ? `Starting a reconstruction from ${enabledVideos} videos` : `Uploading ${file.name} (${formatBytes(file.size)})`);
    try {
      const enhance = {
        analysis: options.repair || options.confidence || options.lod,
        confidence: options.confidence,
        lod: options.lod,
        repair: options.repair,
        repairSettings: options.repairSettings,
      };
      const { id } = multi
        ? await startMultiReconstruction(session.id, { ...options, enhance })
        : await startReconstruction(file, { ...options, enhance });
      if (multi) setSession(null);
      loadedForRef.current = null;
      pushLog(`Job ${id} accepted; reconstruction queued`, 'ok');
      unwatchRef.current?.();
      unwatchRef.current = watchJob(id, {
        onUpdate: (next) => setJob(next),
        // Every line the terminal prints, as the engines print it.
        onLog: (lines) => setLog((prev) => {
          const merged = prev.concat(lines.map((line) => ({
            text: line.message,
            kind: line.level === 'error' ? 'error'
              : line.level === 'warn' ? 'warn'
                : line.level === 'ok' ? 'ok'
                  : line.level === 'engine' ? 'engine' : 'info',
            at: line.at,
            seq: line.seq,
          })));
          // Bounded: a long run emits tens of thousands of lines and React
          // re-renders the whole list.
          return merged.length > 1200 ? merged.slice(merged.length - 1200) : merged;
        }),
        onDone: async (next) => {
          setJob(next);
          setBusy(false);
          pushLog('Reconstruction complete', 'ok');
          if (next.warning) pushLog(next.warning, 'warn');
          await refreshJobs();
          // Only load here if the early-release effect has not already done it.
          // Repair and LOD finish after the early release, so load again for them.
          const derived = await getArtifacts(next.id).catch(() => null);
          const changed = derived?.artifacts.lod.current || derived?.activeModel === 'repaired';
          if (loadedForRef.current !== next.id || changed) {
            const again = loadedForRef.current === next.id;
            loadedForRef.current = next.id;
            await loadResult(next, { keepCamera: again });
          }
        },
        onError: (next) => {
          setJob(next);
          setBusy(false);
          const message = next.error || 'Reconstruction failed';
          setError(message);
          pushLog(message, 'error');
        },
      });
    } catch (startError) {
      setBusy(false);
      setError(startError.message);
      pushLog(startError.message, 'error');
    }
  }, [canStart, multi, enabledVideos, session, file, busy, options, pushLog, refreshJobs, loadResult]);

  const stop = useCallback(async () => {
    if (!job?.id) return;
    try {
      await cancelJob(job.id);
      pushLog('Cancellation requested; stopping the engine', 'warn');
    } catch (cancelError) {
      pushLog(`Could not cancel: ${cancelError.message}`, 'error');
    }
  }, [job?.id, pushLog]);

  useEffect(() => () => unwatchRef.current?.(), []);

  // Load the model the moment the pipeline says it exists, rather than waiting
  // for the whole job. Guarded by a ref so a re-render mid-run cannot restart
  // a multi-megabyte download.
  const loadedForRef = useRef(null);
  const modelAvailable = Boolean(job && (job.modelReady || job.status === 'complete' || job.result?.model));

  useEffect(() => {
    if (!modelAvailable || !job?.id) return;
    if (loadedForRef.current === job.id) return;
    loadedForRef.current = job.id;
    loadResult(job);
  }, [modelAvailable, job?.id, loadResult, job]);

  // Ctrl+Enter starts the run, matching the old console.
  useEffect(() => {
    const onKey = (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); start(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [start]);

  // Offer only the outputs that exist on disk.
  useEffect(() => {
    let stale = false;
    if (!modelAvailable || !job?.id) { setDownloads(null); return undefined; }
    getJobFiles(job.id).then((r) => { if (!stale) setDownloads(r.files); }).catch(() => { if (!stale) setDownloads(null); });
    return () => { stale = true; };
  }, [job?.id, job?.status, modelAvailable, modelVersion]);

  const reloadCurrent = useCallback(() => {
    if (job?.id) loadResult(job, { keepCamera: true });
  }, [job, loadResult]);

  const stages = job?.stages || [];
  // job.progress and stage.progress are ALREADY percentages (the tracker emits
  // Math.round(fraction * 1000) / 10). Multiplying by 100 again is what put
  // "900%" on the bar.
  const progress = Math.max(0, Math.min(100, Number(job?.progress) || 0));
  const telemetry = job?.telemetry || health?.telemetry;

  // Stall detection. The bar cannot tell "slow stage" from "dead process", and
  // some stages legitimately run minutes without printing (global bundle
  // adjustment, atlas packing). Warn at 2 minutes of silence, escalate at 10.
  // A model is downloadable as soon as it is on disk, which is BEFORE the job
  // as a whole finishes: the optional splat stage runs afterwards and can take
  // longer than everything preceding it. Gating the downloads on
  // status === 'complete' is what made a finished mesh look like no mesh.
  const idleMs = job?.idleMs ?? null;
  const stall = !busy || idleMs == null ? null
    : idleMs > 600000 ? { level: 'bad', text: `No engine output for ${formatDuration(idleMs / 1000)}. The stage may be stuck — check the log below, or cancel.` }
      : idleMs > 120000 ? { level: 'warn', text: `Quiet for ${formatDuration(idleMs / 1000)}. Normal during bundle adjustment and atlas packing.` }
        : null;

  const onPickFile = (picked) => {
    if (!picked) return;
    setFile(picked);
    setError(null);
    pushLog(`Selected ${picked.name} (${formatBytes(picked.size)})`);
  };

  const engineLine = useMemo(() => {
    if (!health) return 'Checking engines…';
    if (health.offline) return 'API offline — run `npm start`';
    const bits = [];
    bits.push(health.ffmpeg ? 'FFmpeg ✓' : 'FFmpeg ✗');
    bits.push(health.colmap ? 'COLMAP ✓' : 'COLMAP ✗');
    bits.push(health.openmvs ? 'OpenMVS ✓' : 'OpenMVS ✗');
    if (health.cudnn !== undefined) bits.push(health.cudnn ? 'cuDNN ✓' : 'cuDNN ✗');
    if (health.splat !== undefined) bits.push(health.splat ? 'gsplat ✓' : 'gsplat ✗');
    return bits.join('  ');
  }, [health]);

  return (
    <div className="h-full grid grid-cols-[380px_minmax(0,1fr)] grid-rows-1 max-md:grid-cols-1 max-md:grid-rows-none max-md:overflow-y-auto bg-ink-950">
      {/* ------------------------------------------------- control column */}
      <aside className="h-full min-h-0 max-md:h-auto overflow-y-auto border-r border-ink-700 bg-ink-900 flex flex-col">
        <header className="px-5 py-4 border-b border-ink-700 flex items-center gap-3 sticky top-0 bg-ink-900 z-10">
          <div className="w-9 h-9 rounded-lg bg-jet-500/15 border border-jet-500/40 grid place-items-center text-jet-400 text-lg">◈</div>
          <div className="min-w-0">
            <div className="font-semibold tracking-wide text-mist-200">SKYFORGE</div>
            <div className="kicker truncate">Drone video → 3D</div>
          </div>
        </header>

        <div className="p-5 space-y-5 flex-1">
          {/* ---- 1. footage */}
          <section>
            <p className="kicker mb-2">01 · Footage</p>
            <div className="mb-2">
              <Segmented
                label="Input mode" disabled={busy} value={inputMode} onChange={setInputMode}
                options={[['single', 'Single video'], ['multi', 'Multi-video']]}
              />
            </div>
            {multi && (
              <MultiVideoUpload
                session={session} onSession={setSession} onPending={setPendingUploads} disabled={busy}
                quality={{ Rapid: 'fast', Balanced: 'balanced', Detailed: 'detailed' }[options.quality]}
                frames={options.frames}
              />
            )}
            {!multi && (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); onPickFile(e.dataTransfer.files?.[0]); }}
              className={`panel p-5 text-center transition-colors ${dragging ? 'border-jet-500 bg-jet-500/5' : ''}`}
            >
              {file ? (
                <div className="flex items-center gap-3 text-left">
                  <div className="w-9 h-9 rounded bg-ink-700 grid place-items-center">▣</div>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm truncate">{file.name}</div>
                    <div className="kicker">{formatBytes(file.size)}</div>
                  </div>
                  <button className="btn px-2 py-1" onClick={() => setFile(null)} disabled={busy}>×</button>
                </div>
              ) : (
                <>
                  <div className="text-2xl text-mist-400 mb-1">↑</div>
                  <div className="text-sm mb-1">Drop a drone video</div>
                  <div className="kicker mb-3">MP4 · MOV · streamed straight to disk</div>
                  <label className="btn cursor-pointer">
                    Browse
                    <input
                      type="file"
                      accept="video/*"
                      className="hidden"
                      onChange={(e) => onPickFile(e.target.files?.[0])}
                    />
                  </label>
                </>
              )}
            </div>
            )}
          </section>

          {/* ---- 2. profile */}
          <section>
            <p className="kicker mb-2">02 · Reconstruction profile</p>
            <div className="space-y-1.5">
              {QUALITY.map((q) => (
                <button
                  key={q.id}
                  disabled={busy}
                  onClick={() => setOptions((o) => ({ ...o, quality: q.id }))}
                  className={`panel w-full text-left px-3 py-2.5 ${options.quality === q.id ? 'border-jet-500 bg-jet-500/10' : 'hover:border-ink-600'}`}
                >
                  <div className="text-sm font-medium">{q.title}</div>
                  <div className="kicker normal-case tracking-normal mt-0.5">{q.note}</div>
                </button>
              ))}
            </div>
          </section>

          {/* ---- 3. parameters */}
          <section className="space-y-3">
            <p className="kicker">03 · Parameters</p>

            <label className="block">
              <span className="kicker block mb-1">Flight altitude above ground (m){multi ? ' — default for all videos' : ''}</span>
              <input
                type="number" min="1" max="2000" className="field" disabled={busy}
                value={options.altitude}
                onChange={(e) => setOptions((o) => ({ ...o, altitude: Number(e.target.value) }))}
              />
              <span className="kicker normal-case tracking-normal block mt-1 text-mist-400">
                No GPS in the video, so this sets metric scale. Wrong altitude = wrong measurements.
              </span>
            </label>

            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="kicker block mb-1">{multi ? 'Total frames' : 'Frames'}</span>
                <input
                  type="number" min={multi ? 24 : 12} max={multi ? 2000 : 240} placeholder="Auto" className="field" disabled={busy}
                  value={options.frames}
                  onChange={(e) => setOptions((o) => ({ ...o, frames: e.target.value }))}
                />
              </label>
              <label className="block">
                <span className="kicker block mb-1">Threads</span>
                <input
                  type="number" min="1" max="64" placeholder="Auto" className="field" disabled={busy}
                  value={options.threads}
                  onChange={(e) => setOptions((o) => ({ ...o, threads: e.target.value }))}
                />
              </label>
            </div>

            <div className="space-y-2 pt-1">
              <Toggle
                checked={options.splat}
                disabled={busy || health?.splat === false}
                onChange={(v) => setOptions((o) => ({ ...o, splat: v }))}
                label="Photoreal Gaussian splat"
                note={health?.splat === false
                  ? 'Trainer not installed — run npm run setup:splat'
                  : 'Adds a training stage. This is the view you can fly down a street in.'}
              />
              <Toggle
                checked={options.dsm}
                disabled={busy}
                onChange={(v) => setOptions((o) => ({ ...o, dsm: v }))}
                label="2.5D terrain model (DSM + orthophoto)"
                note="Heightfield for elevation and ortho work. It has no building facades."
              />
              <Toggle
                checked={options.maskDynamic}
                disabled={busy}
                onChange={(v) => setOptions((o) => ({ ...o, maskDynamic: v }))}
                label="Mask moving vehicles and people"
                note="Stops traffic smearing into the geometry. Slower."
              />
              {multi && (
                <>
                  <Toggle
                    checked={options.matching === 'exhaustive'}
                    disabled={busy}
                    onChange={(v) => setOptions((o) => ({ ...o, matching: v ? 'exhaustive' : 'auto' }))}
                    label="Exhaustive matching"
                    note="Compares every frame with every other. Much slower; off uses retrieval-ranked pairs."
                  />
                  <Toggle
                    checked={options.separateComponents}
                    disabled={busy}
                    onChange={(v) => setOptions((o) => ({ ...o, separateComponents: v }))}
                    label="Model unconnected groups separately"
                    note="Videos that cannot be merged get their own model instead of being forced in."
                  />
                </>
              )}
            </div>
          </section>

          {/* ---- 4. enhancements */}
          <section className="space-y-2">
            <p className="kicker">04 · After reconstruction</p>
            <Toggle
              checked={options.confidence} disabled={busy}
              onChange={(v) => setOptions((o) => ({ ...o, confidence: v }))}
              label="Confidence analysis" note="Heatmap of how well each surface is supported by the footage."
            />
            <Toggle
              checked={options.lod} disabled={busy}
              onChange={(v) => setOptions((o) => ({ ...o, lod: v }))}
              label="Levels of detail" note="Lighter models for faster loading and navigation."
            />
            <Toggle
              checked={options.repair} disabled={busy}
              onChange={(v) => setOptions((o) => ({ ...o, repair: v }))}
              label="Automatic mesh repair" note="Writes a repaired copy; the original is always kept."
            />
            {options.repair && (
              <Collapsible title="Repair settings">
                <RepairSettings value={options.repairSettings} disabled={busy} onChange={(repairSettings) => setOptions((o) => ({ ...o, repairSettings }))} />
              </Collapsible>
            )}
          </section>

          {/* ---- run */}
          <section className="space-y-2">
            {busy ? (
              <button className="btn btn-danger w-full" onClick={stop}>Cancel reconstruction</button>
            ) : (
              <button className="btn btn-primary w-full" onClick={start} disabled={!canStart}>
                ◈ Reconstruct scene
                <kbd className="kicker opacity-70">Ctrl ↵</kbd>
              </button>
            )}
            {error && (
              <div className="panel border-red-900/60 bg-red-950/30 p-3 text-xs text-red-200">{error}</div>
            )}
          </section>

          {/* ---- library */}
          {jobs.length > 0 && (
            <section>
              <p className="kicker mb-2">Previous reconstructions</p>
              <div className="space-y-1.5 max-h-56 overflow-y-auto">
                {jobs.slice(0, 12).map((entry) => (
                  <button
                    key={entry.id}
                    onClick={() => { loadedForRef.current = entry.id; setJob(entry); loadResult(entry); }}
                    className={`panel w-full text-left px-3 py-2 hover:border-ink-600 ${job?.id === entry.id ? 'border-jet-500/60' : ''}`}
                  >
                    <div className="text-xs truncate">{entry.fileName || entry.id}</div>
                    <div className="kicker">
                      {entry.mode === 'multi' ? `${entry.videoCount || 'multi'} videos · ` : ''}{entry.metrics?.profile || '—'} · {entry.metrics?.meshFaces
                        ? `${Number(entry.metrics.meshFaces).toLocaleString()} faces`
                        : entry.status}
                    </div>
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>

        <footer className="px-5 py-3 border-t border-ink-700 bg-ink-900 sticky bottom-0">
          <div className="kicker normal-case tracking-normal font-mono text-[11px] text-mist-400">{engineLine}</div>
          {health?.hardwareLabel && (
            <div className="kicker mt-1 truncate">{health.hardwareLabel}</div>
          )}
        </footer>
      </aside>

      {/* ------------------------------------------------- scene + progress */}
      <main className="h-full min-h-0 max-md:h-[100vh] grid grid-rows-[minmax(0,1fr)_auto] min-w-0">
        <div className="relative min-h-0 flex overflow-clip">
        <div className="relative min-h-0 min-w-0 flex-1 bg-ink-950">
          <canvas ref={canvasRef} className="scene-canvas" />

          {/* nav + display controls */}
          <div className="absolute top-3 left-3 flex flex-wrap gap-2 max-w-[calc(100%-7.5rem)] z-10">
            <div className="panel flex p-1 gap-1" role="tablist" aria-label="Scene view">
              {[['viewer', '3D viewer'], ['turntable', 'Turntable']].map(([id, label]) => (
                <button
                  key={id}
                  role="tab"
                  aria-selected={sceneView === id}
                  className={`btn py-1 px-2.5 text-xs border-transparent bg-transparent ${sceneView === id ? 'btn-toggle-on' : ''}`}
                  onClick={() => setSceneView(id)}
                >{label}</button>
              ))}
            </div>

            {sceneView === 'viewer' && (
            <div className="panel flex p-1 gap-1">
              {NAV_MODES.map((mode) => (
                <button
                  key={mode.id}
                  className={`btn py-1 px-2.5 text-xs border-transparent bg-transparent ${display.nav === mode.id ? 'btn-toggle-on' : ''}`}
                  onClick={() => {
                    setDisplay((d) => ({ ...d, nav: mode.id }));
                    viewerRef.current?.setNavMode(mode.id);
                    setNavHint(mode.hint);
                  }}
                >{mode.label}</button>
              ))}
            </div>
            )}

            {sceneView === 'viewer' && (
            <div className="panel flex p-1 gap-1">
              {['authored', 'shaded', 'flat'].map((mode) => (
                <button
                  key={mode}
                  className={`btn py-1 px-2.5 text-xs border-transparent bg-transparent capitalize ${display.shading === mode ? 'btn-toggle-on' : ''}`}
                  onClick={() => setDisplay((d) => ({ ...d, shading: mode }))}
                >{mode}</button>
              ))}
            </div>
            )}

            {sceneView === 'viewer' && (
            <div className="panel flex p-1 gap-1">
              {MEASURE_MODES.map((mode) => (
                <button
                  key={mode.id}
                  className={`btn py-1 px-2.5 text-xs border-transparent bg-transparent ${display.measure === mode.id ? 'btn-toggle-on' : ''}`}
                  onClick={() => setDisplay((d) => ({ ...d, measure: mode.id }))}
                >{mode.label}</button>
              ))}
            </div>
            )}

            {sceneView === 'viewer' && (
            <div className="panel flex p-1 gap-1" aria-label="Enhancement tools">
              {DRAWER_TABS.map(([id, label]) => (
                <button
                  key={id}
                  className={`btn py-1 px-2.5 text-xs border-transparent bg-transparent ${drawer === id ? 'btn-toggle-on' : ''}`}
                  aria-pressed={drawer === id}
                  onClick={() => setDrawer((d) => (d === id ? null : id))}
                >{label}</button>
              ))}
              {lod?.current != null && (
                <span className="kicker self-center px-2">LOD {lod.current}{lod.loading != null ? ` → ${lod.loading}` : ''}</span>
              )}
            </div>
            )}
          </div>

          {/* The turntable covers the canvas rather than replacing it: the
              viewer keeps its WebGL context and loaded model, so switching back
              is instant and does not re-download the mesh. */}
          {sceneView === 'turntable' && (
            <div className="absolute inset-0 z-[5] bg-ink-950">
              <TurntablePlayer
                jobId={job?.id}
                hasVideo={Boolean(job?.result?.video)}
                onNotify={({ kind, message }) => setError(kind === 'error' ? message : null)}
              />
            </div>
          )}

          {/* layers */}
          {sceneView === 'viewer' && (
          <div className="absolute top-3 right-3 panel p-1 flex flex-col gap-1">
            {[
              ['mesh', 'Mesh', true],
              ['splat', 'Splat', sceneInfo?.splat],
              ['points', 'Cloud', sceneInfo?.cloud],
              ['grid', 'Grid', true],
            ].map(([key, label, available]) => (
              <button
                key={key}
                disabled={!available}
                className={`btn py-1 px-2.5 text-xs border-transparent bg-transparent justify-start ${display.layers[key] ? 'btn-toggle-on' : ''}`}
                onClick={() => setDisplay((d) => ({ ...d, layers: { ...d.layers, [key]: !d.layers[key] } }))}
              >{label}</button>
            ))}
            {job?.id && !sceneInfo?.cloud && (
              <button className="btn py-1 px-2.5 text-xs border-transparent bg-transparent justify-start" onClick={loadCloud}>
                + Load cloud
              </button>
            )}
          </div>
          )}

          {sceneView === 'viewer' && views && views.id === job?.id && (
            <DroneViews key={views.id} viewer={viewer} views={views} />
          )}

          {/* hints + measurement readout */}
          <div className="absolute bottom-3 left-3 right-3 flex items-end justify-between gap-3 pointer-events-none">
            <div className="panel px-3 py-2 kicker normal-case tracking-normal text-mist-400">{navHint}</div>
            {measureText && (
              <div className="panel px-3 py-2 text-sm text-jet-400 font-mono">{measureText}</div>
            )}
          </div>

          {/* downloads */}
          {modelAvailable && downloads && (
            <details className="absolute bottom-16 right-3 panel text-xs z-10">
              <summary className="px-3 py-1.5 cursor-pointer select-none">↓ Downloads</summary>
              <div className="p-1 flex flex-col gap-1 max-h-64 overflow-y-auto">
                {DOWNLOADS.filter(([kind]) => downloads[kind]).map(([kind, label]) => (
                  <a key={kind} href={fileUrl(job.id, kind)} download={downloads[kind].name} className="btn py-1 px-2.5 border-transparent bg-transparent justify-start">
                    ↓ {label} <span className="text-mist-400">{formatBytes(downloads[kind].bytes)}</span>
                  </a>
                ))}
              </div>
            </details>
          )}

          {loadPct && (
            <div className="absolute inset-0 grid place-items-center bg-ink-950/70 backdrop-blur-sm">
              <div className="text-center">
                <div className="text-3xl font-mono text-jet-400">{loadPct.pct}%</div>
                <div className="kicker mt-1">
                  Streaming model {formatBytes(loadPct.loaded)}{loadPct.total ? ` / ${formatBytes(loadPct.total)}` : ''}
                </div>
              </div>
            </div>
          )}

          {!job && !loadPct && (
            <div className="absolute inset-0 grid place-items-center pointer-events-none">
              <div className="text-center max-w-md px-6">
                <div className="text-4xl text-ink-600 mb-3">◈</div>
                <p className="text-sm text-mist-400">
                  Drop a drone video on the left to reconstruct a measurable 3D scene,
                  or open a previous reconstruction.
                </p>
              </div>
            </div>
          )}
        </div>
          {drawer && (
            <EnhancementDrawer
              tab={drawer} onTab={setDrawer} onClose={() => setDrawer(null)}
              job={job} viewer={viewer} lod={lod} modelVersion={modelVersion} onReload={reloadCurrent}
            />
          )}
        </div>

        {/* ------------------------------------------------- pipeline */}
        <section className="border-t border-ink-700 bg-ink-900 grid grid-cols-[minmax(0,1fr)_380px] max-lg:grid-cols-1 min-h-0" style={{ height: '248px' }}>
          <div className="p-4 overflow-y-auto">
            <div className="flex items-baseline justify-between mb-3">
              <div>
                <p className="kicker">Pipeline</p>
                <h2 className="text-sm">{job?.stage || (busy ? 'Starting…' : 'Idle')}</h2>
              </div>
              <div className="text-right">
                <div className="font-mono text-lg text-jet-400">{progress}%</div>
                {job?.etaMs > 0 && <div className="kicker">~{formatDuration(job.etaMs / 1000)} left</div>}
                {job?.elapsedMs > 0 && <div className="kicker">{formatDuration(job.elapsedMs / 1000)} elapsed</div>}
              </div>
            </div>

            <div className="h-1.5 rounded-full bg-ink-800 overflow-hidden mb-2">
              <div className="h-full bg-jet-500 transition-[width] duration-300" style={{ width: `${progress}%` }} />
            </div>

            {stall && (
              <div className={`panel px-3 py-2 mb-3 text-xs ${
                stall.level === 'bad'
                  ? 'border-red-900/60 bg-red-950/30 text-red-200'
                  : 'border-amber-900/50 bg-amber-950/20 text-amber-ui'
              }`}
              >
                {stall.text}
              </div>
            )}
            {busy && !stall && idleMs != null && (
              <div className="kicker mb-3">Engine active · last output {Math.round(idleMs / 1000)}s ago</div>
            )}
            {!busy && <div className="mb-3" />}

            <div className="grid grid-cols-3 gap-1.5">
              {stages.map((stage) => {
                // Already a percentage, same as job.progress.
                const pct = Math.max(0, Math.min(100, Number(stage.progress) || 0));
                const done = stage.status === 'complete' || pct >= 100;
                const active = !done && (stage.status === 'active' || pct > 0);
                return (
                  <div
                    key={stage.key}
                    className={`panel px-2.5 py-2 ${active ? 'border-jet-500/60' : ''}`}
                    title={stage.detail || stage.label}
                  >
                    <div className="flex items-center gap-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${done ? 'bg-jet-500' : active ? 'bg-amber-ui live-dot' : 'bg-ink-600'}`} />
                      <span className="text-[11px] truncate">{stage.label}</span>
                    </div>
                    <div className="kicker mt-0.5 flex justify-between gap-2">
                      <span>{done ? 'done' : active ? `${pct.toFixed(0)}%` : 'queued'}</span>
                      {stage.durationMs > 0 && <span className="text-mist-400">{formatDuration(stage.durationMs / 1000)}</span>}
                    </div>
                    {active && (
                      <div className="h-0.5 mt-1 rounded-full bg-ink-800 overflow-hidden">
                        <div className="h-full bg-amber-ui" style={{ width: `${pct}%` }} />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {telemetry && (
              <div className="flex flex-wrap gap-x-5 gap-y-1 mt-3 kicker">
                <span>GPU <b className="text-mist-200">{telemetry.gpu?.utilisationPercent ?? '—'}%</b></span>
                <span>VRAM <b className="text-mist-200">{telemetry.gpu?.memoryUsedMb ?? '—'} / {telemetry.gpu?.memoryTotalMb ?? '—'} MB</b></span>
                <span>TEMP <b className="text-mist-200">{telemetry.gpu?.temperatureC ?? '—'}°C</b></span>
                <span>RAM <b className="text-mist-200">{telemetry.memory?.usedPercent?.toFixed(0) ?? '—'}%</b></span>
                {telemetry.gpu?.name && <span className="truncate max-w-[220px]">{telemetry.gpu.name}</span>}
              </div>
            )}
          </div>

          {/* live log */}
          <div className="border-l border-ink-700 flex flex-col min-h-0 max-lg:hidden">
            <div className="px-3 py-2 border-b border-ink-700 flex items-center justify-between gap-2">
              <p className="kicker shrink-0">Engine log</p>
              <div className="flex items-center gap-1">
                <button
                  className={`btn py-0.5 px-2 text-[11px] ${showEngineLines ? 'btn-toggle-on' : ''}`}
                  onClick={() => setShowEngineLines((v) => !v)}
                  title="Raw COLMAP / OpenMVS / ffmpeg output"
                >raw</button>
                <button
                  className={`btn py-0.5 px-2 text-[11px] ${autoScroll ? 'btn-toggle-on' : ''}`}
                  onClick={() => setAutoScroll((v) => !v)}
                  title="Follow new lines"
                >follow</button>
                <button className="btn py-0.5 px-2 text-[11px]" onClick={() => setLog([])}>clear</button>
              </div>
            </div>
            <div
              ref={logRef}
              className="flex-1 overflow-y-auto px-3 py-2"
              onScroll={(e) => {
                // Turning autoscroll off the moment the user scrolls up is the
                // difference between a readable log and one that yanks away.
                const el = e.currentTarget;
                const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
                if (!atBottom && autoScroll) setAutoScroll(false);
              }}
            >
              {visibleLog.length === 0 && (
                <div className="log-line text-mist-400">
                  {busy ? 'Waiting for engine output…' : 'No job running.'}
                </div>
              )}
              {visibleLog.map((entry) => (
                <div
                  key={entry.seq ?? `${entry.at}-${entry.text}`}
                  className={`log-line whitespace-pre-wrap break-all ${
                    entry.kind === 'error' ? 'text-red-300'
                      : entry.kind === 'warn' ? 'text-amber-ui'
                        : entry.kind === 'ok' ? 'text-jet-400'
                          : entry.kind === 'engine' ? 'text-mist-400/55'
                            : 'text-mist-300'
                  }`}
                >
                  <span className="text-mist-400/40">{clockOf(entry.at)} </span>
                  {entry.text}
                </div>
              ))}
            </div>
            <div className="px-3 py-1.5 border-t border-ink-700 kicker flex justify-between">
              <span>{visibleLog.length} / {log.length} lines</span>
              {!autoScroll && <span className="text-amber-ui">paused — press follow</span>}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
