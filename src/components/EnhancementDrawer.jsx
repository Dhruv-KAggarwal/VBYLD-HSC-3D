/**
 * The side drawer holding the four enhancement tools.
 *
 * It owns what they share: which derived products exist, the one running task,
 * and the single overlay drawn on whichever model (or level) is on screen.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  cancelTask, componentUrl, confidenceDataPath, getArtifacts, getBinary, getJobVideos, provenanceDataPath, setActiveModel, watchTask,
} from '../api.js';
import { OVERLAY_DEFAULTS } from '../viewer/overlay-renderer.js';
import { parseConfidence, parseProvenance } from '../viewer/sidecar.js';
import { VideoConnectivityGraph } from './VideoConnectivityGraph.jsx';
import { MeshRepairPanel } from './MeshRepairPanel.jsx';
import { LODControl } from './LODControl.jsx';
import { ConfidencePanel } from './ConfidencePanel.jsx';
import { Notice, TaskProgress } from './ui.jsx';

export const DRAWER_TABS = [
  ['videos', 'Videos'], ['repair', 'Repair'], ['optimize', 'Optimize'], ['confidence', 'Confidence'],
];

export function EnhancementDrawer({
  tab, onTab, onClose, job, viewer, lod, modelVersion, onReload,
}) {
  const [artifacts, setArtifacts] = useState(null);
  const [task, setTask] = useState(null);
  const [error, setError] = useState(null);
  const [overlay, setOverlay] = useState({ ...OVERLAY_DEFAULTS });
  const [overlayNotice, setOverlayNotice] = useState(null);
  const [record, setRecord] = useState(null);
  const [videos, setVideos] = useState([]);
  const data = useRef({ confidence: null });
  const binaries = useRef(new Map());
  const unwatch = useRef(null);
  const jobId = job?.id;
  const running = task && (task.status === 'queued' || task.status === 'running');
  const processing = job?.status === 'processing';

  const refresh = useCallback(async () => {
    if (!jobId) return null;
    try {
      const next = await getArtifacts(jobId);
      setArtifacts(next);
      setError(null);
      return next;
    } catch (failure) {
      setArtifacts(null);
      setError(failure.message);
      return null;
    }
  }, [jobId]);

  const follow = useCallback((initial) => {
    unwatch.current?.();
    setTask(initial);
    unwatch.current = watchTask(jobId, async (next) => {
      setTask(next);
      if (!next || next.status === 'queued' || next.status === 'running') return;
      binaries.current.clear();
      await refresh();
      // A finished repair or LOD run changes what should be on screen.
      if (next.status === 'complete' && ['repair', 'lod', 'completion'].includes(next.operation)) onReload();
    });
  }, [jobId, refresh, onReload]);

  // A new job (or the end of its reconstruction) resets everything.
  useEffect(() => {
    setTask(null);
    setArtifacts(null);
    setRecord(null);
    binaries.current.clear();
    unwatch.current?.();
    if (!jobId) return undefined;
    let stale = false;
    refresh().then((next) => {
      if (!stale && next?.task && (next.task.status === 'queued' || next.task.status === 'running')) follow(next.task);
    });
    getJobVideos(jobId).then((v) => { if (!stale) setVideos(v.videos.map((x) => ({ id: x.id, label: x.settings?.label || x.originalName || x.id }))); }).catch(() => {});
    return () => { stale = true; unwatch.current?.(); };
  }, [jobId, job?.status, refresh, follow]);

  /** Start a tool. `start` returns {task}; an already-started task may be given. */
  const run = useCallback(async (start, existing = null) => {
    setError(null);
    try {
      follow(existing || (await start()).task);
    } catch (failure) {
      setError(failure.message);
    }
  }, [follow]);

  const activate = useCallback(async (model) => {
    try {
      setArtifacts(await setActiveModel(jobId, model));
      binaries.current.clear();
      await refresh();
      onReload();
    } catch (failure) {
      setError(failure.message);
    }
  }, [jobId, onReload, refresh]);

  const binary = useCallback(async (path, parse) => {
    if (!binaries.current.has(path)) binaries.current.set(path, parse(await getBinary(path)));
    return binaries.current.get(path);
  }, []);

  // Keep the overlay matched to the model on screen.
  const wanted = overlay.mode;
  const level = lod?.current ?? 0;
  useEffect(() => {
    if (!viewer) return undefined;
    let stale = false;
    (async () => {
      setRecord(null);
      data.current.confidence = null;
      setOverlayNotice(null);
      if (!jobId || !artifacts) { viewer.setOverlayData(null); return; }
      // Overlays are recorded per face of the reconstruction, not of the filled copy.
      if (viewer.objects.mesh?.userData.completed) {
        viewer.setOverlayData(null);
        if (wanted !== 'off') setOverlayNotice('Turn off "Filled sides" to see this overlay.');
        return;
      }
      const repaired = artifacts.activeModel === 'repaired';
      const confidenceState = artifacts.artifacts[repaired ? 'confidenceRepaired' : 'confidence'];
      let confidence = null;
      let provenance = null;
      if (confidenceState?.current) {
        confidence = await binary(confidenceDataPath(jobId, artifacts.activeModel, level), parseConfidence)
          .catch(() => { if (!stale && level) setOverlayNotice(`Level ${level} carries no confidence data. Regenerate the levels, or view full resolution.`); return null; });
      }
      if (wanted === 'provenance' && repaired) {
        provenance = await binary(provenanceDataPath(jobId, level), parseProvenance).catch(() => null);
      }
      if (stale) return;
      data.current.confidence = confidence;
      if (wanted === 'off') { viewer.setOverlayData(null); return; }
      const source = wanted === 'confidence' ? confidence : provenance;
      if (!source) {
        viewer.setOverlayData(null);
        if (wanted === 'provenance') setOverlayNotice(repaired ? 'Provenance is not available for this level.' : 'Activate the repaired model to see where its geometry came from.');
        return;
      }
      try {
        viewer.setOverlayData({
          count: source.count,
          score: wanted === 'confidence' ? source.score : null,
          provenance: wanted === 'confidence' ? source.provenance : (f) => source.codes[f],
        });
        viewer.setOverlay(overlay);
      } catch (failure) {
        setOverlayNotice(failure.message);
      }
    })();
    return () => { stale = true; };
    // `overlay` changes are applied below without rebuilding geometry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewer, jobId, artifacts, wanted, level, modelVersion, binary]);

  useEffect(() => { viewer?.setOverlay(overlay); }, [viewer, overlay]);

  useEffect(() => {
    if (!viewer) return undefined;
    viewer.setInspect(tab === 'confidence');
    viewer.onInspect = ({ face }) => setRecord(data.current.confidence?.read(face) || null);
    if (tab !== 'confidence') viewer.clearHighlight();
    return () => { viewer.setInspect(false); viewer.onInspect = null; };
  }, [viewer, tab]);

  const changeOverlay = useCallback((changes) => setOverlay((o) => ({ ...o, ...changes })), []);
  const openComponent = useCallback(async (number) => {
    viewer.closeLod();
    try { await viewer.loadGlb(componentUrl(jobId, number)); } catch (failure) { setError(failure.message); }
  }, [viewer, jobId]);

  const busy = running || processing;
  const shared = { job, viewer, artifacts, busy, onRun: run, overlay, onOverlay: changeOverlay };

  return (
    <aside className="h-full min-h-0 w-[360px] shrink-0 max-w-[90vw] flex flex-col border-l border-ink-700 bg-ink-900 max-xl:absolute max-xl:right-0 max-xl:top-0 max-xl:z-30" aria-label="Enhancement tools">
      <div className="flex items-center gap-1 p-2 border-b border-ink-700" role="tablist">
        {DRAWER_TABS.map(([id, label]) => (
          <button key={id} type="button" role="tab" aria-selected={tab === id} className={`btn py-1 px-2 text-xs ${tab === id ? 'btn-toggle-on' : ''}`} onClick={() => onTab(id)}>{label}</button>
        ))}
        <button type="button" className="btn py-1 px-2 text-xs ml-auto" aria-label="Close tools" onClick={onClose}>×</button>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3" role="tabpanel">
        {!jobId && <Notice>Open or start a reconstruction to use these tools.</Notice>}
        {error && <Notice kind="error">{error}</Notice>}
        {processing && tab !== 'videos' && <Notice>These tools can be run once the reconstruction finishes.</Notice>}
        <TaskProgress task={task} onCancel={() => cancelTask(jobId).catch((e) => setError(e.message))} />
        {jobId && artifacts && !artifacts.hasModel && tab !== 'videos' && !processing && (
          <Notice kind="warn">This job has no 3D mesh, so mesh tools are unavailable.</Notice>
        )}
        {overlayNotice && tab !== 'confidence' && <Notice kind="warn">{overlayNotice}</Notice>}
        {/* Panels stay mounted so settings and open comparisons survive tab switches. */}
        {jobId && (
          <>
            {tab === 'videos' && <VideoConnectivityGraph job={job} onOpenComponent={openComponent} />}
            <div hidden={tab !== 'repair'}>
              <MeshRepairPanel key={jobId} {...shared} onActivate={activate} taskToken={task} />
            </div>
            <div hidden={tab !== 'optimize'}>
              <LODControl key={jobId} {...shared} lod={lod} />
            </div>
            <div hidden={tab !== 'confidence'}>
              <ConfidencePanel key={jobId} {...shared} record={record} videos={videos} approximate={level > 0} overlayNotice={overlayNotice} />
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
