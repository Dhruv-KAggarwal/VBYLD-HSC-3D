/** Run the confidence analysis and explore its results. */

import { useEffect, useState } from 'react';
import { getConfidence, getConfidenceRegions, startConfidence } from '../api.js';
import { ConfidenceLegend } from './ConfidenceLegend.jsx';
import { ConfidenceInspector } from './ConfidenceInspector.jsx';
import { ReconstructionQuality } from './ReconstructionQuality.jsx';
import { Collapsible, Field, Notice } from './ui.jsx';

export const CONFIDENCE_DEFAULTS = Object.freeze({
  // Mirrors DEFAULT_WEIGHTS in lib/confidence/scoring.mjs.
  weights: { coverage: 25, diversity: 20, reprojection: 20, support: 15, quality: 10, depth: 10 },
  lowThreshold: 40,
  resolution: 640,
});

const LABELS = {
  coverage: 'Camera coverage', diversity: 'Viewpoint diversity', reprojection: 'Reprojection consistency',
  support: 'Point-cloud support', quality: 'Image quality', depth: 'Depth consistency',
};

export function ConfidencePanel({
  job, viewer, artifacts, busy, onRun, overlay, onOverlay, record, videos, approximate, overlayNotice,
}) {
  const [settings, setSettings] = useState(() => structuredClone(CONFIDENCE_DEFAULTS));
  const [report, setReport] = useState(null);
  const [regions, setRegions] = useState(null);
  const [error, setError] = useState(null);
  const [wireframe, setWireframe] = useState(false);
  const which = artifacts?.activeModel;
  const state = artifacts?.artifacts[which === 'repaired' ? 'confidenceRepaired' : 'confidence'];

  useEffect(() => {
    let stale = false;
    setReport(null);
    setRegions(null);
    setError(null);
    if (job?.id && state?.current) {
      Promise.all([getConfidence(job.id, which), getConfidenceRegions(job.id, which)])
        .then(([r, g]) => { if (!stale) { setReport(r); setRegions(g); } })
        .catch((e) => { if (!stale) setError(e.message); });
    }
    return () => { stale = true; };
  }, [job?.id, which, state?.current, state?.createdAt]);

  return (
    <div className="space-y-3">
      <Collapsible title="Analysis settings" open={!state?.current}>
        <p className="kicker normal-case tracking-normal">
          Weights are engineering defaults, not calibrated probabilities. They are normalised over whatever evidence exists.
        </p>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(settings.weights).map(([key, value]) => (
            <Field key={key} label={LABELS[key]}>
              <input type="number" min="0" max="100" className="field" disabled={busy} value={value} onChange={(e) => setSettings((s) => ({ ...s, weights: { ...s.weights, [key]: Number(e.target.value) } }))} />
            </Field>
          ))}
          <Field label="Low threshold">
            <input type="number" min="5" max="80" className="field" disabled={busy} value={settings.lowThreshold} onChange={(e) => setSettings((s) => ({ ...s, lowThreshold: Number(e.target.value) }))} />
          </Field>
          <Field label="Visibility resolution">
            <select className="field" disabled={busy} value={settings.resolution} onChange={(e) => setSettings((s) => ({ ...s, resolution: Number(e.target.value) }))}>
              {[320, 480, 640, 960, 1280].map((r) => <option key={r} value={r}>{r} px</option>)}
            </select>
          </Field>
        </div>
      </Collapsible>
      <button type="button" className="btn btn-primary" disabled={busy || !artifacts?.hasModel} onClick={() => onRun(() => startConfidence(job.id, settings))}>
        {artifacts?.artifacts.confidence.exists ? 'Analyse again' : 'Analyse confidence'}
      </button>
      {error && <Notice kind="error">{error}</Notice>}
      {artifacts && !state?.current && (
        <Notice kind={state?.exists ? 'warn' : 'info'}>
          Confidence is unavailable for the {which} model{state?.exists ? `: ${state.reason}` : ' until it is analysed'}.
        </Notice>
      )}
      {state?.current && (
        <>
          {overlayNotice && (
            <Notice kind="warn">
              {overlayNotice}{' '}
              {viewer?.lod && <button type="button" className="btn py-0 px-2 text-[11px]" onClick={() => viewer.lod.setManualLevel(0)}>Show full resolution</button>}
            </Notice>
          )}
          <ConfidenceLegend overlay={overlay} onOverlay={onOverlay} wireframe={wireframe} onWireframe={(v) => { setWireframe(v); viewer.setWireframe(v); }} />
          <ConfidenceInspector record={record} videos={videos} approximate={approximate} />
          {report && (
            <ReconstructionQuality
              job={job} report={report} regions={regions}
              onFocus={(region) => { viewer.focusRegion(region); if (overlay.mode === 'off') onOverlay({ mode: 'confidence' }); }}
            />
          )}
        </>
      )}
    </div>
  );
}
