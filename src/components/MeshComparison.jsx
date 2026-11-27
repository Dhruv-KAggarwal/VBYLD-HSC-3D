/** Original-versus-repaired comparison controls for the shared viewer. */

import { useCallback, useEffect, useState } from 'react';
import { fileUrlFresh, getBinary, provenanceDataPath } from '../api.js';
import { parseProvenance } from '../viewer/sidecar.js';
import { Notice, Segmented, Toggle } from './ui.jsx';

export function MeshComparison({ job, viewer, report }) {
  const [on, setOn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [view, setView] = useState({ mode: 'slider', split: 0.5, showRemoved: false, showRepaired: false });
  const [wireframe, setWireframe] = useState(false);

  const stop = useCallback(() => {
    viewer?.comparison.disable();
    viewer?.setWireframe(false);
    setOn(false);
    setWireframe(false);
  }, [viewer]);

  useEffect(() => stop, [job?.id, report?.generatedAt, stop]);

  const start = async () => {
    setLoading(true);
    setError(null);
    try {
      const token = report?.generatedAt || Date.now();
      const provenance = parseProvenance(await getBinary(provenanceDataPath(job.id)));
      await viewer.comparison.enable({
        originalUrl: fileUrlFresh(job.id, 'original', token),
        repairedUrl: fileUrlFresh(job.id, 'repaired', token),
        removedUrl: report?.removedGeometryFile ? fileUrlFresh(job.id, 'removed', token) : null,
        provenance,
      });
      viewer.comparison.set(view);
      setOn(true);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setLoading(false);
    }
  };

  const change = (changes) => {
    setView((v) => ({ ...v, ...changes }));
    viewer.comparison.set(changes);
  };

  return (
    <div className="panel p-3 space-y-2">
      <div className="flex justify-between items-center">
        <p className="kicker">Before / after comparison</p>
        <button type="button" className={`btn py-0.5 px-2 text-[11px] ${on ? 'btn-toggle-on' : ''}`} disabled={loading || !report} onClick={on ? stop : start}>
          {loading ? 'Loading…' : on ? 'Close' : 'Compare'}
        </button>
      </div>
      {error && <Notice kind="error">{error}</Notice>}
      {on && (
        <>
          <Segmented
            label="Comparison mode" value={view.mode} onChange={(mode) => change({ mode })}
            options={[['original', 'Original'], ['repaired', 'Repaired'], ['side', 'Side by side'], ['slider', 'Slider']]}
          />
          {view.mode === 'slider' && (
            <label className="block text-xs">
              <span className="flex justify-between text-mist-400"><span>Original</span><span>Repaired</span></span>
              <input type="range" min="0.02" max="0.98" step="0.01" className="w-full" aria-label="Comparison slider position" value={view.split} onChange={(e) => change({ split: Number(e.target.value) })} />
            </label>
          )}
          {view.mode === 'side' && <p className="kicker normal-case tracking-normal">Original on the left, repaired on the right. The camera is shared.</p>}
          <Toggle checked={view.showRepaired} label="Highlight repaired areas" onChange={(showRepaired) => change({ showRepaired })} />
          <Toggle checked={view.showRemoved} disabled={!report?.removedGeometryFile} label="Highlight removed geometry (red)" onChange={(showRemoved) => change({ showRemoved })} />
          <Toggle checked={wireframe} label="Wireframe" onChange={(value) => { setWireframe(value); viewer.setWireframe(value); }} />
        </>
      )}
    </div>
  );
}
