/** Mesh quality analysis, repair settings, repair results and provenance. */

import { useCallback, useEffect, useState } from 'react';
import { fileUrl, getMeshQuality, getRepairReport, startRepair } from '../api.js';
import { PROVENANCE_COLORS, PROVENANCE_LABELS } from '../viewer/overlay-renderer.js';
import { MeshComparison } from './MeshComparison.jsx';
import { Collapsible, Field, Notice, Segmented, Stats, Toggle, formatDuration, show } from './ui.jsx';

export const REPAIR_DEFAULTS = Object.freeze({
  preset: 'conservative',
  removeFloating: true,
  floatingAggressiveness: 'conservative',
  removeDuplicates: true,
  repairNonManifold: true,
  fixOrientation: true,
  fillHoles: true,
  maxHoleUnit: 'm',
  maxHoleSize: 1,
  holeStrategy: 'auto',
  preserveOpenings: true,
  smoothing: 'off',
  smoothingIterations: 5,
  edgePreservation: 0.7,
  preserveTextures: true,
});

// Mirrors REPAIR_PRESETS in lib/meshrepair/repair.mjs.
const PRESETS = {
  conservative: { floatingAggressiveness: 'conservative', maxHoleUnit: 'm', maxHoleSize: 1, smoothing: 'off' },
  balanced: { floatingAggressiveness: 'balanced', maxHoleUnit: 'm', maxHoleSize: 2.5, smoothing: 'light' },
  aggressive: { floatingAggressiveness: 'aggressive', maxHoleUnit: 'm', maxHoleSize: 5, smoothing: 'moderate' },
};

export function RepairSettings({ value, onChange, disabled }) {
  const set = (changes) => onChange({ ...value, ...changes });
  const toggle = (key, label, note) => (
    <Toggle checked={value[key]} disabled={disabled} label={label} note={note} onChange={(v) => set({ [key]: v })} />
  );
  return (
    <div className="space-y-2.5">
      <Field label="Repair preset">
        <Segmented
          label="Repair preset" disabled={disabled} value={value.preset}
          options={[['conservative', 'Conservative'], ['balanced', 'Balanced'], ['aggressive', 'Aggressive']]}
          onChange={(preset) => set({ preset, ...PRESETS[preset] })}
        />
      </Field>
      {toggle('removeFloating', 'Remove floating geometry', 'Small islands that are also far from structure or lack point support.')}
      {value.removeFloating && (
        <Segmented
          label="Floating geometry aggressiveness" disabled={disabled} value={value.floatingAggressiveness}
          options={[['conservative', 'Conservative'], ['balanced', 'Balanced'], ['aggressive', 'Aggressive']]}
          onChange={(floatingAggressiveness) => set({ floatingAggressiveness })}
        />
      )}
      {toggle('removeDuplicates', 'Remove duplicate geometry', 'Duplicate vertices and faces; texture seams are never merged.')}
      {toggle('repairNonManifold', 'Repair non-manifold geometry', 'Removes loose fins only where that is safe.')}
      {toggle('fixOrientation', 'Correct face orientation')}
      {toggle('fillHoles', 'Fill small holes')}
      {value.fillHoles && (
        <>
          <div className="grid grid-cols-2 gap-2">
            <Field label="Maximum hole size">
              <input type="number" min="0.1" step="0.1" className="field" disabled={disabled} value={value.maxHoleSize} onChange={(e) => set({ maxHoleSize: e.target.value })} />
            </Field>
            <Field label="Unit">
              <select className="field" disabled={disabled} value={value.maxHoleUnit} onChange={(e) => set({ maxHoleUnit: e.target.value, maxHoleSize: e.target.value === 'm' ? 1 : 0.5 })}>
                <option value="m">metres</option>
                <option value="percent">% of scene size</option>
              </select>
            </Field>
          </div>
          <Field label="Filling method">
            <select className="field" disabled={disabled} value={value.holeStrategy} onChange={(e) => set({ holeStrategy: e.target.value })}>
              <option value="auto">Automatic</option>
              <option value="planar">Planar</option>
              <option value="boundary">Boundary triangulation</option>
              <option value="curvature">Curvature-aware</option>
            </select>
          </Field>
          {toggle('preserveOpenings', 'Preserve architectural openings', 'Leaves likely windows, doors and see-through gaps open.')}
        </>
      )}
      <Field label="Smoothing">
        <Segmented
          label="Smoothing strength" disabled={disabled} value={value.smoothing}
          options={[['off', 'Off'], ['light', 'Light'], ['moderate', 'Moderate'], ['strong', 'Strong'], ['custom', 'Custom']]}
          onChange={(smoothing) => set({ smoothing })}
        />
      </Field>
      {value.smoothing === 'custom' && (
        <Field label="Iterations">
          <input type="number" min="1" max="60" className="field" disabled={disabled} value={value.smoothingIterations} onChange={(e) => set({ smoothingIterations: e.target.value })} />
        </Field>
      )}
      {value.smoothing !== 'off' && (
        <Field label={`Preserve sharp edges: ${Math.round(value.edgePreservation * 100)}%`}>
          <input type="range" min="0" max="1" step="0.05" className="w-full" disabled={disabled} value={value.edgePreservation} onChange={(e) => set({ edgePreservation: Number(e.target.value) })} />
        </Field>
      )}
      {toggle('preserveTextures', 'Preserve textures', 'Off replaces the atlas with per-vertex colour.')}
      <Toggle checked disabled label="Keep original mesh" note="Always on: the original reconstruction is never modified." onChange={() => {}} />
    </div>
  );
}

const QUALITY_ROWS = [
  ['vertices', 'Vertices'], ['triangles', 'Triangles'], ['connectedComponents', 'Connected components'],
  ['floatingComponents', 'Floating components'], ['degenerateFaces', 'Degenerate faces'],
  ['duplicateVertices', 'Duplicate vertices'], ['duplicateFaces', 'Duplicate faces'],
  ['isolatedVertices', 'Isolated vertices'], ['invalidIndices', 'Invalid indices'],
  ['boundaryEdges', 'Boundary edges'], ['boundaryLoops', 'Boundary loops'],
  ['nonManifoldEdges', 'Non-manifold edges'], ['nonManifoldVertices', 'Non-manifold vertices'],
  ['inconsistentOrientationEdges', 'Inconsistent orientation'], ['thinTriangles', 'Very thin triangles'],
];

export function MeshRepairPanel({
  job, viewer, artifacts, busy, onRun, overlay, onOverlay, onActivate, taskToken,
}) {
  const [settings, setSettings] = useState({ ...REPAIR_DEFAULTS });
  const [report, setReport] = useState(null);
  const [quality, setQuality] = useState(null);
  const [error, setError] = useState(null);
  const repair = artifacts?.artifacts.repair;

  useEffect(() => {
    let stale = false;
    setReport(null);
    if (job?.id && repair?.current) getRepairReport(job.id).then((r) => { if (!stale) setReport(r); }).catch((e) => { if (!stale) setError(e.message); });
    return () => { stale = true; };
  }, [job?.id, repair?.current, repair?.createdAt]);

  useEffect(() => { setQuality(null); setError(null); }, [job?.id, repair?.createdAt]);

  const analyse = useCallback(async () => {
    setError(null);
    try {
      const result = await getMeshQuality(job.id);
      if (result.pending) onRun(null, result.task);
      else setQuality(result);
    } catch (e) { setError(e.message); }
  }, [job?.id, onRun]);

  // The analysis was started as a task; fetch it once that finishes.
  useEffect(() => {
    if (taskToken?.operation === 'analyze' && taskToken.status === 'complete') analyse();
  }, [taskToken, analyse]);

  const shown = quality || (report ? { original: report.before, repaired: report.after } : null);

  return (
    <div className="space-y-3">
      <Collapsible title="Repair settings" open={!repair?.current}>
        <RepairSettings value={settings} onChange={setSettings} disabled={busy} />
      </Collapsible>
      <div className="flex flex-wrap gap-2">
        <button type="button" className="btn btn-primary" disabled={busy || !artifacts?.hasModel} onClick={() => onRun(() => startRepair(job.id, settings))}>
          {repair?.exists ? 'Repair again' : 'Repair mesh'}
        </button>
        <button type="button" className="btn" disabled={busy || !artifacts?.hasModel} onClick={analyse}>Analyse quality</button>
      </div>
      {error && <Notice kind="error">{error}</Notice>}
      {repair?.exists && !repair.current && <Notice kind="warn">The repaired model is out of date: {repair.reason}.</Notice>}

      {repair?.current && (
        <>
          <Field label="Active model" note="Levels of detail and confidence follow the active model.">
            <Segmented
              label="Active model" disabled={busy} value={artifacts.activeModel}
              options={[['original', 'Original'], ['repaired', 'Repaired']]} onChange={onActivate}
            />
          </Field>
          <div className="panel p-3 space-y-2">
            <Toggle
              checked={overlay.mode === 'provenance'}
              label="Show repaired geometry"
              note="Colours faces by where they came from. Original surfaces keep their appearance."
              onChange={(on) => onOverlay({ mode: on ? 'provenance' : 'off' })}
            />
            <fieldset className="space-y-1">
              <legend className="sr-only">Geometry provenance legend</legend>
              {PROVENANCE_LABELS.map((label, code) => {
                const share = report?.provenance?.find((p) => p.code === code);
                const visible = !overlay.hiddenProvenance.includes(code);
                return (
                  <label key={label} className="flex items-center gap-2 text-xs cursor-pointer">
                    <input
                      type="checkbox" checked={code === 0 || visible} disabled={code === 0}
                      onChange={() => onOverlay({ hiddenProvenance: visible ? [...overlay.hiddenProvenance, code] : overlay.hiddenProvenance.filter((c) => c !== code) })}
                    />
                    <span className="w-3 h-3 rounded-sm border border-ink-600" style={{ background: code ? PROVENANCE_COLORS[code] : 'transparent' }} />
                    <span className="flex-1">{label}</span>
                    <span className="font-mono text-mist-400">{share ? `${share.areaPercent}%` : ''}</span>
                  </label>
                );
              })}
            </fieldset>
          </div>
          <MeshComparison job={job} viewer={viewer} report={report} />
        </>
      )}

      {report && (
        <div className="panel px-3 py-2">
          <p className="kicker mb-1">Repair statistics</p>
          <Stats
            rows={[
              ['Triangles before', show(report.originalTriangles)],
              ['Triangles after', show(report.repairedTriangles)],
              ['Removed components', show(report.removedComponents)],
              ['Removed faces', show(report.removedFaces)],
              ['Filled holes', show(report.filledHoles)],
              report.holes && ['Holes left open', show(report.holes.leftOpen)],
              report.holes && ['Likely openings kept', show(report.holes.leftOpenByClass.suspectedOpening)],
              ['Surface area change', show(report.surfaceAreaChangeM2, report.scaleReliable ? ' m²' : ' m² (approx.)')],
              ['Modified vertices', show(report.modifiedVertices)],
              ['Estimated geometry', show(report.estimatedGeometryPercent, '%')],
              ['Processing time', formatDuration(report.processingMs / 1000)],
              report.textures && ['Textures intact', report.textures.valid ? 'Yes' : 'No'],
            ]}
          />
          {report.warnings.map((w) => <p key={w} className="text-xs text-amber-ui mt-1">{w}</p>)}
          <div className="flex flex-wrap gap-1 mt-2">
            <a className="btn py-0.5 px-2 text-[11px]" href={fileUrl(job.id, 'repaired')} download="model-repaired.glb">Repaired GLB</a>
            <a className="btn py-0.5 px-2 text-[11px]" href={fileUrl(job.id, 'repair-report')} download="repair-report.json">Report JSON</a>
          </div>
        </div>
      )}

      {shown && (
        <Collapsible title="Mesh quality analysis" open={Boolean(quality)}>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-mist-400 text-left">
                <th className="font-normal">Measure</th><th className="font-normal text-right">Original</th>
                {shown.repaired && <th className="font-normal text-right">Repaired</th>}
              </tr>
            </thead>
            <tbody className="font-mono">
              {QUALITY_ROWS.map(([key, label]) => (
                <tr key={key} className="border-t border-ink-700">
                  <th scope="row" className="font-sans font-normal text-left text-mist-400">{label}</th>
                  <td className="text-right">{show(shown.original[key])}</td>
                  {shown.repaired && <td className="text-right">{show(shown.repaired[key])}</td>}
                </tr>
              ))}
              {[
                ['Surface area (m²)', (q) => q.surfaceAreaM2],
                ['Self-intersecting (sampled)', (q) => q.selfIntersections && `${q.selfIntersections.intersectingFaces}/${q.selfIntersections.sampledFaces}`],
                ['Median normal deviation (°)', (q) => q.surfaceNoise?.medianDegrees],
                ['Unsupported area (%)', (q) => q.unsupported?.unsupportedAreaPercent],
              ].map(([label, read]) => (
                <tr key={label} className="border-t border-ink-700">
                  <th scope="row" className="font-sans font-normal text-left text-mist-400">{label}</th>
                  <td className="text-right">{show(read(shown.original))}</td>
                  {shown.repaired && <td className="text-right">{show(read(shown.repaired))}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </Collapsible>
      )}
    </div>
  );
}
