/** Generate levels of detail, choose how they are displayed, and compare them. */

import { useEffect, useState } from 'react';
import { fileUrl, getLodManifest, lodUrl, startLod } from '../api.js';
import { LOD_MODES } from '../viewer/lod-manager.js';
import { PerformanceMonitor } from './PerformanceMonitor.jsx';
import { Collapsible, Field, Notice, Segmented, Toggle, formatBytes, show } from './ui.jsx';

export const LOD_DEFAULTS = Object.freeze({
  levels: [
    { level: 1, ratio: 0.6, texture: 4096, enabled: true },
    { level: 2, ratio: 0.3, texture: 2048, enabled: true },
    { level: 3, ratio: 0.1, texture: 2048, enabled: true },
    { level: 4, ratio: 0.03, texture: 1024, enabled: true },
  ],
  textureFormat: 'auto',
  textureQuality: 90,
  quantize: false,
  errorScale: 1,
});

function LodSettings({ value, onChange, support, disabled }) {
  const set = (changes) => onChange({ ...value, ...changes });
  const setLevel = (level, changes) => set({ levels: value.levels.map((l) => (l.level === level ? { ...l, ...changes } : l)) });
  return (
    <div className="space-y-2.5">
      <table className="w-full text-xs">
        <thead><tr className="text-mist-400 text-left"><th className="font-normal">Level</th><th className="font-normal">Keep %</th><th className="font-normal">Texture px</th></tr></thead>
        <tbody>
          {value.levels.map((l) => (
            <tr key={l.level}>
              <th scope="row" className="font-normal text-left">
                <label className="flex items-center gap-1"><input type="checkbox" disabled={disabled} checked={l.enabled} onChange={(e) => setLevel(l.level, { enabled: e.target.checked })} />{l.level}</label>
              </th>
              <td><input type="number" min="1" max="90" className="field py-1" aria-label={`Level ${l.level} triangle percentage`} disabled={disabled || !l.enabled} value={Math.round(l.ratio * 100)} onChange={(e) => setLevel(l.level, { ratio: Number(e.target.value) / 100 })} /></td>
              <td>
                <select className="field py-1" aria-label={`Level ${l.level} texture size`} disabled={disabled || !l.enabled} value={l.texture} onChange={(e) => setLevel(l.level, { texture: Number(e.target.value) })}>
                  {[8192, 4096, 2048, 1024, 512, 256].map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Field label="Texture format" note="Textures are never enlarged.">
        <select className="field" disabled={disabled} value={value.textureFormat} onChange={(e) => set({ textureFormat: e.target.value })}>
          <option value="auto">Automatic (keep original, else JPEG)</option>
          <option value="jpeg">JPEG</option>
          <option value="png">PNG (lossless)</option>
          <option value="webp" disabled={support?.webp.available === false}>WebP{support?.webp.available === false ? ' (unavailable)' : ''}</option>
        </select>
      </Field>
      <Toggle checked={value.quantize} disabled={disabled} label="Quantise geometry" note="KHR_mesh_quantization: smaller files and GPU buffers." onChange={(quantize) => set({ quantize })} />
      <Field label={`Error tolerance: ${value.errorScale}×`} note="Lower is stricter; more levels may be rejected.">
        <input type="range" min="0.25" max="4" step="0.25" className="w-full" disabled={disabled} value={value.errorScale} onChange={(e) => set({ errorScale: Number(e.target.value) })} />
      </Field>
      {support && (
        <p className="kicker normal-case tracking-normal">
          Not available: {Object.entries(support).filter(([, v]) => !v.available).map(([k, v]) => `${k} (${v.reason})`).join('; ') || 'nothing'}.
        </p>
      )}
    </div>
  );
}

export function LODControl({ job, viewer, artifacts, lod, busy, onRun }) {
  const [settings, setSettings] = useState(() => structuredClone(LOD_DEFAULTS));
  const [manifest, setManifest] = useState(null);
  const [error, setError] = useState(null);
  const state = artifacts?.artifacts.lod;

  useEffect(() => {
    let stale = false;
    setManifest(null);
    if (job?.id && state?.current) getLodManifest(job.id).then((m) => { if (!stale) setManifest(m); }).catch((e) => { if (!stale) setError(e.message); });
    return () => { stale = true; };
  }, [job?.id, state?.current, state?.createdAt]);

  return (
    <div className="space-y-3">
      <Collapsible title="Level-of-detail settings" open={!state?.current}>
        <LodSettings value={settings} onChange={setSettings} disabled={busy} support={artifacts?.capabilities.compression} />
      </Collapsible>
      <button type="button" className="btn btn-primary" disabled={busy || !artifacts?.hasModel} onClick={() => onRun(() => startLod(job.id, settings))}>
        {state?.exists ? 'Regenerate levels' : 'Generate levels'}
      </button>
      {error && <Notice kind="error">{error}</Notice>}
      {state?.exists && !state.current && <Notice kind="warn">The levels are out of date: {state.reason}.</Notice>}

      {lod && (
        <>
          <Field label="Viewer mode">
            <Segmented
              label="Viewer detail mode" value={lod.mode} onChange={(mode) => viewer.lod?.setMode(mode)}
              options={Object.entries(LOD_MODES).map(([id, m]) => [id, m.label])}
            />
          </Field>
          {lod.error && <Notice kind="warn">{lod.error}</Notice>}
          {!lod.measurementReady && <p className="kicker normal-case tracking-normal">Measuring loads the full-resolution geometry first.</p>}
        </>
      )}
      <PerformanceMonitor viewer={viewer} lod={lod} />

      {manifest && (
        <div className="panel p-3 space-y-2">
          <p className="kicker">Compare levels (click to inspect)</p>
          <table className="w-full text-xs">
            <thead><tr className="text-mist-400"><th className="font-normal text-left">Level</th><th className="font-normal text-right">Triangles</th><th className="font-normal text-right">Size</th><th className="font-normal text-right">Error m</th></tr></thead>
            <tbody className="font-mono">
              {manifest.levels.map((l) => (
                <tr key={l.level} className={`border-t border-ink-700 ${lod?.current === l.level ? 'text-jet-400' : ''}`}>
                  <th scope="row" className="text-left">
                    <button type="button" className="btn py-0 px-2 text-[11px]" aria-pressed={lod?.current === l.level} onClick={() => viewer.lod?.setManualLevel(l.level)} title={l.method} aria-label={`Show level of detail ${l.level} (${l.method})`}>LOD {l.level}</button>
                  </th>
                  <td className="text-right">{show(l.triangleCount)}</td>
                  <td className="text-right">{formatBytes(l.fileSizeBytes)}</td>
                  <td className="text-right">{l.level === 0 ? '0' : show(l.geometricError, '', 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="kicker normal-case tracking-normal">Error: {manifest.geometricErrorDefinition} ({manifest.geometricErrorUnits}).</p>
          {manifest.skipped.map((s) => <p key={s.level} className="text-xs text-amber-ui">LOD {s.level} skipped: {s.reason}</p>)}
          {manifest.warnings.map((w) => <p key={w} className="text-xs text-amber-ui">{w}</p>)}
          <Toggle checked={Boolean(viewer?.wireframe)} label="Wireframe" onChange={(v) => { viewer.setWireframe(v); setManifest({ ...manifest }); }} />
          <div className="flex flex-wrap gap-1">
            {manifest.levels.filter((l) => l.level > 0).map((l) => (
              <a key={l.level} className="btn py-0.5 px-2 text-[11px]" href={lodUrl(job.id, l.level, manifest.generatedAt)} download={`lod${l.level}.glb`}>↓ LOD {l.level}</a>
            ))}
            <a className="btn py-0.5 px-2 text-[11px]" href={fileUrl(job.id, 'lod-report')} download="lod-report.json">↓ Report</a>
          </div>
        </div>
      )}
    </div>
  );
}
