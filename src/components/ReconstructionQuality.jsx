/** Surface-area-weighted quality dashboard and the low-confidence region browser. */

import { useState } from 'react';
import { fileUrl } from '../api.js';
import { PROVENANCE_LABELS } from '../viewer/overlay-renderer.js';
import { Collapsible, Stats, show } from './ui.jsx';

function Bars({ rows, label }) {
  const peak = Math.max(1, ...rows.map((r) => r.value));
  return (
    <ul aria-label={label} className="space-y-0.5">
      {rows.map((row) => (
        <li key={row.name} className="grid grid-cols-[56px_1fr_44px] items-center gap-2 text-[11px] font-mono">
          <span className="text-mist-400">{row.name}</span>
          <span className="h-2 rounded-sm bg-ink-800 overflow-hidden"><span className="block h-full bg-jet-500" style={{ width: `${(row.value / peak) * 100}%` }} /></span>
          <span className="text-right">{row.value}%</span>
        </li>
      ))}
    </ul>
  );
}

export function ReconstructionQuality({ job, report, regions, onFocus }) {
  const [selected, setSelected] = useState(null);
  const s = report.summary;
  return (
    <div className="space-y-3">
      <div className="panel px-3 py-2">
        <p className="kicker mb-1">Quality (weighted by surface area)</p>
        <Stats
          rows={[
            ['Mean confidence', show(s.meanScore)],
            ['Median confidence', show(s.medianScore)],
            ['Evidence coverage', show(s.evidenceCoveragePercent, '%')],
            ['High (60–100)', show(s.highPercent, '%')],
            ['Moderate (40–59)', show(s.moderatePercent, '%')],
            ['Low (0–39)', show(s.lowPercent, '%')],
            ['Unknown', show(s.unknownPercent, '%')],
            ['Estimated geometry', show(s.estimatedGeometryPercent, '%')],
            ['Repaired geometry', show(s.repairedGeometryPercent, '%')],
            ['Low-confidence regions', show(regions?.totalRegions)],
            ['Registered cameras', show(report.cameras)],
          ]}
        />
      </div>
      <Collapsible title="Confidence histogram" open>
        <Bars label="Share of scored surface by confidence" rows={s.histogram.map((h) => ({ name: `${h.from}–${h.to}`, value: h.areaPercent }))} />
      </Collapsible>
      <Collapsible title="Supporting cameras per surface">
        <Bars label="Share of surface by number of cameras" rows={s.cameraDistribution.map((c) => ({ name: c.cameras, value: c.areaPercent }))} />
      </Collapsible>
      <Collapsible title="Videos">
        {report.videos.map((video) => (
          <Stats
            key={video.id}
            rows={[
              ['Video', video.label],
              ['Registered images', video.selectedFrames ? `${video.registeredImages} of ${video.selectedFrames}` : show(video.registeredImages)],
              ['Surface observed', show(s.videoSurfacePercent.find((v) => v.id === video.id)?.areaPercent, '%')],
            ]}
          />
        ))}
      </Collapsible>
      <Collapsible title="Evidence sources">
        {report.metrics.map((metric) => (
          <div key={metric.key} className="text-xs">
            <div className="flex justify-between"><span>{metric.label} ({Math.round(metric.weight * 100)}%)</span><span className={metric.status === 'unavailable' ? 'text-amber-ui' : 'text-jet-400'}>{metric.status}</span></div>
            <div className="kicker normal-case tracking-normal">{metric.source}</div>
          </div>
        ))}
      </Collapsible>

      {regions && (
        <Collapsible title={`Low-confidence regions (${regions.totalRegions})`} open>
          {regions.regions.length === 0 && <p className="text-xs text-mist-400">No region is large enough to list.</p>}
          <ul className="space-y-1 max-h-72 overflow-y-auto">
            {regions.regions.map((region) => (
              <li key={region.id}>
                <button
                  type="button"
                  className={`panel w-full text-left px-2 py-1.5 cursor-pointer text-xs ${selected === region.id ? 'border-jet-500' : ''}`}
                  aria-expanded={selected === region.id}
                  onClick={() => { setSelected(region.id); onFocus(region); }}
                >
                  <span className="flex justify-between gap-2">
                    <span>{region.id.replace('region_', 'Region ')} · {region.orientation}</span>
                    <span className="font-mono">{region.areaM2.toLocaleString()} m²</span>
                  </span>
                  <span className="kicker normal-case tracking-normal block">
                    avg {region.averageConfidence ?? 'unknown'} · min {region.minimumConfidence ?? 'unknown'} · {region.unknownPercent}% unknown · {region.meanSupportingCameras} cameras
                  </span>
                </button>
                {selected === region.id && (
                  <div className="px-2 py-1 text-xs space-y-1">
                    <p className="kicker normal-case tracking-normal">
                      Provenance: {region.provenancePercent.map((p, i) => (p ? `${PROVENANCE_LABELS[i]} ${p}%` : null)).filter(Boolean).join(', ')}
                      {region.supportingVideos.length ? ` · videos ${region.supportingVideos.join(', ')}` : ''}
                    </p>
                    {region.causes.map((c) => <p key={c.key} className="text-amber-ui">{c.label} ({Math.round(c.areaShare * 100)}%)</p>)}
                    {region.recommendations.map((r) => <p key={r}>→ {r}</p>)}
                    {!region.causes.length && <p className="text-mist-400">No single cause covers enough of this region to recommend an action.</p>}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </Collapsible>
      )}

      <div className="flex flex-wrap gap-1">
        {[['confidence-glb', 'Heatmap GLB'], ['confidence-ply', 'PLY'], ['confidence-bin', 'Binary'], ['confidence-report', 'Report']].map(([kind, text]) => (
          <a key={kind} className="btn py-0.5 px-2 text-[11px]" href={fileUrl(job.id, kind)} download>↓ {text}</a>
        ))}
      </div>
      <p className="kicker normal-case tracking-normal">Downloads describe the original mesh. {report.disclaimer}</p>
    </div>
  );
}
