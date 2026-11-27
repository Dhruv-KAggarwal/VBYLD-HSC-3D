/** The evidence behind the face the operator clicked. */

import { PROVENANCE_LABELS } from '../viewer/overlay-renderer.js';
import { Stats, show } from './ui.jsx';

const level = (value) => {
  if (value == null) return 'Unavailable';
  const word = value >= 75 ? 'High' : value >= 40 ? 'Moderate' : 'Low';
  return `${word} (${value}/100)`;
};

export function ConfidenceInspector({ record, videos, approximate }) {
  if (!record) return <div className="panel p-3 text-xs text-mist-400">Click a surface on the model to inspect its evidence.</div>;
  const supporting = videos?.filter((_, i) => record.videoMask & (1 << i)) || [];
  const availability = record.availability >= 80 ? 'High' : record.availability >= 50 ? 'Partial' : 'Insufficient';
  return (
    <div className="panel p-3" aria-live="polite">
      <p className="text-sm mb-1">
        Reconstruction confidence: <b className="font-mono">{record.score == null ? 'Unknown' : `${record.score}/100`}</b>
      </p>
      <Stats
        rows={[
          ['Evidence availability', `${availability} (${record.availability}%)`],
          ['Supporting cameras', show(record.cameras)],
          ['Supporting source videos', record.cameras == null ? 'Unavailable' : `${record.videos}${supporting.length ? `: ${supporting.map((v) => v.label).join(', ')}` : ''}`],
          ['Median reprojection error', show(record.reprojectionError, ' px', 2)],
          ['Viewpoint diversity', level(record.diversity)],
          ['Angular spread of views', show(record.spread, '°')],
          ['Mean viewing angle off normal', show(record.incidence, '°')],
          ['Camera coverage', level(record.coverage)],
          ['Dense point support', level(record.support)],
          ['Image quality', level(record.quality)],
          ['Depth consistency', level(record.depth)],
          ['Geometry provenance', PROVENANCE_LABELS[record.provenance] || 'Unavailable'],
          ['Repair status', record.provenance === 0 ? 'Unmodified' : record.provenance === 5 ? 'Flagged unsupported' : 'Modified by repair'],
          ['Face', record.face.toLocaleString()],
        ]}
      />
      {approximate && <p className="kicker normal-case tracking-normal mt-1">Values on a simplified level are transferred from the full-resolution surface.</p>}
      <p className="kicker normal-case tracking-normal mt-1">Relative confidence, not a certified accuracy figure.</p>
    </div>
  );
}
