/** Which videos see the same things, and which of them registered. */

import { useEffect, useState } from 'react';
import { componentUrl, getConnectivity, getJobVideos, jobThumbnailUrl } from '../api.js';
import { Notice, Stats, formatDuration, show } from './ui.jsx';

const EDGE = {
  strong: { color: '#2fd39a', width: 3, dash: null, label: 'Strong overlap' },
  weak: { color: '#f5b445', width: 2, dash: null, label: 'Weak overlap' },
  insufficient: { color: '#8ea3b8', width: 1, dash: '4 4', label: 'Too few matches to trust' },
};

function Graph({ connectivity, labelOf }) {
  const nodes = connectivity.videos;
  const size = 300;
  const radius = nodes.length > 1 ? 105 : 0;
  const place = new Map(nodes.map((n, i) => {
    const angle = (i / nodes.length) * Math.PI * 2 - Math.PI / 2;
    return [n.id, [size / 2 + radius * Math.cos(angle), size / 2 + radius * Math.sin(angle)]];
  }));
  const disconnected = new Set(connectivity.disconnected);
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full max-w-[300px] mx-auto" role="img" aria-label="Video connectivity graph">
      {connectivity.edges.filter((e) => EDGE[e.strength]).map((edge) => {
        const [x1, y1] = place.get(edge.a);
        const [x2, y2] = place.get(edge.b);
        const style = EDGE[edge.strength];
        return (
          <g key={`${edge.a}-${edge.b}`}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={style.color} strokeWidth={style.width} strokeDasharray={style.dash || undefined}>
              <title>{`${labelOf(edge.a)} and ${labelOf(edge.b)}: ${edge.verifiedPairs} verified pairs, ${edge.inlierMatches} matches`}</title>
            </line>
            <text x={(x1 + x2) / 2} y={(y1 + y2) / 2} fontSize="9" fill="#b6c6d6" textAnchor="middle">{edge.verifiedPairs}</text>
          </g>
        );
      })}
      {nodes.map((node, i) => {
        const [x, y] = place.get(node.id);
        const lost = disconnected.has(node.id);
        return (
          <g key={node.id}>
            <circle cx={x} cy={y} r="17" fill="#141c26" stroke={lost ? '#ef4444' : '#2fd39a'} strokeWidth="2" strokeDasharray={lost ? '3 3' : undefined}>
              <title>{`${labelOf(node.id)}: ${lost ? 'disconnected from the main group' : 'connected'}`}</title>
            </circle>
            <text x={x} y={y + 4} fontSize="12" fill="#d8e3ed" textAnchor="middle">{i + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function VideoConnectivityGraph({ job, onOpenComponent }) {
  const [data, setData] = useState(null);
  const [videos, setVideos] = useState(null);
  const [error, setError] = useState(null);
  const done = job?.status === 'complete';
  const stage = job?.stageKey;

  useEffect(() => {
    if (!job?.id) return undefined;
    let stale = false;
    setError(null);
    getJobVideos(job.id).then((v) => { if (!stale) setVideos(v); }).catch((e) => { if (!stale) setError(e.message); });
    getConnectivity(job.id).then((c) => { if (!stale) setData(c); }).catch(() => { if (!stale) setData(null); });
    return () => { stale = true; };
    // Refetched as the pipeline moves from stage to stage.
  }, [job?.id, done, stage]);

  if (error) return <Notice kind="error">{error}</Notice>;
  if (!videos) return <Notice>Loading video information…</Notice>;
  if (videos.mode === 'single') return <Notice>This is a single-video job. Connectivity applies when several videos are reconstructed together.</Notice>;

  const labelOf = (id) => {
    const video = videos.videos.find((v) => v.id === id);
    return video?.settings?.label || video?.originalName || id;
  };
  const connectivity = data?.connectivity;
  const report = data?.report;

  return (
    <div className="space-y-3">
      {!connectivity && <Notice>Connectivity appears after the matching stage.</Notice>}
      {connectivity && !connectivity.available && <Notice kind="warn">{connectivity.reason}</Notice>}
      {connectivity?.available && (
        <>
          <Notice kind={connectivity.fullyConnected ? 'info' : 'warn'}>
            {connectivity.fullyConnected
              ? 'All videos are connected by verified correspondences.'
              : `${connectivity.components.length} separate groups. Not connected to the main group: ${connectivity.disconnected.map(labelOf).join(', ')}.`}
          </Notice>
          <Graph connectivity={connectivity} labelOf={labelOf} />
          <div className="flex flex-wrap gap-x-3 gap-y-1 kicker normal-case tracking-normal">
            {Object.values(EDGE).map((e) => (
              <span key={e.label} className="flex items-center gap-1"><span className="w-3 h-0.5 inline-block" style={{ background: e.color }} />{e.label}</span>
            ))}
            <span>Numbers on links are verified image pairs.</span>
          </div>
          {connectivity.edges.flatMap((e) => e.warnings.map((w) => (
            <Notice key={`${e.a}${e.b}${w}`} kind="warn">{labelOf(e.a)} / {labelOf(e.b)}: {w}</Notice>
          )))}
        </>
      )}
      {data?.registration?.messages?.map((m) => <Notice key={m} kind="warn">{m}</Notice>)}
      {data?.altitude && !data.altitude.reliable && <Notice kind="warn">{data.altitude.message}</Notice>}

      <ul className="space-y-1.5">
        {videos.videos.map((video, i) => {
          const frames = video.frames;
          const registration = video.registration;
          return (
            <li key={video.id} className="panel p-2">
              <div className="flex gap-2 items-center">
                <img src={jobThumbnailUrl(job.id, video.id)} alt="" className="w-14 h-9 object-cover rounded bg-ink-700" />
                <span className="text-xs truncate">{i + 1}. {labelOf(video.id)}</span>
              </div>
              <Stats
                rows={[
                  frames && ['Selected frames', `${frames.selected} of ${frames.candidates} sampled`],
                  frames && ['Blurry / duplicate removed', `${frames.blurry} / ${frames.duplicates}`],
                  registration && ['Registered', `${registration.registeredFrames} (${registration.registrationPercent}%)`],
                  registration && ['Shared landmarks', show(registration.sharedLandmarks)],
                  !frames && ['Status', video.settings?.enabled === false ? 'Excluded' : 'Pending'],
                ]}
              />
            </li>
          );
        })}
      </ul>

      {report && (
        <div className="panel px-3 py-2">
          <p className="kicker mb-1">Job statistics</p>
          <Stats
            rows={[
              ['Videos', show(report.totalVideos)],
              ['Total duration', formatDuration(report.totalDurationSeconds)],
              ['Sampled / selected frames', `${show(report.candidateFrames)} / ${show(report.selectedFrames)}`],
              ['Blurry rejected', show(report.rejectedBlurryFrames)],
              ['Duplicates removed', show(report.removedDuplicateFrames)],
              ['Registered images', show(report.registeredImages)],
              ['Cross-video pairs', show(report.verifiedCrossVideoPairs)],
              ['Cross-video matches', show(report.verifiedCrossVideoMatches)],
              ['Connected groups', show(report.connectedComponents)],
              ['Sparse points', show(report.sparsePoints)],
              ['Dense points', show(report.densePoints)],
              ['Mesh triangles', show(report.meshTriangles)],
              ['Processing time', formatDuration(report.processingSeconds)],
            ]}
          />
        </div>
      )}
      {report?.separateComponents?.map((component) => (
        <div key={component.component} className="panel p-2 text-xs space-y-1">
          <div>Separate component {component.component}: {component.images} images ({component.videos.map(labelOf).join(', ')})</div>
          {component.model ? (
            <div className="flex gap-1">
              <button type="button" className="btn py-0.5 px-2 text-[11px]" onClick={() => onOpenComponent(component.component)}>View</button>
              <a className="btn py-0.5 px-2 text-[11px]" href={componentUrl(job.id, component.component)} download={`component-${component.component}.glb`}>Download</a>
            </div>
          ) : <p className="text-amber-ui">{component.reason}</p>}
          <p className="kicker normal-case tracking-normal">It shares no coordinate frame with the main model.</p>
        </div>
      ))}
    </div>
  );
}
