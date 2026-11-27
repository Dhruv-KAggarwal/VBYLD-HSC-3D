/** Live figures for the displayed level of detail. */

import { useEffect, useState } from 'react';
import { Stats, formatBytes, show } from './ui.jsx';

export function PerformanceMonitor({ viewer, lod }) {
  const [live, setLive] = useState({ fps: null, frameMs: null });
  useEffect(() => {
    const timer = setInterval(() => {
      if (!viewer) return;
      const shown = viewer.lod?.level(viewer.lod.current);
      setLive({ ...viewer.performanceStats(), screenError: shown ? viewer.lod.screenError(shown) : null });
    }, 500);
    return () => clearInterval(timer);
  }, [viewer]);

  const level = lod?.level;
  const original = lod?.original;
  return (
    <div className="panel px-3 py-2">
      <p className="kicker mb-1">Performance</p>
      <Stats
        rows={[
          lod && ['Active level', lod.current == null ? 'Loading' : `LOD ${lod.current}${lod.loading != null ? ` (loading ${lod.loading})` : ''}`],
          original && ['Original triangles', show(original.triangleCount)],
          level && ['Current triangles', show(level.triangleCount)],
          level && original && ['Geometry reduction', show((1 - level.triangleCount / original.triangleCount) * 100, '%', 1)],
          level && ['Texture resolution', level.textureResolution ? `${level.textureResolution} px` : 'Vertex colour'],
          level && ['Model file size', formatBytes(level.fileSizeBytes)],
          level && ['Compression ratio', show(level.compressionRatio, '×')],
          level && ['Estimated GPU memory', formatBytes((level.geometryGpuBytes || 0) + (level.textureGpuBytes || 0))],
          level && ['Geometric error', level.level === 0 ? '0 (original)' : show(level.geometricError, ' m', 3)],
          level && ['Error on screen', show(live.screenError, ' px', 2)],
          lod && ['Model cache', `${formatBytes(lod.cacheBytes)} of ${formatBytes(lod.cacheBudget)}`],
          ['Frames per second', live.fps == null ? 'Move the view' : show(live.fps, '', 0)],
          ['Frame render time', show(live.frameMs, ' ms', 1)],
        ]}
      />
    </div>
  );
}
