import { useCallback, useEffect, useState } from 'react';
import { fileUrl, fileUrlFresh, generateTurntable, hasFile } from './api.js';

/**
 * The turntable panel: a 360-degree MP4 flyaround of the finished model.
 * ---------------------------------------------------------------------------
 * The interactive viewer is the better tool for inspecting a reconstruction,
 * but it needs WebGL, a GPU and a user who knows how to orbit. This is the
 * answer to "did it work?" that plays anywhere and can be dropped into a chat
 * or a slide as-is.
 *
 * Three states, all of which happen in practice:
 *   - the job has a video (rendered at the end of the pipeline)
 *   - the job has none, because it predates the stage or the render failed —
 *     offer to render one now
 *   - the render is running, which takes tens of seconds of CPU
 */
export function TurntablePlayer({ jobId, hasVideo, onNotify, className = '' }) {
  const [state, setState] = useState('probing');
  const [token, setToken] = useState(() => Date.now());
  const [error, setError] = useState(null);

  // Probe rather than trust the job object: `result.video` is null on every job
  // recovered from disk that predates this stage, and the file may equally
  // have been rendered on demand in a previous session.
  useEffect(() => {
    let cancelled = false;
    if (!jobId) { setState('idle'); return undefined; }
    setState('probing');
    setError(null);
    (async () => {
      const present = hasVideo || await hasFile(jobId, 'video');
      if (cancelled) return;
      setToken(Date.now());
      setState(present ? 'ready' : 'missing');
    })();
    return () => { cancelled = true; };
  }, [jobId, hasVideo]);

  const render = useCallback(async () => {
    if (!jobId) return;
    setState('rendering');
    setError(null);
    try {
      const outcome = await generateTurntable(jobId);
      setToken(Date.now());
      setState('ready');
      onNotify?.({
        kind: 'success',
        message: `Turntable rendered: ${outcome.frames} frames, ${(outcome.bytes / 1e6).toFixed(1)} MB`,
      });
    } catch (failure) {
      setError(failure.message);
      setState('missing');
      onNotify?.({ kind: 'error', message: failure.message });
    }
  }, [jobId, onNotify]);

  if (!jobId) {
    return (
      <Shell className={className}>
        <p className="text-sm text-mist-400">Reconstruct a scene to get a turntable.</p>
      </Shell>
    );
  }

  if (state === 'probing') {
    return (
      <Shell className={className}>
        <p className="kicker animate-pulse">Looking for the turntable…</p>
      </Shell>
    );
  }

  if (state === 'rendering') {
    return (
      <Shell className={className}>
        <div className="text-center">
          <div className="text-3xl text-jet-400 animate-pulse">◈</div>
          <p className="kicker mt-3">Rendering the turntable</p>
          <p className="text-xs text-mist-400 mt-1 max-w-xs">
            Rasterising the mesh frame by frame on the CPU. This takes under a minute for a
            typical model; the page can be left alone.
          </p>
        </div>
      </Shell>
    );
  }

  if (state === 'missing') {
    return (
      <Shell className={className}>
        <div className="text-center max-w-sm">
          <div className="text-3xl text-ink-600 mb-2">▦</div>
          <p className="text-sm">No turntable for this reconstruction.</p>
          <p className="kicker normal-case tracking-normal mt-1 text-mist-400">
            Jobs run before this feature existed do not have one. Rendering it does not touch
            the model.
          </p>
          {error && <p className="text-xs text-red-300 mt-2">{error}</p>}
          <button className="btn btn-primary mt-3" onClick={render}>Generate turntable</button>
        </div>
      </Shell>
    );
  }

  return (
    <div className={`relative h-full w-full bg-ink-950 ${className}`}>
      <video
        key={token}
        src={fileUrlFresh(jobId, 'video', token)}
        poster={fileUrlFresh(jobId, 'poster', token)}
        controls
        loop
        muted
        autoPlay
        playsInline
        preload="metadata"
        className="h-full w-full object-contain bg-black"
        onError={() => {
          // The probe said the file was there, so this is a decode or transport
          // failure rather than a missing file. Offering the render button is
          // the only useful next step either way.
          setError('The browser could not play this file. Re-rendering it may help.');
          setState('missing');
        }}
      />
      <div className="absolute top-3 right-3 flex gap-2">
        <a
          href={fileUrl(jobId, 'video')}
          download="turning.mp4"
          className="btn py-1 px-2.5 text-xs"
        >↓ MP4</a>
        <button className="btn py-1 px-2.5 text-xs" onClick={render} title="Render it again">
          ⟳ Re-render
        </button>
      </div>
    </div>
  );
}

function Shell({ children, className = '' }) {
  return (
    <div className={`h-full w-full grid place-items-center bg-ink-950 p-6 ${className}`}>
      {children}
    </div>
  );
}

export default TurntablePlayer;
