/**
 * Staged upload of several videos into one session, with per-file progress,
 * reorder, replace and remove, and an estimate of what the job will need.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  createUpload, deleteUpload, deleteUploadVideo, getUpload, updateUpload, uploadVideo,
} from '../api.js';
import { VideoInputCard } from './VideoInputCard.jsx';
import { Notice, Stats, formatBytes, formatDuration } from './ui.jsx';

const STORAGE_KEY = 'skyforge.uploadSession';
const PARALLEL = 2;
const remember = (id) => { try { if (id) localStorage.setItem(STORAGE_KEY, id); else localStorage.removeItem(STORAGE_KEY); } catch { /* storage is optional */ } };
const recall = () => { try { return localStorage.getItem(STORAGE_KEY); } catch { return null; } };

export function MultiVideoUpload({ session, onSession, onPending, quality, frames, disabled }) {
  const [uploads, setUploads] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const sessionRef = useRef(session);
  const creating = useRef(null);
  const queue = useRef([]);
  const running = useRef(0);
  const dragged = useRef(null);
  const options = useRef({});
  options.current = { quality, frames };
  sessionRef.current = session;

  useEffect(() => { onPending(uploads.filter((u) => !u.error).length); }, [uploads, onPending]);

  // Restore an unfinished session, and refresh the estimate when options change.
  useEffect(() => {
    const id = sessionRef.current?.id || recall();
    if (!id) return undefined;
    let stale = false;
    getUpload(id, { quality, frames })
      .then((next) => { if (!stale) { sessionRef.current = next; onSession(next); } })
      .catch((failure) => {
        if (stale || failure.status !== 404) return;
        remember(null);
        onSession(null);
      });
    return () => { stale = true; };
  }, [quality, frames, onSession]);

  const apply = useCallback((next) => {
    sessionRef.current = next;
    remember(next?.id);
    onSession(next);
  }, [onSession]);

  const ensureSession = useCallback(async () => {
    if (sessionRef.current?.id) return sessionRef.current;
    creating.current ||= createUpload(options.current).then((next) => { apply(next); return next; })
      .finally(() => { creating.current = null; });
    return creating.current;
  }, [apply]);

  const patch = (key, changes) => setUploads((list) => list.map((u) => (u.key === key ? { ...u, ...changes } : u)));

  const pump = useCallback(() => {
    while (running.current < PARALLEL && queue.current.length) {
      const item = queue.current.shift();
      running.current += 1;
      (async () => {
        try {
          const target = await ensureSession();
          if (item.cancelled) return;
          const transfer = uploadVideo(target.id, item.file, {
            replace: item.replace,
            onProgress: (progress) => patch(item.key, { progress, status: progress >= 1 ? 'Validating' : 'Uploading' }),
          });
          item.abort = transfer.abort;
          await transfer.promise;
          setUploads((list) => list.filter((u) => u.key !== item.key));
          apply(await getUpload(target.id, options.current));
        } catch (failure) {
          if (failure.aborted) setUploads((list) => list.filter((u) => u.key !== item.key));
          else patch(item.key, { error: failure.message });
        } finally {
          running.current -= 1;
          pump();
        }
      })();
    }
  }, [apply, ensureSession]);

  const add = useCallback((files, replace = null) => {
    setError(null);
    for (const file of files) {
      const item = { key: `${file.name}-${file.size}-${Math.random()}`, file, replace, cancelled: false, abort: null };
      queue.current.push(item);
      setUploads((list) => [...list, { key: item.key, name: file.name, size: file.size, progress: 0, status: 'Waiting', error: null, item }]);
    }
    pump();
  }, [pump]);

  const cancel = (upload) => {
    upload.item.cancelled = true;
    queue.current = queue.current.filter((i) => i !== upload.item);
    if (upload.item.abort) upload.item.abort();
    setUploads((list) => list.filter((u) => u.key !== upload.key));
  };

  const act = async (operation) => {
    setError(null);
    try { apply(await operation()); } catch (failure) { setError(failure.message); }
  };

  const videos = session?.videos || [];
  const move = (id, step) => {
    const order = videos.map((v) => v.id);
    const from = order.indexOf(id);
    const to = from + step;
    if (to < 0 || to >= order.length) return;
    order.splice(to, 0, ...order.splice(from, 1));
    act(() => updateUpload(session.id, { order }, options.current));
  };
  const drop = (target) => {
    const source = dragged.current;
    dragged.current = null;
    if (!source || source === target) return;
    move(source, videos.findIndex((v) => v.id === target) - videos.findIndex((v) => v.id === source));
  };

  const clearAll = async () => {
    uploads.forEach(cancel);
    if (session?.id) await deleteUpload(session.id).catch(() => {});
    apply(null);
  };

  const estimate = session?.estimate;

  return (
    <div className="space-y-2">
      <div
        onDragOver={(e) => { if (e.dataTransfer.types.includes('Files')) { e.preventDefault(); setDragging(true); } }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); if (!disabled) add([...e.dataTransfer.files]); }}
        className={`panel p-4 text-center transition-colors ${dragging ? 'border-jet-500 bg-jet-500/5' : ''}`}
      >
        <div className="text-sm mb-1">Drop several videos of one place</div>
        <div className="kicker mb-3 normal-case tracking-normal">MP4, MOV, AVI, MKV and anything else FFmpeg decodes. Flights should overlap.</div>
        <label className={`btn cursor-pointer ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
          Add videos
          <input
            type="file" multiple accept="video/*,.mkv,.avi,.mov,.mts,.m2ts" className="hidden" disabled={disabled}
            onChange={(e) => { add([...e.target.files]); e.target.value = ''; }}
          />
        </label>
      </div>

      {error && <Notice kind="error">{error}</Notice>}

      {uploads.map((upload) => (
        <div key={upload.key} className="panel p-2">
          <div className="flex justify-between gap-2 text-xs">
            <span className="truncate">{upload.name} <span className="text-mist-400">({formatBytes(upload.size)})</span></span>
            <button type="button" className="btn py-0 px-2 text-[11px]" onClick={() => cancel(upload)}>{upload.error ? 'Dismiss' : 'Cancel'}</button>
          </div>
          {upload.error ? <p className="text-xs text-red-300 mt-1" role="alert">{upload.error}</p> : (
            <>
              <div className="h-1 mt-1.5 rounded-full bg-ink-800 overflow-hidden" role="progressbar" aria-valuenow={Math.round(upload.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
                <div className="h-full bg-jet-500" style={{ width: `${upload.progress * 100}%` }} />
              </div>
              <div className="kicker mt-1">{upload.status} {Math.round(upload.progress * 100)}%</div>
            </>
          )}
        </div>
      ))}

      {videos.length > 0 && (
        <>
          <ul className="space-y-1.5">
            {videos.map((video, index) => (
              <VideoInputCard
                key={video.id}
                sessionId={session.id}
                video={video}
                index={index}
                count={videos.length}
                disabled={disabled}
                frames={estimate?.framesPerVideo?.[video.id]}
                onSettings={(id, settings) => act(() => updateUpload(session.id, { settings: { [id]: settings } }, options.current))}
                onRemove={(id) => act(() => deleteUploadVideo(session.id, id, options.current))}
                onReplace={(id, file) => add([file], id)}
                onMove={move}
                onDragStart={(id) => { dragged.current = id; }}
                onDrop={drop}
              />
            ))}
          </ul>
          {estimate && (
            <div className="panel px-3 py-2">
              <p className="kicker mb-1">Estimated requirements</p>
              <Stats
                rows={[
                  ['Videos included', `${estimate.videos} of ${videos.length}`],
                  ['Total duration', formatDuration(estimate.totalDurationSeconds)],
                  ['Total input size', formatBytes(estimate.totalBytes)],
                  estimate.estimatedFrames != null && ['Frames to extract', `~${estimate.estimatedFrames}`],
                  estimate.estimatedIntraPairs != null && ['Pairs within videos', `~${estimate.estimatedIntraPairs.toLocaleString()}`],
                  estimate.estimatedCrossPairs != null && ['Pairs between videos', `~${estimate.estimatedCrossPairs.toLocaleString()}`],
                  estimate.estimatedWorkingBytes && ['Working disk space', `~${formatBytes(estimate.estimatedWorkingBytes)}`],
                ]}
              />
              {estimate.warnings?.map((w) => <p key={w} className="text-xs text-amber-ui mt-1">{w}</p>)}
            </div>
          )}
          <button type="button" className="btn py-1 px-2 text-xs" disabled={disabled} onClick={clearAll}>Remove all</button>
        </>
      )}
    </div>
  );
}
