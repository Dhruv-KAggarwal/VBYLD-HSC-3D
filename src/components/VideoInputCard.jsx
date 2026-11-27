/** One uploaded source video: preview, facts, quality and per-video settings. */

import { useEffect, useState } from 'react';
import { uploadThumbnailUrl } from '../api.js';
import { Field, Toggle, formatBytes, formatDuration } from './ui.jsx';

const QUALITY_STYLE = { good: 'text-jet-400', fair: 'text-amber-ui', poor: 'text-red-300', unknown: 'text-mist-400' };
const CAMERA_MODELS = [
  ['SIMPLE_RADIAL', 'Simple radial (default)'],
  ['RADIAL', 'Radial'],
  ['OPENCV', 'OpenCV'],
  ['OPENCV_FISHEYE', 'Fisheye'],
  ['SIMPLE_PINHOLE', 'Pinhole'],
];

export function VideoInputCard({
  sessionId, video, index, count, disabled, frames, onSettings, onRemove, onReplace, onMove, onDragStart, onDrop,
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(video.settings);
  useEffect(() => setDraft(video.settings), [video.settings]);
  const { probe, quality } = video;
  const set = (key, value) => setDraft((d) => ({ ...d, [key]: value }));
  const commit = (changes = {}) => onSettings(video.id, { ...draft, ...changes });
  const input = (key, props = {}) => (
    <input
      className="field"
      disabled={disabled}
      value={draft[key] ?? ''}
      onChange={(e) => set(key, e.target.value)}
      onBlur={() => commit()}
      {...props}
    />
  );

  return (
    <li
      className={`panel p-2 ${draft.enabled ? '' : 'opacity-60'}`}
      draggable={!disabled && !open}
      onDragStart={() => onDragStart(video.id)}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); e.stopPropagation(); onDrop(video.id); }}
    >
      <div className="flex gap-2">
        <img
          src={uploadThumbnailUrl(sessionId, video)}
          alt=""
          className="w-20 h-12 object-cover rounded bg-ink-700 shrink-0"
        />
        <div className="min-w-0 flex-1">
          <div className="text-xs truncate" title={video.originalName}>{index + 1}. {draft.label || video.originalName}</div>
          <div className="kicker normal-case tracking-normal">
            {formatDuration(probe.durationSeconds)} · {probe.width}×{probe.height} · {probe.fps} fps · {formatBytes(video.bytes)}
          </div>
          <div className="kicker normal-case tracking-normal">
            <span className={QUALITY_STYLE[quality?.rating] || ''} title={quality?.reasons?.join(' ')}>
              Quality: {quality?.rating || 'unknown'}
            </span>
            {frames != null && draft.enabled && <span> · about {frames} frames</span>}
          </div>
        </div>
      </div>
      {quality?.reasons?.length > 0 && <p className="kicker normal-case tracking-normal mt-1">{quality.reasons.join(' ')}</p>}
      <div className="flex flex-wrap gap-1 mt-1.5">
        <button type="button" className="btn py-0.5 px-2 text-[11px]" aria-expanded={open} onClick={() => setOpen(!open)}>Settings</button>
        <button type="button" className="btn py-0.5 px-2 text-[11px]" disabled={disabled || index === 0} aria-label="Move up" onClick={() => onMove(video.id, -1)}>↑</button>
        <button type="button" className="btn py-0.5 px-2 text-[11px]" disabled={disabled || index === count - 1} aria-label="Move down" onClick={() => onMove(video.id, 1)}>↓</button>
        <label className={`btn py-0.5 px-2 text-[11px] ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
          Replace
          <input type="file" accept="video/*,.mkv,.avi,.mov,.mts,.m2ts" className="hidden" disabled={disabled} onChange={(e) => { const file = e.target.files?.[0]; e.target.value = ''; if (file) onReplace(video.id, file); }} />
        </label>
        <button type="button" className="btn btn-danger py-0.5 px-2 text-[11px]" disabled={disabled} onClick={() => onRemove(video.id)}>Remove</button>
      </div>
      {open && (
        <div className="mt-2 space-y-2 border-t border-ink-700 pt-2">
          <Toggle checked={draft.enabled} disabled={disabled} label="Include in reconstruction" onChange={(v) => { set('enabled', v); commit({ enabled: v }); }} />
          <Field label="Label">{input('label', { placeholder: video.originalName, maxLength: 80 })}</Field>
          <div className="grid grid-cols-2 gap-2">
            <Field label="Altitude (m)">{input('altitudeM', { type: 'number', min: 2, max: 2000, placeholder: 'Job value' })}</Field>
            <Field label="Frame weight">{input('weight', { type: 'number', min: 0.25, max: 4, step: 0.25 })}</Field>
            <Field label="Trim start (s)">{input('trimStart', { type: 'number', min: 0 })}</Field>
            <Field label="Trim end (s)">{input('trimEnd', { type: 'number', min: 0, placeholder: 'End' })}</Field>
          </div>
          <Field label="Camera model">
            <select className="field" disabled={disabled} value={draft.cameraModel} onChange={(e) => { set('cameraModel', e.target.value); commit({ cameraModel: e.target.value }); }}>
              {CAMERA_MODELS.map(([id, text]) => <option key={id} value={id}>{text}</option>)}
            </select>
          </Field>
          <Field label="Horizontal field of view (°)" note="Optional. Used as the starting focal length.">
            {input('horizontalFovDeg', { type: 'number', min: 10, max: 170, placeholder: 'Estimate' })}
          </Field>
          <Field label="Camera">{input('cameraName', { placeholder: [probe.metadata?.make, probe.metadata?.model].filter(Boolean).join(' ') || 'Optional' })}</Field>
          <Field label="Flight notes">{input('flightNotes', { placeholder: 'Optional', maxLength: 500 })}</Field>
          <p className="kicker normal-case tracking-normal">
            Codec {probe.codec}{probe.rotation ? ` · rotated ${probe.rotation}°` : ''}{probe.metadata?.creationTime ? ` · recorded ${probe.metadata.creationTime}` : ''}
          </p>
        </div>
      )}
    </li>
  );
}
