/** Small shared controls and formatters for the console. */

import { useState } from 'react';

export function formatBytes(bytes) {
  if (!bytes) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let index = 0;
  let value = bytes;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`;
}

/** A number from the backend, or an honest "Unavailable". */
export function show(value, suffix = '', digits = null) {
  if (value === null || value === undefined || value === '' || Number.isNaN(value)) return 'Unavailable';
  if (typeof value !== 'number') return `${value}${suffix}`;
  return `${value.toLocaleString(undefined, digits == null ? {} : { maximumFractionDigits: digits })}${suffix}`;
}

export function Toggle({ checked, onChange, label, note, disabled }) {
  return (
    <label className={`flex gap-2.5 items-start ${disabled ? 'opacity-50' : 'cursor-pointer'}`}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 accent-[var(--color-jet-500)] w-4 h-4 shrink-0"
      />
      <span className="min-w-0">
        <span className="block text-xs">{label}</span>
        {note && <span className="kicker normal-case tracking-normal block mt-0.5 text-mist-400">{note}</span>}
      </span>
    </label>
  );
}

export function Segmented({ options, value, onChange, label, disabled }) {
  return (
    <div className="flex flex-wrap gap-1" role="radiogroup" aria-label={label}>
      {options.map(([id, text]) => (
        <button
          key={id}
          type="button"
          role="radio"
          aria-checked={value === id}
          disabled={disabled}
          className={`btn py-1 px-2 text-xs ${value === id ? 'btn-toggle-on' : ''}`}
          onClick={() => onChange(id)}
        >{text}</button>
      ))}
    </div>
  );
}

export function Field({ label, note, children }) {
  return (
    <label className="block">
      <span className="kicker block mb-1">{label}</span>
      {children}
      {note && <span className="kicker normal-case tracking-normal block mt-1">{note}</span>}
    </label>
  );
}

export function Collapsible({ title, children, open: initially = false }) {
  const [open, setOpen] = useState(initially);
  return (
    <div className="panel">
      <button type="button" className="w-full flex justify-between items-center px-3 py-2 text-xs cursor-pointer" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span>{title}</span>
        <span aria-hidden="true" className="text-mist-400">{open ? '−' : '+'}</span>
      </button>
      {open && <div className="px-3 pb-3 space-y-2.5">{children}</div>}
    </div>
  );
}

export function Stats({ rows }) {
  return (
    <dl className="text-xs divide-y divide-ink-700">
      {rows.filter(Boolean).map(([label, value]) => (
        <div key={label} className="flex justify-between gap-3 py-1">
          <dt className="text-mist-400">{label}</dt>
          <dd className="font-mono text-right break-words min-w-0">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Notice({ kind = 'info', children }) {
  const styles = {
    info: 'text-mist-300',
    warn: 'border-amber-900/50 bg-amber-950/20 text-amber-ui',
    error: 'border-red-900/60 bg-red-950/30 text-red-200',
  };
  return <div className={`panel px-3 py-2 text-xs ${styles[kind]}`} role={kind === 'error' ? 'alert' : 'status'}>{children}</div>;
}

export function TaskProgress({ task, onCancel }) {
  if (!task) return null;
  const running = task.status === 'queued' || task.status === 'running';
  if (task.status === 'error') return <Notice kind="error">{task.operation}: {task.error}</Notice>;
  if (!running) return task.summary ? <Notice>{task.summary}</Notice> : <Notice>{task.operation}: {task.status}</Notice>;
  return (
    <div className="panel px-3 py-2 space-y-1.5" role="status">
      <div className="flex justify-between text-xs">
        <span>{task.operation}: {task.message}</span>
        <span className="font-mono text-jet-400">{Math.round(task.progress)}%</span>
      </div>
      <div className="h-1 rounded-full bg-ink-800 overflow-hidden">
        <div className="h-full bg-jet-500" style={{ width: `${task.progress}%` }} />
      </div>
      <button type="button" className="btn btn-danger py-0.5 px-2 text-[11px]" onClick={onCancel}>Cancel</button>
    </div>
  );
}
