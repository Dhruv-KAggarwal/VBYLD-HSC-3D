/** Heatmap legend and display filters. */

import { PALETTE_COLORS } from '../viewer/overlay-renderer.js';
import { Field, Segmented, Toggle } from './ui.jsx';

const BANDS = [
  [4, 'High', '80–100'], [3, 'Good', '60–79'], [2, 'Moderate', '40–59'],
  [1, 'Low', '20–39'], [0, 'Very low', '0–19'], [5, 'Unknown', 'insufficient evidence'],
];

export function ConfidenceLegend({ overlay, onOverlay, wireframe, onWireframe }) {
  const on = overlay.mode === 'confidence';
  const colors = PALETTE_COLORS[overlay.palette];
  const set = (changes) => onOverlay({ mode: 'confidence', ...changes });
  return (
    <div className="panel p-3 space-y-2.5">
      <Toggle checked={on} label="Confidence heatmap" note="Drawn over the model; the original texture is unchanged." onChange={(v) => onOverlay({ mode: v ? 'confidence' : 'off' })} />
      <ul className="space-y-1" aria-label="Confidence colour legend">
        {BANDS.map(([index, label, range]) => (
          <li key={label} className="flex items-center gap-2 text-xs">
            <span className="w-3 h-3 rounded-sm" style={{ background: colors[index] }} />
            <span className="flex-1">{label}</span>
            <span className="font-mono text-mist-400">{range}</span>
          </li>
        ))}
      </ul>
      <Segmented
        label="Colour palette" value={overlay.palette} onChange={(palette) => onOverlay({ palette })}
        options={[['standard', 'Standard'], ['accessible', 'Colour-vision safe']]}
      />
      <Field label={`Overlay opacity ${Math.round(overlay.opacity * 100)}%`}>
        <input type="range" min="0.1" max="1" step="0.05" className="w-full" value={overlay.opacity} onChange={(e) => onOverlay({ opacity: Number(e.target.value) })} />
      </Field>
      <div className="grid grid-cols-2 gap-2">
        <Field label="Show from">
          <input type="number" min="0" max="100" className="field" value={overlay.min} onChange={(e) => set({ min: Math.min(Number(e.target.value), overlay.max) })} />
        </Field>
        <Field label="Show to">
          <input type="number" min="0" max="100" className="field" value={overlay.max} onChange={(e) => set({ max: Math.max(Number(e.target.value), overlay.min) })} />
        </Field>
      </div>
      <Field label={`Low-confidence threshold: below ${overlay.threshold}`}>
        <input type="range" min="5" max="80" step="5" className="w-full" value={overlay.threshold} onChange={(e) => onOverlay({ threshold: Number(e.target.value) })} />
      </Field>
      <Toggle checked={overlay.onlyLow} label="Only low-confidence regions" onChange={(onlyLow) => set({ onlyLow, onlyUnknown: false })} />
      <Toggle checked={overlay.onlyUnknown} label="Only unknown regions" onChange={(onlyUnknown) => set({ onlyUnknown, onlyLow: false, showUnknown: true })} />
      <Toggle checked={overlay.onlyRepaired} label="Only repaired regions" onChange={(onlyRepaired) => set({ onlyRepaired })} />
      <Toggle checked={overlay.showUnknown} label="Show unknown (grey)" onChange={(showUnknown) => set({ showUnknown })} />
      <Toggle checked={overlay.isolate} label="Hide the model beneath" onChange={(isolate) => set({ isolate })} />
      <Toggle checked={wireframe} label="Wireframe" onChange={onWireframe} />
      <button type="button" className="btn py-1 px-2 text-xs" onClick={() => onOverlay({ mode: on ? 'off' : 'confidence' })}>
        {on ? 'Compare with original texture' : 'Back to heatmap'}
      </button>
    </div>
  );
}
