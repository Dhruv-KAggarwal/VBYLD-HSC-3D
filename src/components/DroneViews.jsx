/** Jump between the drone's own camera positions, and frame the filmed area. */

import { useState } from 'react';

export function DroneViews({ viewer, views }) {
  const [index, setIndex] = useState(views.start);
  const [focus, setFocus] = useState(true);
  const count = views.views.length;

  const go = (next) => {
    const value = (next + count) % count;
    setIndex(value);
    viewer?.showDroneView(value);
  };

  return (
    <div className="absolute bottom-14 left-3 panel flex flex-wrap items-center gap-1.5 p-1.5 z-10 max-w-[calc(100%-11rem)]" aria-label="Drone camera views">
      <button type="button" className="btn py-1 px-2 text-xs" onClick={() => go(index)} title="Look through the drone camera">Drone view</button>
      <button type="button" className="btn py-1 px-2 text-xs" aria-label="Previous camera position" onClick={() => go(index - 1)}>‹</button>
      <input
        type="range" min="0" max={count - 1} value={index} className="w-32"
        aria-label="Camera position along the flight"
        onChange={(e) => go(Number(e.target.value))}
      />
      <button type="button" className="btn py-1 px-2 text-xs" aria-label="Next camera position" onClick={() => go(index + 1)}>›</button>
      <span className="kicker">{index + 1}/{count}</span>
      <button type="button" className="btn py-1 px-2 text-xs" onClick={() => viewer?.showOverview()}>Overview</button>
      <button
        type="button"
        className={`btn py-1 px-2 text-xs ${focus ? 'btn-toggle-on' : ''}`}
        aria-pressed={focus}
        title="Hide the thinly observed surroundings. Nothing is deleted."
        onClick={() => { setFocus(!focus); viewer?.setFocusClip(!focus); }}
      >Filmed area only</button>
    </div>
  );
}
