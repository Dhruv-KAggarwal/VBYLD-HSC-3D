/**
 * Browser fullscreen for the scene area.
 *
 * Fly and walk use Esc to hand the mouse back, and Esc is also how a browser
 * leaves fullscreen. Where the Keyboard Lock API exists (Chrome, Edge) the page
 * gets Esc first, so one press returns to orbit and stays fullscreen, and a
 * second press leaves. Elsewhere Esc leaves both; the number keys still switch
 * mode without leaving.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export function useFullscreen(viewerRef) {
  const ref = useRef(null);
  const [active, setActive] = useState(false);

  const toggle = useCallback(async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await ref.current?.requestFullscreen();
    } catch { /* refused by the browser; the windowed view carries on */ }
  }, []);

  useEffect(() => {
    const onChange = () => {
      const on = Boolean(ref.current) && document.fullscreenElement === ref.current;
      setActive(on);
      if (on) navigator.keyboard?.lock?.(['Escape'])?.catch(() => {});
      else navigator.keyboard?.unlock?.();
    };
    // Capture phase: this has to read the navigation mode before the viewer's
    // own Esc handler switches it back to orbit.
    const onKey = (event) => {
      if (event.code !== 'Escape' || event.repeat || !document.fullscreenElement) return;
      if (viewerRef.current?.navMode === 'orbit') document.exitFullscreen().catch(() => {});
    };
    document.addEventListener('fullscreenchange', onChange);
    window.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('fullscreenchange', onChange);
      window.removeEventListener('keydown', onKey, true);
    };
  }, [viewerRef]);

  return { ref, active, toggle };
}
