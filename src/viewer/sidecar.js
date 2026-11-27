/**
 * Browser readers for the per-face binary sidecars written by the pipeline
 * (schemas in lib/confidence/scoring.mjs and lib/meshrepair/provenance.mjs).
 */

export const UNKNOWN = 255;

const magic = (view) => String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));

export function parseConfidence(buffer) {
  const view = new DataView(buffer);
  if (magic(view) !== 'SFCF') throw new Error('Not a confidence data file.');
  const size = view.getUint16(6, true);
  const count = view.getUint32(8, true);
  const masks = view.getUint32(12, true) & 1 ? 16 + count * size : null;
  const optional = (value) => (value === UNKNOWN ? null : value);
  const at = (face) => 16 + face * size;
  return {
    count,
    score: (face) => view.getUint8(at(face)),
    provenance: (face) => view.getUint8(at(face) + 13),
    read(face) {
      if (face < 0 || face >= count) return null;
      const o = at(face);
      const cameras = view.getUint16(o + 8, true);
      const error = view.getUint16(o + 10, true);
      const score = view.getUint8(o);
      return {
        face,
        score: score === UNKNOWN ? null : score,
        availability: view.getUint8(o + 1),
        coverage: optional(view.getUint8(o + 2)),
        diversity: optional(view.getUint8(o + 3)),
        reprojection: optional(view.getUint8(o + 4)),
        support: optional(view.getUint8(o + 5)),
        quality: optional(view.getUint8(o + 6)),
        depth: optional(view.getUint8(o + 7)),
        cameras: cameras === 65535 ? null : cameras,
        reprojectionError: error === 65535 ? null : error / 100,
        videos: view.getUint8(o + 12),
        provenance: view.getUint8(o + 13),
        incidence: optional(view.getUint8(o + 14)),
        spread: optional(view.getUint8(o + 15)),
        videoMask: masks == null ? 0 : view.getUint32(masks + face * 4, true),
      };
    },
  };
}

export function parseProvenance(buffer) {
  const view = new DataView(buffer);
  if (magic(view) !== 'SFPV') throw new Error('Not a provenance data file.');
  const count = view.getUint32(8, true);
  return { count, codes: new Uint8Array(buffer, 12, count) };
}
