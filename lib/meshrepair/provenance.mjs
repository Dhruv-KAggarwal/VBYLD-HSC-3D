/**
 * Where each face of a repaired mesh came from.
 *
 * Codes are ordered by how far the face is from photographic evidence, so
 * "worse" is simply "larger" and a face can only ever move up the scale.
 *
 * Binary sidecar (little-endian), indexed by face in the written GLB's order:
 *   bytes 0-3   magic "SFPV"
 *   bytes 4-5   uint16 version (1)
 *   bytes 6-7   uint16 reserved
 *   bytes 8-11  uint32 faceCount
 *   then        uint8  code[faceCount], zero-padded to a 4-byte boundary
 *   then        int32  sourceFace[faceCount]  (face of the original mesh in
 *                       ITS order, or -1 for generated geometry)
 */

import { readFile, writeFile } from 'node:fs/promises';

export const PROVENANCE = Object.freeze({
  OBSERVED: 0,
  CLEANED: 1,
  SMOOTHED: 2,
  INTERPOLATED: 3,
  HOLE_FILLED: 4,
  UNSUPPORTED: 5,
});

export const PROVENANCE_INFO = Object.freeze([
  { code: 0, key: 'observed', label: 'Original geometry', color: null, description: 'Unmodified reconstruction output.' },
  { code: 1, key: 'cleaned', label: 'Cleaned', color: '#3b82f6', description: 'Vertex welding or face orientation changed; shape unchanged.' },
  { code: 2, key: 'smoothed', label: 'Smoothed', color: '#22d3ee', description: 'Vertices moved by noise smoothing.' },
  { code: 3, key: 'interpolated', label: 'Interpolated', color: '#facc15', description: 'Generated across a small gap that dense points support.' },
  { code: 4, key: 'holeFilled', label: 'Hole filled', color: '#fb923c', description: 'Generated to close a hole without dense-point support.' },
  { code: 5, key: 'unsupported', label: 'Unsupported / estimated', color: '#ef4444', description: 'No dense-point evidence lies near this surface.' },
]);

export const isGenerated = (code) => code === 3 || code === 4;

export function encodeProvenance(codes, source) {
  const count = codes.length;
  const padded = Math.ceil(count / 4) * 4;
  const out = Buffer.alloc(12 + padded + count * 4);
  out.write('SFPV', 0, 'ascii');
  out.writeUInt16LE(1, 4);
  out.writeUInt32LE(count, 8);
  out.set(codes, 12);
  for (let i = 0; i < count; i += 1) out.writeInt32LE(source[i], 12 + padded + i * 4);
  return out;
}

export function decodeProvenance(buffer) {
  if (buffer.toString('ascii', 0, 4) !== 'SFPV') throw new Error('Not a provenance file.');
  const count = buffer.readUInt32LE(8);
  const padded = Math.ceil(count / 4) * 4;
  const codes = Uint8Array.from(buffer.subarray(12, 12 + count));
  const source = new Int32Array(count);
  for (let i = 0; i < count; i += 1) source[i] = buffer.readInt32LE(12 + padded + i * 4);
  return { count, codes, source };
}

export async function writeProvenance(path, codes, source) {
  await writeFile(path, encodeProvenance(codes, source));
}

export async function readProvenance(path) {
  return decodeProvenance(await readFile(path));
}

/** Area share of each category, as percentages. */
export function summarizeProvenance(codes, areas) {
  const totals = new Array(PROVENANCE_INFO.length).fill(0);
  const faces = new Array(PROVENANCE_INFO.length).fill(0);
  let total = 0;
  for (let f = 0; f < codes.length; f += 1) {
    totals[codes[f]] += areas[f];
    faces[codes[f]] += 1;
    total += areas[f];
  }
  return PROVENANCE_INFO.map((info) => ({
    code: info.code,
    key: info.key,
    label: info.label,
    faces: faces[info.code],
    areaPercent: total ? Math.round((totals[info.code] / total) * 10000) / 100 : 0,
  }));
}
