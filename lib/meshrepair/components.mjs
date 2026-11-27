/**
 * Floating geometry removal.
 *
 * Size alone does not make a component noise: a car, a tree or a detached shed
 * is small and separate and real. A component is removed only when it is small
 * AND there is independent reason to doubt it (little dense-point support, or it
 * hangs far from every substantial structure). The largest component is never
 * removed.
 */

import { buildTopology, faceComponents, faceCount } from '../meshdata.mjs';
import { faceGeometry } from '../glbmesh.mjs';
import { boundingDiagonal, filterFaces, supported } from './state.mjs';

export const AGGRESSIVENESS = Object.freeze({
  conservative: { faceRatio: 0.001, minFaces: 50, areaFraction: 0.0005, support: 0.35, gap: 0.02, protectSupport: 0.8 },
  balanced: { faceRatio: 0.005, minFaces: 200, areaFraction: 0.002, support: 0.5, gap: 0.01, protectSupport: 0.9 },
  aggressive: { faceRatio: 0.02, minFaces: 1000, areaFraction: 0.01, support: 0.75, gap: 0.005, protectSupport: 0.98 },
});

function boxGap(a, b) {
  let sum = 0;
  for (let k = 0; k < 3; k += 1) {
    const d = Math.max(0, a.min[k] - b.max[k], b.min[k] - a.max[k]);
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export function describeComponents(mesh, cloud) {
  const topology = buildTopology(mesh);
  const { component, count } = faceComponents(topology);
  const list = Array.from({ length: count }, (_, id) => ({
    id, faces: 0, area: 0, min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity], samples: [],
  }));
  const faces = faceCount(mesh);
  for (let f = 0; f < faces; f += 1) {
    const c = list[component[f]];
    c.faces += 1;
    c.area += faceGeometry(mesh.positions, mesh.indices, f).area;
    for (let k = 0; k < 3; k += 1) {
      const v = mesh.indices[f * 3 + k] * 3;
      for (let a = 0; a < 3; a += 1) {
        const x = mesh.positions[v + a];
        if (x < c.min[a]) c.min[a] = x;
        if (x > c.max[a]) c.max[a] = x;
      }
    }
  }
  // Up to 200 evenly spaced faces per component for the support estimate.
  const seen = new Int32Array(count);
  for (let f = 0; f < faces; f += 1) {
    const c = list[component[f]];
    const step = Math.max(1, Math.floor(c.faces / 200));
    if (seen[c.id] % step === 0) c.samples.push(f);
    seen[c.id] += 1;
  }
  for (const c of list) {
    c.support = null;
    if (cloud) {
      let hits = 0;
      for (const f of c.samples) {
        const g = faceGeometry(mesh.positions, mesh.indices, f);
        if (supported(cloud, ...g.centroid, Math.max(3 * cloud.spacing, Math.sqrt(g.area)))) hits += 1;
      }
      c.support = c.samples.length ? hits / c.samples.length : 0;
    }
    delete c.samples;
  }
  return { component, list };
}

export function removeFloatingComponents(state, { cloud, aggressiveness = 'conservative', metresPerUnit = 1 }) {
  const rules = AGGRESSIVENESS[aggressiveness] || AGGRESSIVENESS.conservative;
  const { mesh } = state;
  const { component, list } = describeComponents(mesh, cloud);
  if (list.length < 2) return { removed: [], removedComponents: [], totalComponents: list.length, rules };
  const largest = list.reduce((a, b) => (b.faces > a.faces ? b : a));
  const totalArea = list.reduce((sum, c) => sum + c.area, 0);
  const diagonal = boundingDiagonal(mesh);
  const major = list.filter((c) => c === largest || c.area >= 0.05 * largest.area);
  const doomed = new Set();
  const removedComponents = [];
  for (const c of list) {
    if (c === largest) continue;
    const small = c.faces < Math.max(rules.minFaces, rules.faceRatio * largest.faces)
      && c.area < rules.areaFraction * totalArea;
    if (!small) continue;
    const gap = Math.min(...major.filter((m) => m !== c).map((m) => boxGap(c, m)));
    const far = gap > rules.gap * diagonal;
    const weak = c.support != null && c.support < rules.support;
    if (!far && !weak) continue;
    // Well-supported structure of real size stays, however isolated it is.
    if (c.support != null && c.support >= rules.protectSupport && c.area * metresPerUnit ** 2 > 4) continue;
    doomed.add(c.id);
    removedComponents.push({
      faces: c.faces,
      areaM2: Math.round(c.area * metresPerUnit ** 2 * 1000) / 1000,
      sizeM: Math.round(Math.hypot(c.max[0] - c.min[0], c.max[1] - c.min[1], c.max[2] - c.min[2]) * metresPerUnit * 100) / 100,
      distanceToStructureM: Math.round(gap * metresPerUnit * 100) / 100,
      pointSupport: c.support == null ? null : Math.round(c.support * 100) / 100,
      reasons: [far ? 'far from every major structure' : null, weak ? 'little dense-point support' : null].filter(Boolean),
    });
  }
  const keep = new Uint8Array(faceCount(mesh));
  for (let f = 0; f < keep.length; f += 1) keep[f] = doomed.has(component[f]) ? 0 : 1;
  const removed = filterFaces(state, keep);
  removedComponents.sort((a, b) => b.faces - a.faces);
  return { removed, removedComponents, totalComponents: list.length, rules };
}
