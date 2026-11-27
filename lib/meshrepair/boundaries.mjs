/**
 * Hole detection and classification.
 *
 * A boundary loop can be a reconstruction accident, a window, a courtyard, the
 * edge of the mapped area, or surface no camera ever saw. Only the first kind
 * should be closed without being asked, so each loop is measured and classified
 * conservatively: when in doubt it is left open.
 */

import { boundaryLoops, buildTopology, faceComponents } from '../meshdata.mjs';
import { faceGeometry } from '../glbmesh.mjs';
import { supported } from './state.mjs';

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const normalize = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** Best-fit plane frame of a polygon (Newell normal). */
export function planeFrame(points) {
  const centre = [0, 1, 2].map((k) => points.reduce((s, p) => s + p[k], 0) / points.length);
  const n = [0, 0, 0];
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    n[0] += (a[1] - b[1]) * (a[2] + b[2]);
    n[1] += (a[2] - b[2]) * (a[0] + b[0]);
    n[2] += (a[0] - b[0]) * (a[1] + b[1]);
  }
  const normal = normalize(n);
  const helper = Math.abs(normal[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const u = normalize(cross(helper, normal));
  const v = cross(normal, u);
  const project = (p) => { const d = sub(p, centre); return [dot(d, u), dot(d, v)]; };
  return { centre, normal, u, v, project };
}

function polygonArea(points) {
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    sum += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(sum) / 2;
}

/** Polygon area over its tightest rotated bounding rectangle: 1 for a rectangle. */
function rectangularity(flat) {
  let best = Infinity;
  for (let step = 0; step < 18; step += 1) {
    const angle = (step * Math.PI) / 36;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    let x0 = Infinity; let x1 = -Infinity; let y0 = Infinity; let y1 = -Infinity;
    for (const [x, y] of flat) {
      const rx = c * x + s * y;
      const ry = -s * x + c * y;
      x0 = Math.min(x0, rx); x1 = Math.max(x1, rx);
      y0 = Math.min(y0, ry); y1 = Math.max(y1, ry);
    }
    best = Math.min(best, (x1 - x0) * (y1 - y0));
  }
  return best > 0 ? Math.min(1, polygonArea(flat) / best) : 0;
}

function insidePolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function interiorSamples(flat, wanted = 40) {
  const xs = flat.map((p) => p[0]);
  const ys = flat.map((p) => p[1]);
  const x0 = Math.min(...xs); const x1 = Math.max(...xs);
  const y0 = Math.min(...ys); const y1 = Math.max(...ys);
  const side = Math.ceil(Math.sqrt(wanted * 2));
  const out = [];
  for (let i = 0; i < side; i += 1) {
    for (let j = 0; j < side; j += 1) {
      const x = x0 + ((x1 - x0) * (i + 0.5)) / side;
      const y = y0 + ((y1 - y0) * (j + 0.5)) / side;
      if (insidePolygon(x, y, flat)) out.push([x, y]);
    }
  }
  return out.length ? out : [[0, 0]];
}

/**
 * @param {object} options
 * @param {number} options.maxHoleSize  largest repairable diameter, mesh units
 * @returns {Array<object>} one record per loop
 */
export function analyzeHoles(mesh, { cloud, up, metresPerUnit, maxHoleSize, preserveOpenings }) {
  const topology = buildTopology(mesh);
  const { loops } = boundaryLoops(topology);
  if (!loops.length) return { topology, holes: [] };
  const components = faceComponents(topology);
  const boxes = Array.from({ length: components.count }, () => ({ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }));
  for (let f = 0; f < components.component.length; f += 1) {
    const box = boxes[components.component[f]];
    for (let k = 0; k < 3; k += 1) {
      const v = mesh.indices[f * 3 + k] * 3;
      for (let a = 0; a < 3; a += 1) {
        box.min[a] = Math.min(box.min[a], mesh.positions[v + a]);
        box.max[a] = Math.max(box.max[a], mesh.positions[v + a]);
      }
    }
  }
  const pointOf = (welded) => {
    const v = topology.welded.representative[welded] * 3;
    return [mesh.positions[v], mesh.positions[v + 1], mesh.positions[v + 2]];
  };

  const holes = loops.map((loop, id) => {
    const points = loop.vertices.map(pointOf);
    let perimeter = 0;
    for (let i = 0; i < points.length; i += 1) perimeter += Math.hypot(...sub(points[(i + 1) % points.length], points[i]));
    // Largest extent across the loop; exact for small loops, sampled for long ones.
    let diameter = 0;
    const stride = Math.max(1, Math.floor(points.length / 200));
    for (let i = 0; i < points.length; i += stride) {
      for (let j = i + 1; j < points.length; j += stride) diameter = Math.max(diameter, Math.hypot(...sub(points[i], points[j])));
    }
    const component = components.component[(loop.halves[0] / 3) | 0];
    const box = boxes[component];
    const componentSize = Math.hypot(...sub(box.max, box.min)) || 1;
    return {
      id,
      loop,
      points,
      perimeter,
      diameter,
      component,
      spansComponent: diameter / componentSize,
    };
  });

  // The rim of an open surface (the edge of the mapped area) is the longest loop
  // of its component and spans most of it.
  const rim = new Map();
  for (const hole of holes) {
    const best = rim.get(hole.component);
    if (!best || hole.perimeter > best.perimeter) rim.set(hole.component, hole);
  }

  for (const hole of holes) {
    const frame = planeFrame(hole.points);
    const flat = hole.points.map(frame.project);
    let off = 0;
    for (const p of hole.points) off += dot(sub(p, frame.centre), frame.normal) ** 2;
    hole.planarity = Math.sqrt(off / hole.points.length) / Math.max(hole.diameter, 1e-12);
    hole.rectangularity = rectangularity(flat);
    hole.areaApprox = polygonArea(flat);
    // Normal of the surrounding surface, and how much it bends around the loop.
    const around = [0, 0, 0];
    const normals = hole.loop.halves.map((h) => faceGeometry(mesh.positions, mesh.indices, (h / 3) | 0).normal);
    for (const n of normals) for (let k = 0; k < 3; k += 1) around[k] += n[k];
    const surface = normalize(around);
    hole.surfaceNormal = surface;
    hole.curvatureDegrees = normals.reduce((s, n) => s + Math.acos(Math.max(-1, Math.min(1, dot(n, surface)))), 0)
      / normals.length * (180 / Math.PI);
    // 0 for a hole in a floor or roof, 1 for a hole in a vertical wall.
    hole.verticality = 1 - Math.abs(dot(surface, up));
    hole.support = null;
    hole.behind = null;
    if (cloud) {
      const samples = interiorSamples(flat).map(([x, y]) => [0, 1, 2].map((k) => frame.centre[k] + frame.u[k] * x + frame.v[k] * y));
      const radius = Math.max(3 * cloud.spacing, hole.diameter * 0.05);
      let hits = 0;
      let through = 0;
      for (const s of samples) {
        if (supported(cloud, ...s, radius)) hits += 1;
        // Points seen further back, through the opening.
        for (let depth = 3 * radius; depth <= Math.max(3 * radius, hole.diameter * 1.5); depth += 2 * radius) {
          if (supported(cloud, s[0] - surface[0] * depth, s[1] - surface[1] * depth, s[2] - surface[2] * depth, radius)) { through += 1; break; }
        }
      }
      hole.support = hits / samples.length;
      hole.behind = through / samples.length;
    }

    const diameterM = hole.diameter * metresPerUnit;
    const unsupportedInside = hole.support != null && hole.support < 0.25;
    const windowLike = hole.verticality > 0.65 && hole.rectangularity > 0.75 && hole.planarity < 0.08
      && diameterM >= 0.4 && diameterM <= 8 && (hole.support == null || hole.support < 0.35);
    const seeThrough = unsupportedInside && hole.behind != null && hole.behind > 0.5;
    let classification;
    let reason;
    if (rim.get(hole.component) === hole && hole.spansComponent > 0.5) {
      classification = 'outer-boundary';
      reason = 'Edge of the reconstructed surface, not a hole.';
    } else if (preserveOpenings && (windowLike || seeThrough)) {
      classification = 'suspected-opening';
      reason = windowLike
        ? 'Rectangular, planar gap in a vertical surface with no points inside: likely a window, door or passage.'
        : 'Surfaces are visible through this gap, so it may be a real opening.';
    } else if (hole.diameter > maxHoleSize) {
      classification = unsupportedInside || hole.support == null ? 'large-missing' : 'large-gap';
      reason = 'Larger than the maximum repairable size.';
    } else if (hole.support != null && hole.support >= 0.3) {
      classification = 'supported-gap';
      reason = 'Within the size limit with dense points inside: the surface was observed but not meshed.';
    } else {
      classification = 'unsupported-gap';
      reason = 'Within the size limit but without dense-point support; filling it is a geometric estimate.';
    }
    hole.classification = classification;
    hole.reason = reason;
    hole.fillable = classification === 'supported-gap' || classification === 'unsupported-gap';
  }
  return { topology, holes };
}
