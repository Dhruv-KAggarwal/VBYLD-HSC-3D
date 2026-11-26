// buildings.mjs — extract distinct, extruded building blocks (LOD1) from the DSM.
//
// Why this exists: a single forward aerial pass reconstructs the city as one
// smooth 2.5D "draped" surface, which reads like a landscape photo rather than a
// 3D model. This module turns that surface into UNMISTAKABLY 3D geometry: it
// estimates a bare-earth ground model (DTM) from the digital-surface model (DSM),
// measures height-above-ground, segments the raised regions into separate
// footprints, and extrudes each one into a flat-roofed solid with real vertical
// walls. The result is a CityGML-LOD1-style block model that stands apart from the
// terrain — distinct buildings you can orbit, not a painted carpet.
//
// It operates entirely in the terrain's aligned, UNSCALED grid frame (the same
// frame buildTerrain uses), so the caller applies the single metric `scale` to the
// returned positions and the blocks line up exactly with terrain.glb / model.glb.

// Median of a numeric array (ignores NaN). Returns NaN for an empty set.
function median(values) {
  const clean = values.filter((v) => Number.isFinite(v));
  if (!clean.length) return Number.NaN;
  clean.sort((a, b) => a - b);
  return clean[clean.length >> 1];
}

// Percentile (0..1) of a numeric array, NaN-safe.
function percentile(values, p) {
  const clean = values.filter((v) => Number.isFinite(v));
  if (!clean.length) return Number.NaN;
  clean.sort((a, b) => a - b);
  const idx = Math.min(clean.length - 1, Math.max(0, Math.round(p * (clean.length - 1))));
  return clean[idx];
}

/**
 * Approximate a bare-earth DTM from the DSM heightfield.
 *
 * Buildings are local maxima sitting on the ground, so a low percentile taken over
 * a window WIDER than the biggest building recovers street level; bilinear upsampling
 * of those coarse ground anchors gives a smooth terrain datum. Cheap (O(cells)) and
 * robust to the NaN holes in the grid.
 */
function estimateGround(height, mask, cols, rows, block, p = 0.12) {
  const bc = Math.max(1, Math.ceil(cols / block));
  const br = Math.max(1, Math.ceil(rows / block));
  const coarse = new Float32Array(bc * br);
  for (let by = 0; by < br; by += 1) {
    for (let bx = 0; bx < bc; bx += 1) {
      const samples = [];
      const x0 = bx * block;
      const y0 = by * block;
      for (let y = y0; y < Math.min(rows, y0 + block); y += 1) {
        for (let x = x0; x < Math.min(cols, x0 + block); x += 1) {
          const i = y * cols + x;
          if (mask[i] && Number.isFinite(height[i])) samples.push(height[i]);
        }
      }
      // Low percentile = street level even when a block is mostly rooftops.
      coarse[by * bc + bx] = percentile(samples, p);
    }
  }
  // Fill empty coarse cells from their neighbours so bilinear upsampling is defined
  // everywhere (a few relaxation sweeps).
  for (let pass = 0; pass < 4; pass += 1) {
    for (let k = 0; k < coarse.length; k += 1) {
      if (Number.isFinite(coarse[k])) continue;
      const bx = k % bc;
      const by = (k / bc) | 0;
      const near = [];
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = bx + dx;
          const ny = by + dy;
          if (nx < 0 || ny < 0 || nx >= bc || ny >= br) continue;
          const v = coarse[ny * bc + nx];
          if (Number.isFinite(v)) near.push(v);
        }
      }
      if (near.length) coarse[k] = median(near);
    }
  }
  // Bilinear upsample the coarse ground back to full grid resolution.
  const dtm = new Float32Array(cols * rows);
  for (let y = 0; y < rows; y += 1) {
    const gy = Math.min(br - 1, y / block);
    const y0 = Math.floor(gy);
    const y1 = Math.min(br - 1, y0 + 1);
    const fy = gy - y0;
    for (let x = 0; x < cols; x += 1) {
      const gx = Math.min(bc - 1, x / block);
      const x0 = Math.floor(gx);
      const x1 = Math.min(bc - 1, x0 + 1);
      const fx = gx - x0;
      const v00 = coarse[y0 * bc + x0];
      const v10 = coarse[y0 * bc + x1];
      const v01 = coarse[y1 * bc + x0];
      const v11 = coarse[y1 * bc + x1];
      const top = v00 + (v10 - v00) * fx;
      const bot = v01 + (v11 - v01) * fx;
      dtm[y * cols + x] = top + (bot - top) * fy;
    }
  }
  return dtm;
}

/**
 * 3x3 median filter over the heightfield.
 *
 * We now extrude each cell at its OWN roof height instead of one flat height per
 * footprint, which is what makes the massing read as 3D — but raw DSM cells carry
 * reconstruction speckle, and speckle would turn into a forest of spurious little
 * wall steps. A 3x3 median kills single-cell spikes while preserving the genuine
 * step edges between neighbouring rooftops (which is exactly what a median does and
 * a blur would not).
 */
function medianFilter3(height, mask, cols, rows) {
  const out = new Float32Array(cols * rows);
  const win = [];
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const i = y * cols + x;
      if (!mask[i] || !Number.isFinite(height[i])) { out[i] = Number.NaN; continue; }
      win.length = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        const ny = y + dy;
        if (ny < 0 || ny >= rows) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          if (nx < 0 || nx >= cols) continue;
          const n = ny * cols + nx;
          if (mask[n] && Number.isFinite(height[n])) win.push(height[n]);
        }
      }
      out[i] = win.length ? median(win) : height[i];
    }
  }
  return out;
}

// 4-connected flood fill over a boolean cell set → array of components (each an
// array of cell indices). Small components (noise) are dropped by the caller.
function connectedComponents(isBuilding, cols, rows) {
  const seen = new Uint8Array(cols * rows);
  const components = [];
  const stack = [];
  for (let start = 0; start < isBuilding.length; start += 1) {
    if (!isBuilding[start] || seen[start]) continue;
    const comp = [];
    stack.length = 0;
    stack.push(start);
    seen[start] = 1;
    while (stack.length) {
      const i = stack.pop();
      comp.push(i);
      const x = i % cols;
      const y = (i / cols) | 0;
      if (x > 0) { const n = i - 1; if (isBuilding[n] && !seen[n]) { seen[n] = 1; stack.push(n); } }
      if (x < cols - 1) { const n = i + 1; if (isBuilding[n] && !seen[n]) { seen[n] = 1; stack.push(n); } }
      if (y > 0) { const n = i - cols; if (isBuilding[n] && !seen[n]) { seen[n] = 1; stack.push(n); } }
      if (y < rows - 1) { const n = i + cols; if (isBuilding[n] && !seen[n]) { seen[n] = 1; stack.push(n); } }
    }
    components.push(comp);
  }
  return components;
}

/**
 * Extract extruded building blocks from a terrain grid.
 *
 * @param {{cols:number, rows:number, box:{min:number[],max:number[]}, height:Float32Array, mask:Uint8Array}} grid
 * @param {object} [opts]
 * @param {number} [opts.metersPerUnit=1]   grid Y is in these units; thresholds are metres.
 * @param {number} [opts.minHeightM=2.5]     min height-above-ground to count as a structure.
 * @param {number} [opts.minFootprintM2=30]  drop blobs with a smaller plan area.
 * @returns {{positions:Float32Array, colors:Uint8Array, indices:Uint32Array, count:number}|null}
 *          Mesh arrays in the grid's UNSCALED frame (caller applies `scale`), or null
 *          when nothing building-like was found. Never throws.
 */
export function extractBuildings(grid, opts = {}) {
  try {
    if (!grid || !grid.height || !grid.mask || !grid.box) return null;
    const { cols, rows, box, height, mask } = grid;
    if (!(cols > 4) || !(rows > 4)) return null;
    const metersPerUnit = opts.metersPerUnit > 0 ? opts.metersPerUnit : 1;
    const minHeightUnits = (opts.minHeightM ?? 2.5) / metersPerUnit;

    const spanX = box.max[0] - box.min[0];
    const spanZ = box.max[2] - box.min[2];
    if (!(spanX > 0) || !(spanZ > 0)) return null;
    const cellX = spanX / (cols - 1);
    const cellZ = spanZ / (rows - 1);
    const cellAreaM2 = cellX * cellZ * metersPerUnit * metersPerUnit;
    const minCells = Math.max(4, Math.round((opts.minFootprintM2 ?? 30) / Math.max(1e-6, cellAreaM2)));

    // Ground datum. The window MUST be wider than a whole city block or it never
    // sees a street: in dense fabric a 28 m window sits entirely on rooftops, the
    // DTM gets dragged up to roof level, height-above-ground collapses to ~0 and
    // every building extrudes as a flat plate. 60 m reaches real street level in
    // normal urban fabric, and the 5th percentile (was 12th) resists the case where
    // a window is still mostly roof.
    const blockM = opts.groundWindowM ?? 60;
    const block = Math.max(8, Math.round(blockM / Math.max(1e-6, cellX * metersPerUnit)));
    const dtm = estimateGround(height, mask, cols, rows, block, 0.05);
    // Per-cell roof heights, de-speckled (see medianFilter3).
    const roofField = medianFilter3(height, mask, cols, rows);
    // One storey ~3 m. Quantising roof heights to storeys is what turns noisy,
    // slightly-varying cells into clean architectural massing: neighbours on the
    // same floor snap flush (no wall between them), genuine height changes survive
    // as crisp vertical steps, and the triangle budget stays bounded.
    const storeyM = opts.storeyM ?? 3;
    const storeyUnits = Math.max(1e-6, storeyM / metersPerUnit);

    // Height-above-ground classification.
    const isBuilding = new Uint8Array(cols * rows);
    for (let i = 0; i < isBuilding.length; i += 1) {
      if (!mask[i] || !Number.isFinite(height[i])) continue;
      if (height[i] - dtm[i] >= minHeightUnits) isBuilding[i] = 1;
    }

    const components = connectedComponents(isBuilding, cols, rows).filter((c) => c.length >= minCells);
    if (!components.length) return null;

    // World X/Z of a cell corner (grid index space maps linearly onto box).
    const px = (c) => box.min[0] + c * cellX;
    const pz = (r) => box.min[2] + r * cellZ;

    const positions = [];
    const colors = [];
    const indices = [];
    const member = new Uint8Array(cols * rows); // per-component membership scratch
    const roofY = new Float32Array(cols * rows); // quantised roof height per cell

    const pushVert = (x, y, z, col) => {
      const idx = positions.length / 3;
      positions.push(x, y, z);
      colors.push(col[0], col[1], col[2]);
      return idx;
    };
    const pushQuad = (a, b, c, d) => { indices.push(a, b, c, a, c, d); };

    // Hard ceiling so a pathological grid can never blow up memory.
    const MAX_QUADS = 1_200_000;
    let quads = 0;

    for (const comp of components) {
      // Shared base for the footprint, from the bare-earth model beneath it.
      // NOTE: the base is NOT clamped against the roof any more. The old code did
      // `min(ground, medianRoof - minHeight*0.5)`, which silently pinned every
      // building to ~1.25 m tall whenever the ground estimate came out high — the
      // single biggest reason the blocks looked like flat plates.
      const baseH = percentile(comp.map((i) => dtm[i]), 0.15);
      if (!Number.isFinite(baseH)) continue;

      // Each cell keeps its OWN height, snapped to a whole number of storeys above
      // the shared base. This is what breaks a merged city block apart into stepped
      // masses instead of collapsing it to one median plateau.
      let any = false;
      for (const i of comp) {
        const h = Number.isFinite(roofField[i]) ? roofField[i] : height[i];
        if (!Number.isFinite(h)) { roofY[i] = Number.NaN; continue; }
        const storeys = Math.max(1, Math.round((h - baseH) / storeyUnits));
        roofY[i] = baseH + storeys * storeyUnits;
        member[i] = 1;
        any = true;
      }
      if (!any) continue;

      // Slight per-building tint so adjacent blocks read as separate masses.
      const tint = 150 + ((comp[0] * 37) % 70);
      const wallCol = [Math.round(tint * 0.72), Math.round(tint * 0.72), Math.round(tint * 0.70)];

      // Height of whatever abuts this edge: a same-footprint neighbour's roof, or
      // ground outside the footprint. A wall is then simply the gap between this
      // cell's roof and that — which yields perimeter walls AND internal storey
      // steps from one rule.
      const abutting = (nx, ny) => {
        if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) return baseH;
        const n = ny * cols + nx;
        if (!member[n] || !Number.isFinite(roofY[n])) return baseH;
        return roofY[n];
      };

      for (const i of comp) {
        if (quads >= MAX_QUADS) break;
        const top = roofY[i];
        if (!Number.isFinite(top)) continue;
        const cx = i % cols;
        const cy = (i / cols) | 0;
        const x0 = px(cx) - cellX * 0.5;
        const x1 = px(cx) + cellX * 0.5;
        const z0 = pz(cy) - cellZ * 0.5;
        const z1 = pz(cy) + cellZ * 0.5;

        // Roofs brighten with height so storey changes are legible even head-on.
        const lift = Math.max(0, Math.min(1, (top - baseH) / (storeyUnits * 8)));
        const roofCol = [
          Math.min(240, Math.round(tint + 30 + lift * 45)),
          Math.min(232, Math.round(tint + 22 + lift * 40)),
          Math.min(220, Math.round(tint + 4 + lift * 34)),
        ];

        // Roof quad (upward).
        const r0 = pushVert(x0, top, z0, roofCol);
        const r1 = pushVert(x1, top, z0, roofCol);
        const r2 = pushVert(x1, top, z1, roofCol);
        const r3 = pushVert(x0, top, z1, roofCol);
        pushQuad(r0, r1, r2, r3);
        quads += 1;

        // -Z edge
        let bot = abutting(cx, cy - 1);
        if (top > bot + 1e-6) {
          const a = pushVert(x0, bot, z0, wallCol);
          const b = pushVert(x1, bot, z0, wallCol);
          const c = pushVert(x1, top, z0, wallCol);
          const d = pushVert(x0, top, z0, wallCol);
          pushQuad(a, b, c, d);
          quads += 1;
        }
        // +Z edge
        bot = abutting(cx, cy + 1);
        if (top > bot + 1e-6) {
          const a = pushVert(x1, bot, z1, wallCol);
          const b = pushVert(x0, bot, z1, wallCol);
          const c = pushVert(x0, top, z1, wallCol);
          const d = pushVert(x1, top, z1, wallCol);
          pushQuad(a, b, c, d);
          quads += 1;
        }
        // -X edge
        bot = abutting(cx - 1, cy);
        if (top > bot + 1e-6) {
          const a = pushVert(x0, bot, z1, wallCol);
          const b = pushVert(x0, bot, z0, wallCol);
          const c = pushVert(x0, top, z0, wallCol);
          const d = pushVert(x0, top, z1, wallCol);
          pushQuad(a, b, c, d);
          quads += 1;
        }
        // +X edge
        bot = abutting(cx + 1, cy);
        if (top > bot + 1e-6) {
          const a = pushVert(x1, bot, z0, wallCol);
          const b = pushVert(x1, bot, z1, wallCol);
          const c = pushVert(x1, top, z1, wallCol);
          const d = pushVert(x1, top, z0, wallCol);
          pushQuad(a, b, c, d);
          quads += 1;
        }
      }

      for (const i of comp) { member[i] = 0; roofY[i] = 0; }
      if (quads >= MAX_QUADS) break;
    }

    if (!indices.length) return null;
    return {
      positions: new Float32Array(positions),
      colors: new Uint8Array(colors),
      indices: new Uint32Array(indices),
      count: components.length,
    };
  } catch {
    // Buildings are a bonus layer — never let an edge case break the main export.
    return null;
  }
}
