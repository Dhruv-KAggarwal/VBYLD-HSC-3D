/**
 * Offline multi-angle renderer for a textured GLB.
 * ---------------------------------------------------------------------------
 * The browser console shows the model interactively, but to *analyse* a
 * reconstruction from the terminal — and to compare two runs side by side — we
 * need flat images from fixed viewpoints. This honours the per-face UV atlas
 * OpenMVS bakes, so it shows the true textured 3-D mesh as the viewer would.
 *
 * The rasteriser itself now lives in lib/meshrender.mjs, because the turntable
 * video needs the identical renderer driven around a circle. This file is the
 * command-line front end for it.
 *
 *   node tools/render-glb.mjs output/latest/model.glb --out=output/analysis --views=all
 */

import { writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { encodePng } from '../lib/png.mjs';
import { loadRenderModel, renderView, VIEWS } from '../lib/meshrender.mjs';

async function main() {
  const args = process.argv.slice(2);
  const flag = (name, fallback) => (args.find((a) => a.startsWith(`--${name}=`)) || `--${name}=${fallback}`).split('=').slice(1).join('=');
  const input = resolve(args.find((a) => !a.startsWith('--')) || 'output/latest/model.glb');
  const outDir = resolve(flag('out', 'output/analysis'));
  const width = Number(flag('w', 1000));
  const height = Number(flag('h', 750));
  const viewsArg = flag('views', 'iso,top,front,left');
  const viewNames = viewsArg === 'all' ? Object.keys(VIEWS) : viewsArg.split(',').filter((v) => VIEWS[v]);
  const flipV = args.includes('--flipv');
  const flat = args.includes('--flat'); // ignore texture, show pure geometry
  const supersample = args.includes('--ss'); // 2x2 texture average
  await mkdir(outDir, { recursive: true });

  const model = await loadRenderModel(input);
  const { box, texture } = model;
  console.log(`Model: ${model.vertexCount} verts, ${model.faceCount} faces, texture ${texture ? `${texture.width}x${texture.height}` : 'none'}`);
  console.log(`Extent: X=${(box.max[0] - box.min[0]).toFixed(1)} Y=${(box.max[1] - box.min[1]).toFixed(1)} Z=${(box.max[2] - box.min[2]).toFixed(1)} m`);

  for (const name of viewNames) {
    const { az, el } = VIEWS[name];
    const { pixels, drawn } = renderView(model, {
      width, height, az, el, distanceScale: 2.6, flat, flipV, supersample,
    });
    const outPath = resolve(outDir, `${name}.png`);
    await writeFile(outPath, encodePng(pixels, width, height));
    console.log(`  ${name.padEnd(6)} -> ${outPath} (${drawn} faces)`);
  }
}

main().catch((error) => { console.error(error); process.exit(1); });
