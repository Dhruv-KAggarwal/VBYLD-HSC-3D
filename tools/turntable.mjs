/**
 * Render a turntable MP4 from a GLB, outside the pipeline.
 * ---------------------------------------------------------------------------
 * The pipeline renders one automatically at the end of every job and the
 * server can render one on demand, but this is the way to make a turntable
 * from an arbitrary file, re-render one with different settings, or check that
 * FFmpeg and the rasteriser work without starting a reconstruction.
 *
 *   node tools/turntable.mjs output/latest/model.glb
 *   node tools/turntable.mjs model.glb --out=docs/sample.mp4 --frames=180 --w=1920 --h=1080
 *   node tools/turntable.mjs model.glb --el=45 --fps=24 --crf=20 --dist=3.0
 */

import { resolve } from 'node:path';
import { commands } from '../lib/reconstruct.mjs';
import { renderTurntable, TURNTABLE_DEFAULTS } from '../lib/turntable.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.split('=').slice(1).join('=');
};

const input = resolve(args.find((a) => !a.startsWith('--')) || 'output/latest/model.glb');
const outPath = resolve(flag('out', input.replace(/\.glb$/i, '') + '-turning.mp4'));

console.log(`Rendering turntable from ${input}`);
const outcome = await renderTurntable({
  glbPath: input,
  outPath,
  ffmpeg: process.env.FFMPEG_PATH || commands.ffmpeg,
  frames: Number(flag('frames', TURNTABLE_DEFAULTS.frames)),
  width: Number(flag('w', TURNTABLE_DEFAULTS.width)),
  height: Number(flag('h', TURNTABLE_DEFAULTS.height)),
  fps: Number(flag('fps', TURNTABLE_DEFAULTS.fps)),
  elevation: Number(flag('el', TURNTABLE_DEFAULTS.elevation)),
  crf: Number(flag('crf', TURNTABLE_DEFAULTS.crf)),
  distanceScale: Number(flag('dist', TURNTABLE_DEFAULTS.distanceScale)),
  onProgress: ({ frame, total }) => {
    process.stdout.write(`\r  frame ${frame}/${total}   `);
  },
});
process.stdout.write('\r');

if (!outcome.ok) {
  console.error(`Failed: ${outcome.reason}`);
  process.exit(1);
}
console.log(
  `Wrote ${outcome.path}\n`
  + `  ${outcome.frames} frames, ${outcome.width}x${outcome.height} @ ${outcome.fps}fps `
  + `(${outcome.durationSec.toFixed(1)}s), ${(outcome.bytes / 1e6).toFixed(2)} MB\n`
  + `  model: ${outcome.faceCount.toLocaleString()} faces, ${outcome.textured ? 'textured atlas' : 'vertex colours'}\n`
  + `  took ${(outcome.elapsedMs / 1000).toFixed(1)}s`
  + (outcome.posterPath ? `\n  poster: ${outcome.posterPath}` : ''),
);
