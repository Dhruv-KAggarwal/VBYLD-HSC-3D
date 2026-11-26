/**
 * Flip TEXCOORD_0 V in an existing GLB, in place.
 *
 * One-shot repair for models produced while the pipeline was wrongly applying
 * an atlas V flip. Those files render as a uniform untextured blob because
 * every face samples the unused region of the atlas. The pipeline no longer
 * flips, so new runs do not need this; it exists so a finished reconstruction
 * does not have to be rebuilt from scratch.
 *
 * Usage: node tools/repair-uv.mjs <file.glb> [...]
 */

import { readFileSync, writeFileSync, copyFileSync, existsSync } from 'node:fs';
import { flipTextureV } from '../lib/glbtransform.mjs';

const files = process.argv.slice(2);
if (!files.length) {
  console.error('Usage: node tools/repair-uv.mjs <file.glb> [...]');
  process.exit(2);
}

for (const file of files) {
  if (!existsSync(file)) { console.log(`skip (missing): ${file}`); continue; }
  try {
    const before = readFileSync(file);
    const after = flipTextureV(before);
    if (after.length === before.length && after.equals(before)) {
      console.log(`unchanged (no float UVs found): ${file}`);
      continue;
    }
    // Keep the original next to it; this is a destructive rewrite otherwise.
    const backup = `${file}.bak`;
    if (!existsSync(backup)) copyFileSync(file, backup);
    writeFileSync(file, after);
    console.log(`repaired: ${file}  (original kept at ${backup})`);
  } catch (error) {
    console.log(`failed: ${file} — ${error.message}`);
  }
}
