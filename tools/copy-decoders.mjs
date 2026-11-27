/**
 * Copy the Draco and Basis decoders that ship with three.js into public/, so the
 * viewer never fetches them from a CDN. Runs before `dev` and `build`.
 */

import { cp, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const source = join(root, 'node_modules', 'three', 'examples', 'jsm', 'libs');

for (const name of ['draco', 'basis']) {
  const from = join(source, name);
  if (!existsSync(from)) {
    console.warn(`three.js ${name} decoder not found; compressed assets of that kind will not load.`);
    continue;
  }
  const to = join(root, 'public', 'decoders', name);
  await mkdir(to, { recursive: true });
  await cp(from, to, { recursive: true });
}
