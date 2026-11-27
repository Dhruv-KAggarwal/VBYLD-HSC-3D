/**
 * Dependency tracking for derived products.
 *
 * Every enhancement records a fingerprint of the files it was computed from. A
 * derived product is current only while those files still have that content, so
 * repairing a mesh or re-running a reconstruction can never leave an older LOD
 * set or confidence map looking like it describes the new model.
 */

import { createHash } from 'node:crypto';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { readJson, writeJson } from './multivideo/manifest.mjs';

export const productsDir = (work) => join(work, 'products');
const registryPath = (work) => join(productsDir(work), 'artifacts.json');

const cache = new Map();

/** Content hash, memoised on size and modification time. */
export async function fingerprint(path) {
  if (!path || !existsSync(path)) return null;
  const info = statSync(path);
  const stamp = `${info.size}:${info.mtimeMs}`;
  const cached = cache.get(path);
  if (cached?.stamp === stamp) return cached.hash;
  const hash = await new Promise((resolve, reject) => {
    const digest = createHash('sha1');
    createReadStream(path)
      .on('data', (chunk) => digest.update(chunk))
      .on('error', reject)
      .on('end', () => resolve(digest.digest('hex')));
  });
  cache.set(path, { stamp, hash });
  return hash;
}

/** The immutable reconstruction output every enhancement starts from. */
export function originalModelPath(work) {
  return ['model.glb', 'terrain.glb'].map((name) => join(productsDir(work), name)).find(existsSync) || null;
}

export const repairedModelPath = (work) => join(productsDir(work), 'model-repaired.glb');

export async function readRegistry(work) {
  return (await readJson(registryPath(work))) || { version: 1, activeModel: 'original', entries: {} };
}

/**
 * @param {string[]} sources  files the product was computed from
 */
export async function recordArtifact(work, kind, { sources, settings = null, ...rest }) {
  const registry = await readRegistry(work);
  const fingerprints = {};
  for (const source of sources) {
    fingerprints[source.slice(productsDir(work).length + 1).replace(/\\/g, '/')] = await fingerprint(source);
  }
  registry.entries[kind] = { createdAt: new Date().toISOString(), sources: fingerprints, settings, ...rest };
  await writeJson(registryPath(work), registry);
  return registry;
}

export async function removeArtifact(work, kind) {
  const registry = await readRegistry(work);
  delete registry.entries[kind];
  await writeJson(registryPath(work), registry);
}

export async function setActiveModel(work, which) {
  const registry = await readRegistry(work);
  registry.activeModel = which;
  await writeJson(registryPath(work), registry);
  return registry;
}

/** @returns {{exists:boolean, current:boolean, reason:string|null, entry:object|null}} */
export async function artifactState(work, kind, registry = null) {
  const state = registry || await readRegistry(work);
  const entry = state.entries[kind];
  if (!entry) return { exists: false, current: false, reason: 'not generated', entry: null };
  for (const [name, hash] of Object.entries(entry.sources)) {
    const now = await fingerprint(join(productsDir(work), name));
    if (now !== hash) {
      return { exists: true, current: false, reason: `${name} changed after this was generated`, entry };
    }
  }
  // A product derived from the active model goes stale when another one is activated.
  if (entry.followsActive && entry.model !== state.activeModel) {
    return { exists: true, current: false, reason: `generated from the ${entry.model} model, but the ${state.activeModel} model is active`, entry };
  }
  return { exists: true, current: true, reason: null, entry };
}

/** The model currently presented to the operator; falls back to the original. */
export async function activeModel(work) {
  const registry = await readRegistry(work);
  if (registry.activeModel === 'repaired') {
    const state = await artifactState(work, 'repair', registry);
    if (state.current && existsSync(repairedModelPath(work))) return { which: 'repaired', path: repairedModelPath(work) };
  }
  return { which: 'original', path: originalModelPath(work) };
}

export async function allStates(work) {
  const registry = await readRegistry(work);
  const states = {};
  for (const kind of ['analysis', 'repair', 'confidence', 'confidenceRepaired', 'lod']) {
    const { exists, current, reason, entry } = await artifactState(work, kind, registry);
    states[kind] = { exists, current, reason, createdAt: entry?.createdAt || null, settings: entry?.settings || null, model: entry?.model || null };
  }
  const active = await activeModel(work);
  return { activeModel: active.which, hasModel: Boolean(originalModelPath(work)), artifacts: states };
}
