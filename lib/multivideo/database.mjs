/**
 * Read (and narrowly prune) the COLMAP SQLite database.
 *
 * Uses Node's built-in `node:sqlite` (Node 22.13+), so no dependency is added.
 * On an older Node `openDatabase` returns null and callers fall back to
 * strategies that do not need it, saying so in the log.
 */

const MAX_IMAGES = 2147483647;

let sqlite;
export async function openDatabase(path, { readOnly = true } = {}) {
  if (sqlite === undefined) {
    try { sqlite = await import('node:sqlite'); } catch { sqlite = null; }
  }
  if (!sqlite) return null;
  const db = new sqlite.DatabaseSync(path, { readOnly });
  db.exec('PRAGMA busy_timeout = 10000');
  return db;
}

export function pairIds(pairId) {
  const second = pairId % MAX_IMAGES;
  return [(pairId - second) / MAX_IMAGES, second];
}

export function pairId(a, b) {
  const [low, high] = a < b ? [a, b] : [b, a];
  return low * MAX_IMAGES + high;
}

export function readImages(db) {
  const images = new Map();
  for (const row of db.prepare('SELECT image_id, name, camera_id FROM images').all()) {
    images.set(Number(row.image_id), { id: Number(row.image_id), name: row.name, cameraId: Number(row.camera_id) });
  }
  return images;
}

/** Geometric-verification results plus the raw match count of each pair. */
export function readGeometries(db) {
  const raw = new Map();
  for (const row of db.prepare('SELECT pair_id, rows FROM matches').all()) raw.set(Number(row.pair_id), Number(row.rows));
  return db.prepare('SELECT pair_id, rows, config FROM two_view_geometries').all().map((row) => {
    const id = Number(row.pair_id);
    const [a, b] = pairIds(id);
    return { a, b, inliers: Number(row.rows), config: Number(row.config), matches: raw.get(id) || 0 };
  });
}

/**
 * Local descriptors of one image as floats, evenly subsampled.
 *
 * SIFT descriptors are bytes and get the RootSIFT mapping; learned descriptors
 * are float32 stored in the same blob.
 */
export function readDescriptors(db, imageId, limit = 1200) {
  const row = db.prepare('SELECT * FROM descriptors WHERE image_id = ?').get(imageId);
  if (!row?.data || !row.rows) return null;
  const rows = Number(row.rows);
  const cols = Number(row.cols);
  const bytes = row.data;
  let dim = cols;
  let isFloat = false;
  if (bytes.length === rows * cols * 4) isFloat = true;
  else if (Number(row.type || 0) > 1 && cols % 4 === 0) { isFloat = true; dim = cols / 4; }
  else if (Number(row.type || 0) === 1 && cols % 4 === 0 && cols > 128) { isFloat = true; dim = cols / 4; }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = Math.min(rows, limit);
  const step = rows / count;
  const data = new Float32Array(count * dim);
  for (let i = 0; i < count; i += 1) {
    const source = Math.floor(i * step);
    let sum = 0;
    for (let k = 0; k < dim; k += 1) {
      const value = isFloat ? view.getFloat32((source * dim + k) * 4, true) : bytes[source * dim + k];
      data[i * dim + k] = value;
      sum += isFloat ? value * value : value;
    }
    if (sum > 0) {
      for (let k = 0; k < dim; k += 1) {
        data[i * dim + k] = isFloat ? data[i * dim + k] / Math.sqrt(sum) : Math.sqrt(data[i * dim + k] / sum);
      }
    }
  }
  return { data, count, dim };
}

/** Forget a pair so a later matching pass will attempt it again. */
export function deletePairs(db, ids) {
  const matches = db.prepare('DELETE FROM matches WHERE pair_id = ?');
  const geometry = db.prepare('DELETE FROM two_view_geometries WHERE pair_id = ?');
  db.exec('BEGIN');
  try {
    for (const id of ids) { matches.run(id); geometry.run(id); }
    db.exec('COMMIT');
  } catch (error) {
    db.exec('ROLLBACK');
    throw error;
  }
}
