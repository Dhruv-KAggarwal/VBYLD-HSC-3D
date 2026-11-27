/**
 * A least-recently-used cache of loaded models with a GPU memory budget.
 * Pinned entries (on screen, or needed for measurement) are never evicted.
 */

export class ModelCache {
  constructor(budgetBytes, dispose) {
    this.budget = budgetBytes;
    this.dispose = dispose;
    this.entries = new Map();
  }

  get(key) {
    const entry = this.entries.get(key);
    if (entry) {
      this.entries.delete(key);
      this.entries.set(key, entry);
    }
    return entry || null;
  }

  has(key) { return this.entries.has(key); }

  put(key, entry) {
    this.remove(key);
    this.entries.set(key, entry);
  }

  remove(key) {
    const entry = this.entries.get(key);
    if (!entry) return;
    this.entries.delete(key);
    this.dispose(entry.model);
  }

  bytes() {
    let total = 0;
    for (const entry of this.entries.values()) total += entry.bytes || 0;
    return total;
  }

  trim(pinned) {
    for (const [key] of [...this.entries]) {
      if (this.bytes() <= this.budget) break;
      if (!pinned.has(key)) this.remove(key);
    }
  }

  clear() {
    for (const [key] of [...this.entries]) this.remove(key);
  }
}
