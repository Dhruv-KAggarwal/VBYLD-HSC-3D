/**
 * Staged multi-video uploads.
 *
 * Each video is its own request, streamed to disk while being hashed, then
 * validated with FFprobe. The session lives in runtime/uploads/<id>/ until a
 * reconstruction claims it, it is deleted, or it goes stale.
 */

import { createHash, randomUUID } from 'node:crypto';
import { createWriteStream, existsSync } from 'node:fs';
import { mkdir, readdir, rename, rm, stat, statfs, utimes } from 'node:fs/promises';
import { join } from 'node:path';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { assessQuality, displayName, makeThumbnail, probeMedia, safeExtension } from './multivideo/video-input.mjs';
import { normalizeVideoSettings, readJson, videoId, writeJson } from './multivideo/manifest.mjs';
import { allocateBudget, totalFrameBudget, trimmedDuration } from './multivideo/frame-selection.mjs';
import { PROFILE_MATCHING } from './multivideo/matching.mjs';

const GB = 1024 ** 3;

export function uploadLimits() {
  const number = (name, fallback) => Number(process.env[name]) > 0 ? Number(process.env[name]) : fallback;
  return {
    maxVideos: Math.round(number('SKYFORGE_MAX_VIDEOS', 64)),
    maxVideoBytes: number('SKYFORGE_MAX_VIDEO_GB', 16) * GB,
    maxTotalBytes: number('SKYFORGE_MAX_TOTAL_GB', 64) * GB,
    // Left free for frames, the database, depth maps and products.
    reserveBytes: number('SKYFORGE_DISK_RESERVE_GB', 5) * GB,
    sessionHours: number('SKYFORGE_UPLOAD_TTL_HOURS', 24),
  };
}

const fail = (status, message) => Object.assign(new Error(message), { status });

export async function freeBytes(path) {
  try {
    const info = await statfs(path);
    return Number(info.bavail) * Number(info.bsize);
  } catch {
    return null;
  }
}

export function createUploadStore({ root, commands }) {
  const base = join(root, 'uploads');
  const locks = new Map();
  const sessionDir = (id) => {
    if (!/^[0-9a-f-]{36}$/.test(id)) throw fail(404, 'Upload session not found.');
    return join(base, id);
  };
  const sessionPath = (id) => join(sessionDir(id), 'session.json');

  /** Serialise read-modify-write of one session; uploads run in parallel. */
  const locked = (id, task) => {
    const next = (locks.get(id) || Promise.resolve()).then(task, task);
    locks.set(id, next.then(() => undefined, () => undefined));
    return next;
  };
  const load = async (id) => {
    const session = await readJson(sessionPath(id));
    if (!session) throw fail(404, 'Upload session not found.');
    return session;
  };
  const save = (session) => writeJson(sessionPath(session.id), session);
  // Bytes promised to uploads still in flight, so parallel requests cannot each
  // pass the same disk check.
  const reserved = new Map();

  return {
    limits: uploadLimits,
    async create() {
      const id = randomUUID();
      await mkdir(join(sessionDir(id), 'inputs'), { recursive: true });
      await mkdir(join(sessionDir(id), 'thumbs'), { recursive: true });
      const session = { id, createdAt: new Date().toISOString(), nextNumber: 1, videos: [] };
      await save(session);
      return session;
    },

    get: load,

    thumbnailPath(id, vid) {
      if (!/^video_\d{3,}$/.test(vid)) return null;
      const path = join(sessionDir(id), 'thumbs', `${vid}.jpg`);
      return existsSync(path) ? path : null;
    },

    /**
     * Receive one video. `replace` names a video whose slot, order and settings
     * the new file takes over.
     */
    async receive(id, request, { fileName, replace = null }) {
      const limits = uploadLimits();
      const declared = Number(request.headers['content-length'] || 0);
      if (declared > limits.maxVideoBytes) throw fail(413, `This video exceeds the ${(limits.maxVideoBytes / GB).toFixed(0)} GB per-file limit.`);
      const slot = await locked(id, async () => {
        const session = await load(id);
        const previous = replace ? session.videos.find((v) => v.id === replace) : null;
        if (replace && !previous) throw fail(404, 'The video to replace is not in this session.');
        const pending = [...reserved.values()].filter((r) => r.session === id);
        if (!replace && session.videos.length + pending.filter((p) => !p.replace).length >= limits.maxVideos) {
          throw fail(413, `A job accepts at most ${limits.maxVideos} videos (SKYFORGE_MAX_VIDEOS).`);
        }
        const total = session.videos.reduce((s, v) => s + (v.id === replace ? 0 : v.bytes), 0)
          + pending.reduce((s, p) => s + p.bytes, 0) + declared;
        if (total > limits.maxTotalBytes) throw fail(413, `The inputs exceed the ${(limits.maxTotalBytes / GB).toFixed(0)} GB total limit (SKYFORGE_MAX_TOTAL_GB).`);
        const free = await freeBytes(base);
        const promised = [...reserved.values()].reduce((s, p) => s + p.bytes, 0);
        if (free != null && free - promised - declared < limits.reserveBytes) {
          throw fail(507, `Not enough disk space: ${(free / GB).toFixed(1)} GB free, and ${(limits.reserveBytes / GB).toFixed(0)} GB must remain for processing.`);
        }
        const token = randomUUID();
        reserved.set(token, { session: id, bytes: declared, replace });
        return { token };
      });

      const inputs = join(sessionDir(id), 'inputs');
      const partial = join(inputs, `${slot.token}.part`);
      try {
        const hash = createHash('sha256');
        let bytes = 0;
        const meter = new Transform({
          transform(chunk, _encoding, done) {
            bytes += chunk.length;
            if (bytes > limits.maxVideoBytes) { done(fail(413, 'The video exceeds the per-file limit.')); return; }
            hash.update(chunk);
            done(null, chunk);
          },
        });
        try {
          await pipeline(request, meter, createWriteStream(partial));
        } catch (error) {
          throw error.status ? error : fail(400, `Upload failed or was cancelled: ${error.message}`);
        }
        if (bytes < 16 * 1024) throw fail(400, 'The uploaded file is empty or truncated.');
        if (declared && bytes !== declared) throw fail(400, 'The upload was truncated.');
        const sha256 = hash.digest('hex');
        const probe = await probeMedia(existsSync(commands.ffprobe) ? commands.ffprobe : 'ffprobe', partial);
        if (!probe.ok) throw fail(415, probe.error);
        const quality = await assessQuality(commands.ffmpeg, partial, probe);

        return await locked(id, async () => {
          const session = await load(id);
          const duplicate = session.videos.find((v) => v.sha256 === sha256 && v.id !== replace);
          if (duplicate) throw fail(409, `This file is identical to "${duplicate.originalName}", which is already in the list.`);
          const previous = replace ? session.videos.find((v) => v.id === replace) : null;
          const vid = previous?.id || videoId(session.nextNumber++);
          if (previous) await rm(join(sessionDir(id), previous.file), { force: true });
          const file = `inputs/${vid}${safeExtension(fileName)}`;
          await rename(partial, join(sessionDir(id), file));
          const thumb = join(sessionDir(id), 'thumbs', `${vid}.jpg`);
          await rm(thumb, { force: true });
          await makeThumbnail(commands.ffmpeg, join(sessionDir(id), file), thumb, Math.min(1, probe.durationSeconds / 2)).catch(() => false);
          const entry = {
            id: vid,
            order: previous?.order ?? session.videos.length,
            originalName: displayName(fileName),
            file,
            bytes,
            sha256,
            status: 'ready',
            uploadedAt: new Date().toISOString(),
            probe,
            quality,
            settings: previous?.settings || normalizeVideoSettings(),
          };
          if (previous) session.videos[session.videos.indexOf(previous)] = entry;
          else session.videos.push(entry);
          await save(session);
          return entry;
        });
      } finally {
        reserved.delete(slot.token);
        await rm(partial, { force: true }).catch(() => {});
      }
    },

    async update(id, { order = null, settings = null } = {}) {
      return locked(id, async () => {
        const session = await load(id);
        if (Array.isArray(order)) {
          const rank = new Map(order.map((vid, i) => [vid, i]));
          session.videos.sort((a, b) => (rank.get(a.id) ?? 1e9) - (rank.get(b.id) ?? 1e9));
          session.videos.forEach((v, i) => { v.order = i; });
        }
        if (settings && typeof settings === 'object') {
          for (const video of session.videos) {
            if (settings[video.id]) video.settings = normalizeVideoSettings({ ...video.settings, ...settings[video.id] });
          }
        }
        await save(session);
        return session;
      });
    },

    async removeVideo(id, vid) {
      return locked(id, async () => {
        const session = await load(id);
        const video = session.videos.find((v) => v.id === vid);
        if (!video) throw fail(404, 'Video not found.');
        await rm(join(sessionDir(id), video.file), { force: true });
        await rm(join(sessionDir(id), 'thumbs', `${vid}.jpg`), { force: true });
        session.videos = session.videos.filter((v) => v !== video);
        session.videos.forEach((v, i) => { v.order = i; });
        await save(session);
        return session;
      });
    },

    async remove(id) {
      await load(id);
      await rm(sessionDir(id), { recursive: true, force: true });
      locks.delete(id);
    },

    /** Hand the session's files to a job directory. */
    async claim(id, work) {
      return locked(id, async () => {
        const session = await load(id);
        if ([...reserved.values()].some((r) => r.session === id)) throw fail(409, 'Uploads are still in progress.');
        const enabled = session.videos.filter((v) => v.settings.enabled);
        if (!enabled.length) throw fail(400, 'Upload and enable at least one video first.');
        await mkdir(join(work, 'manifests'), { recursive: true });
        await rename(join(sessionDir(id), 'inputs'), join(work, 'inputs'));
        await rename(join(sessionDir(id), 'thumbs'), join(work, 'manifests', 'thumbs'));
        await writeJson(join(work, 'manifests', 'inputs.json'), { version: 1, videos: session.videos });
        await rm(sessionDir(id), { recursive: true, force: true });
        locks.delete(id);
        return session;
      });
    },

    async cleanupStale() {
      if (!existsSync(base)) return 0;
      const cutoff = Date.now() - uploadLimits().sessionHours * 3600 * 1000;
      let removed = 0;
      for (const name of await readdir(base)) {
        if ([...reserved.values()].some((r) => r.session === name)) continue;
        const dir = join(base, name);
        const info = await stat(join(dir, 'session.json')).catch(() => stat(dir).catch(() => null));
        if (info && info.mtimeMs < cutoff) { await rm(dir, { recursive: true, force: true }); removed += 1; }
      }
      return removed;
    },

    touch: async (id) => utimes(sessionPath(id), new Date(), new Date()).catch(() => {}),
  };
}

/** What a set of videos will demand of the pipeline, before it is started. */
export function estimateRequirements(videos, { quality = 'balanced', hardware = {}, maxFrames = null } = {}) {
  const enabled = videos.filter((v) => v.settings?.enabled !== false);
  const totalDuration = enabled.reduce((s, v) => s + trimmedDuration(v), 0);
  const totalBytes = videos.reduce((s, v) => s + v.bytes, 0);
  if (!enabled.length) return { videos: 0, totalDurationSeconds: 0, totalBytes, estimatedFrames: 0 };
  const budgets = allocateBudget(enabled, quality, totalFrameBudget(quality, hardware, maxFrames));
  const frames = [...budgets.values()].reduce((a, b) => a + b, 0);
  const profile = PROFILE_MATCHING[quality] || PROFILE_MATCHING.balanced;
  const intra = frames * (profile.minWindow + profile.loops + 3);
  const cross = enabled.length > 1 ? Math.round((frames * profile.crossPerImage) / 2) : 0;
  const width = { fast: 1280, balanced: 1920, detailed: 2560 }[quality] || 1920;
  const megapixels = enabled.reduce((s, v) => {
    const w = Math.min(width, v.probe.width);
    return s + ((w * w * v.probe.height) / v.probe.width / 1e6) * budgets.get(v.id);
  }, 0);
  // Per extracted megapixel: JPEG frame, undistorted copy, database and dense data.
  const workingBytes = Math.round(megapixels * 4.5 * 1024 * 1024);
  const warnings = [];
  if (frames > 350 && hardware.ramGb && hardware.ramGb < 24) warnings.push('Large jobs on under 24 GB of RAM can exhaust memory during meshing.');
  if (enabled.some((v) => v.quality?.rating === 'poor')) warnings.push('At least one video has quality problems; it may register poorly.');
  return {
    videos: enabled.length,
    totalDurationSeconds: Math.round(totalDuration),
    totalBytes,
    estimatedFrames: frames,
    framesPerVideo: Object.fromEntries(budgets),
    estimatedIntraPairs: intra,
    estimatedCrossPairs: cross,
    estimatedWorkingBytes: workingBytes,
    warnings,
    basis: 'estimates from clip duration and profile limits; actual counts depend on motion and overlap',
  };
}
