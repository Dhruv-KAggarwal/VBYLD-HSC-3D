/**
 * The multi-video front end of the reconstruction pipeline.
 *
 * Takes the job from stored source videos to a COLMAP database holding verified
 * intra- and cross-video correspondences. Everything after that (SfM, dense
 * reconstruction, meshing, texturing) is the same code single-video jobs run.
 */

import { existsSync, statSync } from 'node:fs';
import { mkdir, readdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { analyzeVideo, allocateBudget, chooseFrames, extractFrames, totalFrameBudget, trimmedDuration } from './frame-selection.mjs';
import { markDuplicatesAcrossVideos, markDuplicatesWithinVideo } from './frame-deduplication.mjs';
import { exhaustiveCrossPairs, intraVideoPairs, interVideoPairs, isCrossPair, matcherArgs, PairSet, writePairs } from './matching.mjs';
import { analyzeConnectivity, isVerified } from './connectivity.mjs';
import { deletePairs, openDatabase, pairId, readGeometries, readImages } from './database.mjs';
import { buildRetrievalIndex } from './retrieval.mjs';
import { connectivityPath, frameRecord, framesPath, readInputs, videoOfImage, writeInputs, writeJson } from './manifest.mjs';

const EXHAUSTIVE_IMAGE_LIMIT = 50;
const COMPONENT_PAIR_LIMIT = 4000;

/** Videos taking part, in operator order, with display labels. */
export async function activeVideos(work) {
  const manifest = await readInputs(work);
  if (!manifest?.videos?.length) throw new Error('The job has no input manifest.');
  return manifest.videos
    .filter((v) => v.settings?.enabled !== false)
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
    .map((v) => ({ ...v, path: resolve(work, v.file), label: v.settings?.label || v.originalName || v.id }));
}

const even = (value) => Math.max(2, Math.round(value / 2) * 2);

function cameraParams(model, fovDeg, width, height) {
  if (!fovDeg) return null;
  const f = width / (2 * Math.tan((fovDeg * Math.PI) / 360));
  const cx = width / 2;
  const cy = height / 2;
  const values = {
    SIMPLE_PINHOLE: [f, cx, cy],
    SIMPLE_RADIAL: [f, cx, cy, 0],
    RADIAL: [f, cx, cy, 0, 0],
    OPENCV: [f, f, cx, cy, 0, 0, 0, 0],
    OPENCV_FISHEYE: [f, f, cx, cy, 0, 0, 0, 0],
  }[model];
  return values ? values.map((v) => Number(v.toFixed(4))).join(',') : null;
}

function abortSignalFor(job) {
  const controller = new AbortController();
  if (job.cancelled) controller.abort();
  const hook = () => controller.abort();
  job.cancelHooks?.add(hook);
  return { signal: controller.signal, release: () => job.cancelHooks?.delete(hook) };
}

/** Frame sampling for every video: analyse, select, deduplicate, extract. */
export async function extractAllVideos({ job, work, frames, quality, hardware, tracker, logger, run, commands }) {
  const videos = await activeVideos(work);
  for (const video of videos) {
    if (!existsSync(video.path) || statSync(video.path).size < 1024) throw new Error(`${video.label} is missing or empty.`);
  }
  const total = totalFrameBudget(quality.quality, hardware, job.maxFrames);
  const budgets = allocateBudget(videos, quality.quality, total);
  logger.metric('frame budget', `${[...budgets.values()].reduce((a, b) => a + b, 0)} across ${videos.length} videos`);
  tracker.start('extract', `Analysing ${videos.length} videos`);
  const { signal, release } = abortSignalFor(job);
  const analysed = [];
  try {
    for (let i = 0; i < videos.length; i += 1) {
      const video = videos[i];
      logger.info(`${video.label}: measuring sharpness and camera motion`);
      const { candidates, rate } = await analyzeVideo(commands.ffmpeg, video, budgets.get(video.id), signal);
      chooseFrames(candidates, budgets.get(video.id));
      markDuplicatesWithinVideo(candidates);
      analysed.push({ video, candidates, rate });
      tracker.fraction((0.5 * (i + 1)) / videos.length);
    }
  } finally {
    release();
  }
  const acrossRemoved = markDuplicatesAcrossVideos(analysed.map((a) => a.candidates));
  if (acrossRemoved) logger.warn(`${acrossRemoved} frames repeat footage already present in an earlier video and were removed.`);

  for (let i = 0; i < analysed.length; i += 1) {
    const entry = analysed[i];
    const { video } = entry;
    const width = even(Math.min(quality.presetWidth || quality.mvsWidth, video.probe.width));
    entry.size = { width, height: even((width * video.probe.height) / video.probe.width) };
    await extractFrames({
      ffmpeg: commands.ffmpeg,
      video,
      candidates: entry.candidates,
      rate: entry.rate,
      width,
      outDir: frames,
      run,
      job,
      label: `Extracting frames from ${video.label} (${i + 1}/${analysed.length})`,
      stage: 'extract',
    });
    tracker.fraction(0.5 + (0.5 * (i + 1)) / analysed.length);
  }
  return { videos, analysed };
}

/** frames.json and the per-video counters shown to the operator. */
export async function writeFrameManifest(work, analysed) {
  const records = [];
  const perVideo = analysed.map(({ video, candidates, size }) => {
    for (const c of candidates) records.push(frameRecord(c, size || { width: video.probe.width, height: video.probe.height }));
    const count = (test) => candidates.filter(test).length;
    return {
      id: video.id,
      label: video.label,
      durationSeconds: trimmedDuration(video),
      candidates: candidates.length,
      selected: count((c) => c.selected),
      blurry: count((c) => c.rejectedReason === 'blur'),
      duplicates: count((c) => c.duplicate),
      redundant: count((c) => c.rejectedReason === 'redundant'),
      extractedWidth: size?.width,
      extractedHeight: size?.height,
    };
  });
  await writeJson(framesPath(work), { version: 1, perVideo, frames: records });
  return perVideo;
}

/** Selected frames per video with cumulative visual travel, for matching. */
function matchingVideos(analysed) {
  return analysed.map(({ video, candidates }) => {
    let travel = 0;
    const frames = [];
    for (const c of candidates) {
      travel += c.motion;
      if (c.selected) frames.push({ fileName: c.fileName, travel });
    }
    return { id: video.id, label: video.label, frames };
  }).filter((v) => v.frames.length);
}

export async function extractFeaturesPerVideo({
  job, work, frames, database, quality, analysed, maskDir, learned, tracker, run, commands,
}) {
  let done = 0;
  for (const { video, candidates, size } of analysed) {
    const names = candidates.filter((c) => c.selected).map((c) => c.fileName);
    if (!names.length) continue;
    const list = join(work, 'manifests', `images_${video.id}.txt`);
    await writeFile(list, `${names.join('\n')}\n`);
    const model = video.settings?.cameraModel || 'SIMPLE_RADIAL';
    const params = cameraParams(model, video.settings?.horizontalFovDeg, size.width, size.height);
    // One invocation per video gives each video its own intrinsics (its own
    // lens, zoom and resolution) while frames within a video share them.
    await run(commands.colmap, [
      'feature_extractor',
      '--database_path', database,
      '--image_path', frames,
      '--image_list_path', list,
      '--ImageReader.single_camera', '1',
      '--ImageReader.camera_model', model,
      ...(params ? ['--ImageReader.camera_params', params] : []),
      ...(maskDir ? ['--ImageReader.mask_path', maskDir] : []),
      ...(learned
        ? ['--FeatureExtraction.type', 'ALIKED_N16ROT', '--AlikedExtraction.max_num_features', String(Math.min(8192, Math.round(Number(quality.features) / 2) || 4096))]
        : [
          '--SiftExtraction.max_num_features', quality.features,
          '--SiftExtraction.max_num_orientations', '2',
          '--SiftExtraction.peak_threshold', quality.peakThreshold,
          '--SiftExtraction.edge_threshold', '10',
        ]),
      '--FeatureExtraction.use_gpu', '1',
      '--FeatureExtraction.gpu_index', '0',
      '--FeatureExtraction.num_threads', quality.threads,
    ], work, job, `Detecting features in ${video.label}`, { stage: 'features', timeoutMs: 90 * 60 * 1000 });
    done += names.length;
    tracker.fraction(done / Math.max(1, analysed.reduce((n, a) => n + a.candidates.filter((c) => c.selected).length, 0)));
  }
}

async function runMatching({ job, work, database, quality, run, commands, pairs, name, label, stage, learned, fallback = false }) {
  if (!pairs.length) return;
  const list = join(work, 'manifests', `pairs_${name}.txt`);
  await writePairs(list, pairs);
  await run(commands.colmap, [
    'matches_importer',
    '--database_path', database,
    '--match_list_path', list,
    '--match_type', 'pairs',
    ...matcherArgs(learned, fallback),
    '--FeatureMatching.use_gpu', '1',
    '--FeatureMatching.gpu_index', '0',
    '--FeatureMatching.num_threads', quality.threads,
  ], work, job, `${label} (${pairs.length.toLocaleString()} pairs)`, { stage, timeoutMs: 3 * 60 * 60 * 1000 });
}

async function measureConnectivity(database, videos) {
  const db = await openDatabase(database);
  if (!db) return null;
  try {
    return analyzeConnectivity(readImages(db), readGeometries(db), videos);
  } finally {
    db.close();
  }
}

/** Intra-video matching, retrieval, cross-video matching and recovery rounds. */
export async function matchAndConnect({ job, work, database, quality, analysed, learned, tracker, logger, run, commands }) {
  const videos = matchingVideos(analysed);
  const imageCount = videos.reduce((n, v) => n + v.frames.length, 0);
  const wanted = job.matching || 'auto';
  const exhaustive = wanted === 'exhaustive' || (wanted === 'auto' && imageCount <= EXHAUSTIVE_IMAGE_LIMIT);

  tracker.start('match', 'Matching within each video');
  let index = null;
  const db = await openDatabase(database);
  if (db) {
    try {
      logger.info('Building global image descriptors (VLAD) for loop closure and cross-video retrieval');
      index = buildRetrievalIndex(db, (f) => tracker.fraction(0.15 * f));
    } finally {
      db.close();
    }
  }
  if (!index) logger.warn('Image retrieval is unavailable (needs Node 22.13+ and extracted descriptors); cross-video pairs are sampled exhaustively instead.');

  const intra = intraVideoPairs(videos, quality.quality, index);
  if (exhaustive) {
    for (const v of videos) {
      for (let i = 0; i < v.frames.length; i += 1) {
        for (let j = i + 1; j < v.frames.length; j += 1) intra.add(v.frames[i].fileName, v.frames[j].fileName, 'exhaustive');
      }
    }
  }
  await runMatching({ job, work, database, quality, run, commands, learned, pairs: intra.values(), name: 'intra', label: 'Intra-video matching', stage: 'match' });
  tracker.complete('match');

  tracker.start('crossmatch', 'Matching between videos');
  let cross = new PairSet();
  if (videos.length > 1) {
    if (exhaustive) cross = exhaustiveCrossPairs(videos);
    else if (index) cross = interVideoPairs(videos, quality.quality, index);
    else cross = exhaustiveCrossPairs(videos, { limit: 8000 });
    await runMatching({ job, work, database, quality, run, commands, learned, pairs: cross.values(), name: 'cross', label: 'Cross-video matching', stage: 'crossmatch' });
  }
  tracker.complete('crossmatch');

  tracker.start('connectivity', 'Analysing overlap between videos');
  let report = await measureConnectivity(database, videos);
  const attempts = [];
  const tried = new PairSet();
  for (const p of [...intra.values(), ...cross.values()]) tried.add(p.a, p.b, p.why);

  if (report && videos.length > 1) {
    const rounds = [
      {
        name: 'expanded-retrieval',
        label: 'Retrying disconnected videos with more candidates',
        pairs: (only) => (index ? interVideoPairs(videos, quality.quality, index, { expand: 3, only, skip: tried }) : new PairSet()),
      },
      {
        name: 'exhaustive-between-components',
        label: 'Exhaustive search between disconnected groups',
        pairs: (only) => exhaustiveCrossPairs(videos, { only, skip: tried, limit: COMPONENT_PAIR_LIMIT }),
      },
      { name: 'fallback-matcher', label: 'Retrying with the fallback matcher', fallback: true },
    ];
    for (const round of rounds) {
      if (report.fullyConnected) break;
      const group = new Map(report.videos.map((v) => [v.id, v.component]));
      const apart = (a, b) => group.get(a) !== group.get(b);
      let pairs;
      if (round.fallback) {
        // Re-match the most promising unverified pairs between the groups with
        // a different matcher. COLMAP skips known pairs, so forget them first.
        const writable = await openDatabase(database, { readOnly: false });
        const candidates = [];
        try {
          const images = readImages(writable);
          const verified = new Set(readGeometries(writable).filter(isVerified).map((g) => pairId(g.a, g.b)));
          const ids = new Map([...images.values()].map((i) => [i.name, i.id]));
          const position = new Map(index?.names.map((n, i) => [n, i]) || []);
          for (const p of tried.values()) {
            if (!isCrossPair(p) || !apart(videoOfImage(p.a), videoOfImage(p.b))) continue;
            const a = ids.get(p.a);
            const b = ids.get(p.b);
            if (!a || !b || verified.has(pairId(a, b))) continue;
            const score = index ? index.similarity(position.get(p.a), position.get(p.b)) : 0;
            candidates.push({ ...p, id: pairId(a, b), score });
          }
          candidates.sort((x, y) => y.score - x.score);
          pairs = candidates.slice(0, 1500);
          deletePairs(writable, pairs.map((p) => p.id));
        } finally {
          writable.close();
        }
      } else {
        pairs = round.pairs(apart).values();
        for (const p of pairs) tried.add(p.a, p.b, p.why);
      }
      if (!pairs.length) continue;
      logger.warn(`Videos are not all connected (${report.components.length} groups); ${round.label.toLowerCase()}.`);
      try {
        await runMatching({ job, work, database, quality, run, commands, learned, pairs, name: round.name, label: round.label, stage: 'connectivity', fallback: round.fallback });
      } catch (error) {
        if (/cancelled|timeout/i.test(error.message)) throw error;
        logger.warn(`${round.label} failed: ${error.message.slice(0, 300)}`);
        attempts.push({ name: round.name, pairs: pairs.length, error: error.message.slice(0, 300) });
        continue;
      }
      const before = report.components.length;
      report = await measureConnectivity(database, videos);
      attempts.push({ name: round.name, pairs: pairs.length, componentsBefore: before, componentsAfter: report.components.length });
    }
  }

  const summary = {
    available: Boolean(report),
    ...(report || { reason: 'node:sqlite is unavailable, so verified matches could not be read before reconstruction.' }),
    retrieval: index ? { method: index.method, dimension: index.dimension } : { method: 'none' },
    pairs: { intra: intra.size, cross: tried.size - intra.size, strategy: exhaustive ? 'exhaustive' : 'retrieval' },
    matcher: learned ? 'ALIKED + LightGlue' : 'SIFT',
    recoveryAttempts: attempts,
  };
  await writeJson(connectivityPath(work), summary);
  if (report && !report.fullyConnected) {
    const names = new Map(videos.map((v) => [v.id, v.label]));
    logger.warn(`Disconnected after all attempts: ${report.disconnected.map((id) => names.get(id)).join(', ')} share no reliable overlap with the main group and will not be forced into it.`);
  } else if (report) {
    logger.ok(`All ${videos.length} videos are connected (${report.verifiedCrossPairs} verified cross-video pairs)`);
  }
  tracker.complete('connectivity');
  return summary;
}

/** Record which videos actually produced usable frames. */
export async function saveInputStatus(work, analysed) {
  const manifest = await readInputs(work);
  const counts = new Map(analysed.map((a) => [a.video.id, a.candidates.filter((c) => c.selected).length]));
  for (const video of manifest.videos) video.selectedFrames = counts.get(video.id) ?? 0;
  await writeInputs(work, manifest.videos);
}

export async function ensureMultiVideoDirs(work) {
  await mkdir(join(work, 'manifests'), { recursive: true });
  await mkdir(join(work, 'inputs'), { recursive: true });
}

export async function frameFileCount(frames) {
  return (await readdir(frames)).filter((name) => name.endsWith('.jpg')).length;
}
