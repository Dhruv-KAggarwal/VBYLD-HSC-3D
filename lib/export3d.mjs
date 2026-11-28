import { existsSync, statSync } from 'node:fs';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { parsePly, writeBinaryPly } from './ply.mjs';
import { writeGlb } from './glb.mjs';
import { readCameraPoses, readPoints3D } from './colmap.mjs';
import { estimateUp } from './gravity.mjs';
import { bakeOrthophoto } from './texture.mjs';
import { buildTerrain, sampleTerrainHeight, trimFootprint } from './terrain.mjs';
import { resolveAltitudeScale } from './multivideo/registration.mjs';
import { videoOfImage } from './multivideo/manifest.mjs';
import {
  applyRotation,
  boundsOf,
  centroid,
  classifyCoverage,
  compactCloud,
  densityFilter,
  pcaUp,
  rotationToYUp,
  scalePositions,
  subsampleCloud,
} from './geometry.mjs';

function usableCloud(path) {
  return path && existsSync(path) && statSync(path).size > 1024;
}

function rotatePoint(point, rotation, origin) {
  const x = point[0] - origin[0];
  const y = point[1] - origin[1];
  const z = point[2] - origin[2];
  return [
    rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z,
    rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z,
    rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z,
  ];
}

function medianOf(values) {
  if (!values.length) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/**
 * Metric scale without GPS: the reconstruction is in arbitrary units, so we
 * measure how far the solved cameras fly above the reconstructed ground and
 * stretch that to the operator's stated altitude.
 */
function scaleFromFlightAltitude(grid, cameraCentres, altitude) {
  if (!cameraCentres.length || !(altitude > 0)) return null;
  const clearances = [];
  for (const centre of cameraCentres) {
    const ground = sampleTerrainHeight(grid, centre[0], centre[2]);
    if (!Number.isFinite(ground)) continue;
    const clearance = centre[1] - ground;
    if (clearance > 1e-4) clearances.push(clearance);
  }
  if (clearances.length < 3) return null;
  const median = medianOf(clearances);
  if (!(median > 1e-4)) return null;
  return { scale: altitude / median, clearance: median, samples: clearances.length };
}

/**
 * Height of each camera above the ground it was looking at, from the sparse
 * landmarks its photograph contains.
 *
 * The ground directly beneath a forward-looking drone is often never filmed, so
 * reading it there depends on guesswork: on one multi-video job it put the same
 * flight 60% off. What a camera saw is measured. Needs a level frame.
 *
 * @returns {Array<{video:string|null, clearance:number}>}
 */
async function landmarkClearances(workDir, poses, rotation, origin) {
  const points = workDir ? await readPoints3D(workDir) : null;
  if (!points?.tracks) return [];
  const heights = new Map(poses.map((pose) => [pose.imageId, []]));
  for (let i = 0; i < points.count; i += 1) {
    const p = [points.positions[i * 3], points.positions[i * 3 + 1], points.positions[i * 3 + 2]];
    const y = rotatePoint(p, rotation, origin)[1];
    for (let k = points.trackOffsets[i]; k < points.trackOffsets[i + 1]; k += 1) heights.get(points.tracks[k])?.push(y);
  }
  const out = [];
  for (const pose of poses) {
    const seen = heights.get(pose.imageId);
    if (!seen || seen.length < 30) continue;
    // Low in the view, but above the stray points that fall below the surface.
    seen.sort((a, b) => a - b);
    const clearance = rotatePoint(pose.center, rotation, origin)[1] - seen[Math.floor(seen.length * 0.15)];
    if (clearance > 1e-4) out.push({ video: videoOfImage(pose.name), clearance });
  }
  return out;
}

/** Scale that puts the median camera at the stated altitude, or null. */
function scaleFromClearances(clearances, altitude) {
  if (clearances.length < 3 || !(altitude > 0)) return null;
  const median = medianOf(clearances.map((c) => c.clearance));
  return { scale: altitude / median, clearance: median, samples: clearances.length };
}

/**
 * Derive just the gravity rotation, metric scale and centroid — no DSM.
 *
 * The 3D path only ever needed `transform` out of buildProducts, but getting it
 * meant running the whole 2.5D export: a full-resolution height grid, slope
 * limiting, pyramid fill, hillshading and an orthophoto bake, ~2m29s of
 * single-threaded JavaScript on every job, for a product most runs never ship.
 *
 * Everything expensive in that chain exists to make the DSM *look* right.
 * Nothing here needs that. The scale comes from the median height of the solved
 * cameras above the ground beneath them, and a median over dozens of cameras is
 * insensitive to grid resolution — so a coarse grid with no cosmetic passes
 * gives the same number for a fraction of the cost.
 *
 * @returns {{ rotation: number[][], scale: number, origin: number[], scaleSource: string, clearance: number|null, cameraPoses: number }}
 */
/**
 * Write the point-cloud and metadata deliverables without building a DSM.
 *
 * Oriented and scaled with the same transform the mesh uses, so the cloud and
 * the model overlay exactly.
 */
export async function writeCloudProducts({
  cloudPath,
  outDir,
  transform,
  altitude = 82,
  sourceName = 'single-pass.mp4',
  maxPoints = 400000,
  extra = {},
}) {
  await mkdir(outDir, { recursive: true });
  const parsed = parsePly(await readFile(cloudPath));
  const cloud = compactCloud(parsed.positions, parsed.colors, parsed.normals);
  const oriented = applyRotation(cloud.positions, transform.rotation, transform.origin);
  const scaled = scalePositions(oriented, transform.scale);
  const view = subsampleCloud(scaled, cloud.colors, maxPoints) || { positions: scaled, colors: cloud.colors };
  const box = boundsOf(scaled);

  const files = {
    pointCloud: join(outDir, 'pointcloud.ply'),
    metadata: join(outDir, 'metadata.json'),
  };
  await writeFile(files.pointCloud, writeBinaryPly(view.positions, view.colors, null));

  const metadata = {
    source: sourceName,
    generatedAt: new Date().toISOString(),
    points: scaled.length / 3,
    exportedPoints: view.positions.length / 3,
    altitudeM: altitude,
    metersPerUnit: 1,
    extentM: [box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]],
    bbox: box,
    layers: classifyCoverage(scaled, cloud.colors),
    crs: 'local tangent plane, Y-up, metres (altitude-scaled)',
    orientation: transform.orientation,
    cameraPoses: transform.cameraPoses,
    scaleSource: transform.scaleSource,
    scaleApplied: transform.scale,
    ...extra,
    notes: [
      transform.altitudeReport
        ? `Gravity from ${transform.cameraPoses} solved cameras. ${transform.altitudeReport.message}`
        : transform.scaleSource === 'flight-altitude'
        ? `Gravity from ${transform.cameraPoses} solved cameras; scale set so the flight sits ${altitude} m above the reconstructed ground.`
        : `Gravity from ${transform.cameraPoses} solved cameras; scale is a scene-height heuristic because flight clearance could not be measured, so distances are relative, not survey-grade.`,
      'No GPS was available, so the model is georeferenced only to a local tangent plane.',
    ].join(' '),
  };
  await writeFile(files.metadata, JSON.stringify(metadata, null, 2));
  return { files, metadata };
}

export async function computeSceneTransform({
  cloudPath, meshPath, workDir = null, altitude = 82, videos = null, cameraOnly = false,
}) {
  const sourceCloud = usableCloud(cloudPath) ? cloudPath : meshPath;
  if (!usableCloud(sourceCloud)) throw new Error('Point cloud is empty or missing.');

  const parsed = parsePly(await readFile(sourceCloud));
  if (!parsed.vertexCount) throw new Error('Point cloud contains no vertices.');
  const cloud = compactCloud(parsed.positions, parsed.colors, parsed.normals);
  if (cloud.vertexCount < 64) throw new Error('Point cloud contains too few valid coordinates.');

  // The undistorted model is always the one that was densified; sparse/0 is not
  // necessarily it when the mapper split the footage into several models.
  const poses = workDir ? await readCameraPoses(workDir, { preferDense: true }) : [];
  // `cameraOnly` reproduces the frame of jobs exported before surfaces were used.
  const gravity = estimateUp(poses, cameraOnly || !parsed.normals ? null : cloud.normals);
  const rotation = rotationToYUp(gravity.up || pcaUp(cloud.positions));
  const origin = centroid(cloud.positions);

  const aligned = applyRotation(cloud.positions, rotation, origin);
  const trimmed = trimFootprint(aligned, cloud.colors);
  const cameraCentres = poses.map((pose) => rotatePoint(pose.center, rotation, origin));

  // Coarse grid, cosmetic passes off. 192 cells across is ample to read ground
  // height under a camera and costs a small fraction of the export grid.
  const coarse = buildTerrain(trimmed.positions, trimmed.colors, {
    resolution: 192,
    smoothing: 1,
    slope: 8,
    cameraXZ: cameraCentres.map((centre) => [centre[0], centre[2]]),
  });

  const looked = cameraOnly ? [] : await landmarkClearances(workDir, poses, rotation, origin);
  const fit = scaleFromClearances(looked, altitude) || scaleFromFlightAltitude(coarse.grid, cameraCentres, altitude);
  let altitudeReport = null;
  if (videos?.length) {
    const clearances = looked.length >= 3 ? looked : [];
    if (!clearances.length) {
      cameraCentres.forEach((centre, i) => {
        const ground = sampleTerrainHeight(coarse.grid, centre[0], centre[2]);
        const clearance = centre[1] - ground;
        if (Number.isFinite(clearance) && clearance > 1e-4) {
          clearances.push({ video: videoOfImage(poses[i].name), clearance });
        }
      });
    }
    altitudeReport = resolveAltitudeScale(clearances, videos, altitude);
  }
  let scale;
  if (altitudeReport) {
    scale = altitudeReport.scale;
  } else if (fit) {
    scale = fit.scale;
  } else if (altitude > 0) {
    const box = boundsOf(trimmed.positions);
    const fallbackHeight = Math.max(1e-6, box.max[1] - box.min[1]);
    scale = altitude / Math.max(fallbackHeight * 0.45, 1e-6);
  } else {
    scale = 1;
  }

  return {
    rotation,
    scale,
    origin,
    scaleSource: altitudeReport?.scaleSource || (fit ? 'flight-altitude' : 'scene-height-heuristic'),
    altitudeReport,
    clearance: fit ? fit.clearance : null,
    cameraPoses: poses.length,
    orientation: gravity.up ? gravity.method : 'point-cloud-pca',
  };
}

export async function buildProducts({
  cloudPath,
  meshPath,
  outDir,
  workDir = null,
  altitude = 82,
  sourceName = 'single-pass.mp4',
  mode = 'view',
  surface = null,
}) {
  await mkdir(outDir, { recursive: true });
  const profiles = {
    fast: { resolution: 520, smoothing: 2, slope: 3.2, ortho: 1536, points: 180000 },
    balanced: { resolution: 720, smoothing: 2, slope: 4.5, ortho: 1792, points: 280000 },
    detailed: { resolution: 960, smoothing: 1, slope: 7, ortho: 2048, points: 400000 },
    full: { resolution: 960, smoothing: 1, slope: 7, ortho: 2048, points: 400000 },
    view: { resolution: 0, smoothing: 4, slope: 2, ortho: 1536, points: 240000 },
  };
  const profile = surface || profiles[mode] || profiles.view;
  const sourceCloud = usableCloud(cloudPath) ? cloudPath : meshPath;
  if (!usableCloud(sourceCloud)) throw new Error('Point cloud is empty or missing.');
  console.log(`Export ${mode}: cloud=${basename(sourceCloud)} grid=${profile.resolution || 'auto'} slope=${profile.slope}`);

  const parsed = parsePly(await readFile(sourceCloud));
  if (!parsed.vertexCount) throw new Error('Point cloud contains no vertices.');
  let cloud = compactCloud(parsed.positions, parsed.colors, parsed.normals);
  if (cloud.vertexCount < 64) throw new Error('Point cloud contains too few valid coordinates.');
  // Taken before denoising, which keeps positions and colours only.
  const normals = parsed.normals ? cloud.normals : null;
  const denoised = densityFilter(cloud.positions, cloud.colors, 96, 3);
  cloud.positions = denoised.positions;
  cloud.colors = denoised.colors;
  cloud.vertexCount = denoised.positions.length / 3;

  const poses = workDir ? await readCameraPoses(workDir, { preferDense: true }) : [];
  const gravity = estimateUp(poses, normals);
  const orientation = gravity.up ? gravity.method : 'point-cloud-pca';
  const rotation = rotationToYUp(gravity.up || pcaUp(cloud.positions));
  const origin = centroid(cloud.positions);

  const aligned = applyRotation(cloud.positions, rotation, origin);
  const trimmed = trimFootprint(aligned, cloud.colors);
  const unscaledBox = boundsOf(trimmed.positions);

  const cameraCentres = poses.map((pose) => rotatePoint(pose.center, rotation, origin));
  const draft = buildTerrain(trimmed.positions, trimmed.colors, {
    resolution: profile.resolution,
    smoothing: profile.smoothing,
    slope: profile.slope,
    cameraXZ: cameraCentres.map((centre) => [centre[0], centre[2]]),
  });

  const baked = await bakeOrthophoto({
    workDir,
    grid: draft.grid,
    albedo: draft.albedo,
    rotation,
    origin,
    maxEdge: profile.ortho,
  });
  if (baked.cameras) {
    console.log(`Orthophoto: ${baked.width}x${baked.height} from ${baked.cameras} frames, ${(baked.coverage * 100).toFixed(0)}% photo coverage`);
  }

  const fit = scaleFromClearances(await landmarkClearances(workDir, poses, rotation, origin), altitude)
    || scaleFromFlightAltitude(draft.grid, cameraCentres, altitude);
  const fallbackHeight = Math.max(1e-6, unscaledBox.max[1] - unscaledBox.min[1]);
  const scale = fit ? fit.scale : (altitude > 0 ? altitude / Math.max(fallbackHeight * 0.45, 1e-6) : 1);

  const scaledCloud = scalePositions(trimmed.positions, scale);
  const mesh = {
    positions: scalePositions(draft.positions, scale),
    // Preserve real dense-cloud albedo as a vertex-colour fallback. Hillshade is
    // presentation shading, not photographic colour, and was wrong for PLY/GLB exports.
    colors: draft.albedo || draft.colors,
    uvs: draft.uvs,
    indices: draft.indices,
    ortho: baked.png,
    textured: baked.coverage > 0.05,
  };

  const viewCloud = subsampleCloud(scaledCloud, trimmed.colors, profile.points);
  console.log(`Terrain: ${mesh.positions.length / 3} verts, ${mesh.indices.length / 3} faces, coverage ${(draft.coverage * 100).toFixed(0)}%`);
  const glb = writeGlb(mesh, viewCloud);

  const scaledBox = boundsOf(mesh.positions);
  const metadata = {
    source: sourceName,
    generatedAt: new Date().toISOString(),
    points: trimmed.positions.length / 3,
    meshVertices: mesh.positions.length / 3,
    meshFaces: mesh.indices.length / 3,
    altitudeM: altitude,
    metersPerUnit: 1,
    extentM: [
      scaledBox.max[0] - scaledBox.min[0],
      scaledBox.max[1] - scaledBox.min[1],
      scaledBox.max[2] - scaledBox.min[2],
    ],
    bbox: scaledBox,
    layers: classifyCoverage(scaledCloud, trimmed.colors),
    crs: 'local tangent plane, Y-up, metres (altitude-scaled)',
    orientation,
    cameraPoses: poses.length,
    scaleSource: fit ? 'flight-altitude' : 'scene-height-heuristic',
    groundSampleM: (scaledBox.max[0] - scaledBox.min[0]) / Math.max(1, draft.grid.cols - 1),
    coverage: draft.coverage,
    photoCoverage: baked.coverage,
    textureSize: [baked.width, baked.height],
    notes: [
      fit
        ? `Gravity from ${poses.length} solved cameras; scale set so the flight sits ${altitude} m above the reconstructed ground.`
        : poses.length
          ? `Gravity from ${poses.length} solved cameras; scale is a scene-height heuristic because flight clearance could not be measured.`
          : 'Gravity estimated from the point cloud; scale is a heuristic because camera poses were unavailable.',
      baked.cameras
        ? `Orthophoto baked from ${baked.cameras} undistorted frames (${(baked.coverage * 100).toFixed(0)}% of the surface has photographic colour).`
        : 'No source frames were available to bake a photographic texture.',
    ].join(' '),
    unscaledExtent: [
      unscaledBox.max[0] - unscaledBox.min[0],
      unscaledBox.max[1] - unscaledBox.min[1],
      unscaledBox.max[2] - unscaledBox.min[2],
    ],
    scaleApplied: scale,
    exportMode: mode,
  };

  const files = {
    glb: join(outDir, 'terrain.glb'),
    mesh: join(outDir, 'terrain.ply'),
    pointCloud: join(outDir, 'pointcloud.ply'),
    ortho: join(outDir, 'orthomosaic.png'),
    metadata: join(outDir, 'metadata.json'),
  };
  await writeFile(files.glb, glb);
  await writeFile(files.ortho, mesh.ortho);
  await writeFile(files.metadata, JSON.stringify(metadata, null, 2));
  await writeFile(files.mesh, writeBinaryPly(mesh.positions, mesh.colors, mesh.indices));
  const cloudOut = viewCloud || subsampleCloud(scaledCloud, trimmed.colors, 600000);
  await writeFile(files.pointCloud, writeBinaryPly(cloudOut.positions, cloudOut.colors, null));
  return {
    files,
    metadata,
    sourceCloud: basename(cloudPath),
    // The gravity rotation, metric scale and centroid this export derived. A mesh
    // built by a separate engine lives in COLMAP's arbitrary world frame, so it has
    // to be stood upright and sized by exactly the same transform or the mesh and
    // the DSM will not overlay.
    transform: { rotation, scale, origin },
  };
}
