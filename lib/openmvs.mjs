/**
 * OpenMVS high-detail textured-mesh stage.
 * ---------------------------------------------------------------------------
 * COLMAP gives us *camera poses* (where each frame was shot). That is enough
 * for the 2.5-D digital surface model in `terrain.mjs`, but a drone operator
 * who asks for "a detailed 3-D texture" wants the real thing: a watertight
 * triangle mesh whose every face carries photographic colour, viewable from
 * any angle, not a height-field seen from above.
 *
 * OpenMVS is the open-source engine that turns COLMAP's undistorted workspace
 * into exactly that. This module drives its four relevant tools in sequence:
 *
 *     InterfaceCOLMAP    COLMAP dense/ workspace            ->  scene.mvs
 *     DensifyPointCloud  scene.mvs  (GPU / CUDA depth maps) ->  scene_dense.{mvs,ply}
 *     ReconstructMesh    dense cloud -> Delaunay + graph-cut ->  mesh.ply
 *     RefineMesh*        photo-consistent vertex optimisation ->  mesh_refined.ply   (*optional)
 *     TextureMesh        per-face texture atlas  -> GLB      ->  model.glb
 *
 * Every tool is a CUDA-aware native binary; the 8 GB RTX 4060 on the reference
 * machine does the depth-map and texturing passes on the GPU while the i7's 28
 * threads carry mesh reconstruction. The knobs that trade detail for VRAM and
 * time (resolution level, max resolution, point distance) come in through the
 * `mvs` block of `qualitySettings`, so one profile change retunes the whole
 * chain.
 *
 * The module is deliberately dependency-injected: it never imports `run` or
 * `commands` from `reconstruct.mjs` (that would be a circular import). The
 * caller passes the spawn helper in, which also means every OpenMVS process
 * inherits the same progress tracking, cancellation and timeout handling the
 * COLMAP stages already use.
 */

import { existsSync, statSync, readdirSync, readFileSync, openSync, readSync, closeSync } from 'node:fs';
import { join } from 'node:path';

/**
 * Face count from a PLY header, without reading the body.
 *
 * A dense mesh PLY is hundreds of megabytes; the only thing needed to size the
 * texture-stage decimation is one integer in the first few hundred bytes.
 * Returns 0 when the header cannot be read, which callers treat as "do not
 * decimate" rather than guessing.
 */
export function plyFaceCount(path) {
  let fd = null;
  try {
    fd = openSync(path, 'r');
    const head = Buffer.alloc(2048);
    const read = readSync(fd, head, 0, head.length, 0);
    const text = head.subarray(0, read).toString('latin1');
    return Number((text.match(/element face\s+(\d+)/) || [])[1] || 0);
  } catch {
    return 0;
  } finally {
    if (fd !== null) try { closeSync(fd); } catch { /* already gone */ }
  }
}

/**
 * True when `output` exists, is non-trivial, and is newer than every input.
 *
 * Used only when stage reuse is explicitly requested: re-running a job with
 * different tuning must not silently serve the previous run's artefacts, so
 * the caller gates this behind an opt-in flag.
 */
function upToDate(output, inputs = [], minBytes = 1024) {
  if (!produced(output, minBytes)) return false;
  let newest = 0;
  for (const input of inputs) {
    if (!input || !existsSync(input)) continue;
    newest = Math.max(newest, statSync(input).mtimeMs);
  }
  return statSync(output).mtimeMs >= newest;
}

/** Resolve the five OpenMVS executables under an install folder. */
export function openmvsCommands(base) {
  return {
    interface: join(base, 'InterfaceCOLMAP.exe'),
    densify: join(base, 'DensifyPointCloud.exe'),
    reconstructMesh: join(base, 'ReconstructMesh.exe'),
    refineMesh: join(base, 'RefineMesh.exe'),
    textureMesh: join(base, 'TextureMesh.exe'),
  };
}

/**
 * Is the textured-mesh path available on this machine? We only require the
 * tools the pipeline actually runs; RefineMesh is optional, so a build that
 * ships without it still counts as available.
 */
export function openmvsAvailable(base) {
  if (!base) return false;
  const c = openmvsCommands(base);
  return ['interface', 'densify', 'reconstructMesh', 'textureMesh'].every((key) => existsSync(c[key]));
}

/**
 * Only a USER CANCELLATION may abort the whole run.
 *
 * A timeout must not. The pipeline used to test `/cancelled|timeout/` and rethrow
 * for both, which meant a stage that merely ran long took the entire job down even
 * when its fallback was already sitting on disk. That is exactly how a 156-frame
 * run died with no product at all: ReconstructMesh *finished* computing a good
 * 8.3 M-face mesh, got killed a few seconds later while serialising it, and the
 * rethrow discarded a perfectly usable 474 MB dense cloud instead of falling back
 * to it. Timeouts are how a legitimately slow stage ends on a laptop GPU — they
 * are a signal to degrade, not to abort.
 */
export const isCancellation = (error) => /cancell?ed/i.test(error?.message || '');

const sizeOf = (path) => {
  try { return existsSync(path) ? statSync(path).size : 0; } catch { return 0; }
};

/** A product file only counts if it exists and is more than a stub header. */
const produced = (path, minBytes = 2048) => sizeOf(path) >= minBytes;

/**
 * Did TextureMesh bake a real photographic atlas, or a blank one?
 *
 * The atlas path used to be disabled outright because a thin aerial pass can leave
 * most faces without a clean view, so they bake to flat "empty colour" and the model
 * renders blank — visually far worse than vertex colours. Rather than give the atlas
 * up (it is the ONLY way to get 4K-frame detail onto the surface, independent of mesh
 * density), we keep it and sanity-check the result.
 *
 * The check is deliberately crude and dependency-free: a photographic atlas is noisy
 * and compresses poorly (megabytes at 4–8 K), while a flat/empty one collapses to a
 * few KB of PNG. If the baked texture is implausibly small we reject it and fall back
 * to vertex colours. Note the failure mode is SAFE: a wrong rejection just lands on
 * the previous, known-good behaviour — it can never produce something worse.
 */
/** Read width/height straight out of a PNG IHDR or a JPEG SOFn marker. */
function imageDimensions(bytes) {
  if (bytes.length > 24 && bytes[0] === 0x89 && bytes[1] === 0x50) {
    return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
  }
  if (bytes.length > 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let i = 2;
    while (i + 9 < bytes.length) {
      if (bytes[i] !== 0xff) { i += 1; continue; }
      const marker = bytes[i + 1];
      // SOF0..SOF15, skipping the non-frame markers in that range.
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { height: bytes.readUInt16BE(i + 5), width: bytes.readUInt16BE(i + 7) };
      }
      i += 2 + bytes.readUInt16BE(i + 2);
    }
  }
  return null;
}

/** Pull every embedded image out of a GLB without decoding the geometry. */
function glbImages(glbPath) {
  const out = [];
  try {
    const buf = readFileSync(glbPath);
    if (buf.length < 20 || buf.readUInt32LE(0) !== 0x46546c67) return out;
    let offset = 12;
    let json = null;
    let bin = null;
    while (offset + 8 <= buf.length) {
      const len = buf.readUInt32LE(offset);
      const type = buf.readUInt32LE(offset + 4);
      const body = buf.subarray(offset + 8, offset + 8 + len);
      if (type === 0x4e4f534a) json = JSON.parse(body.toString('utf8'));
      else if (type === 0x004e4942) bin = body;
      offset += 8 + len + ((4 - (len % 4)) % 4);
    }
    if (!json?.images || !bin) return out;
    for (const image of json.images) {
      if (image.bufferView == null) continue;
      const view = json.bufferViews[image.bufferView];
      if (!view) continue;
      const start = view.byteOffset || 0;
      out.push(bin.subarray(start, start + view.byteLength));
    }
  } catch { /* unreadable or malformed -> no images, caller decides */ }
  return out;
}

/**
 * Decide whether a baked atlas actually carries photographic detail.
 *
 * The old version weighed sidecar files against a flat 128 KB floor and, when
 * OpenMVS embedded the texture in the GLB instead (which `--export-type glb`
 * does), returned ok unconditionally — so the check never ran on the path that
 * matters. It also had no notion of atlas size: 128 KB is plausible for a 1K
 * atlas and obviously blank for an 8K one.
 *
 * This version reads the real image bytes, from the GLB or from sidecars, and
 * judges compressed bytes per pixel. Photographic texture is high-entropy and
 * never compresses below roughly 0.02 B/px; a flat or empty-colour atlas is
 * near-uniform and collapses far below that whatever its dimensions.
 *
 * The failure mode stays safe: a wrong rejection only falls back to vertex
 * colours, which is the previous known-good behaviour.
 */
function atlasLooksPhotographic(dir, glbPath, minBytesPerPixel = 0.02) {
  const images = [];
  for (const bytes of glbImages(glbPath)) images.push({ bytes, source: 'glb' });
  if (!images.length) {
    try {
      for (const name of readdirSync(dir)) {
        if (!/\.(png|jpg|jpeg)$/i.test(name)) continue;
        images.push({ bytes: readFileSync(join(dir, name)), source: name });
      }
    } catch { /* unreadable dir */ }
  }
  if (!images.length) return { ok: false, reason: 'no texture image found', textureBytes: 0, found: 0 };

  let textureBytes = 0;
  let pixels = 0;
  for (const image of images) {
    textureBytes += image.bytes.length;
    const dims = imageDimensions(image.bytes);
    if (dims) pixels += dims.width * dims.height;
  }
  if (!pixels) {
    // Dimensions unreadable: fall back to an absolute floor rather than guessing.
    return { ok: textureBytes >= 128 * 1024, textureBytes, found: images.length, bytesPerPixel: null };
  }
  const bytesPerPixel = textureBytes / pixels;
  return {
    ok: bytesPerPixel >= minBytesPerPixel,
    textureBytes,
    found: images.length,
    pixels,
    bytesPerPixel,
  };
}

/**
 * Runs the full OpenMVS densify -> mesh -> (refine) -> texture chain.
 *
 * All OpenMVS tools run with their working directory set to `mvsDir`, so the
 * `*.dmap` depth maps, temporary archives and the engine's own timestamped log
 * stay in one disposable folder. Image paths inside `scene.mvs` are resolved
 * against the undistorted `dense/images` folder, which never moves.
 *
 * @param {object}   ctx
 * @param {Function} ctx.run       spawn helper from reconstruct.mjs (handles
 *                                 progress, cancellation, timeout, engine.log)
 * @param {object}   ctx.commands  OpenMVS binary paths (from openmvsCommands)
 * @param {string}   ctx.dense     COLMAP undistorted workspace (has images/, sparse/)
 * @param {string}   ctx.mvsDir    scratch folder for OpenMVS artefacts
 * @param {object}   ctx.job       the live job (for cancel hooks / stage labels)
 * @param {object}   ctx.tracker   progress tracker (stage transitions)
 * @param {object}   ctx.mvs       tuning knobs from qualitySettings().mvs
 * @param {object}   ctx.logger    structured logger
 * @param {number}   ctx.imageCount number of undistorted frames (for progress totals)
 * @returns {Promise<{ modelGlb: string|null, denseCloud: string|null, meshPly: string|null, refined: boolean }>}
 */
export async function buildTexturedMesh({
  run, commands, dense, mvsDir, job, tracker, mvs, logger, imageCount = 0,
}) {
  const scene = join(mvsDir, 'scene.mvs');
  const denseScene = join(mvsDir, 'scene_dense.mvs');
  const denseCloud = join(mvsDir, 'scene_dense.ply');
  const meshPly = join(mvsDir, 'mesh.ply');
  const refinedPly = join(mvsDir, 'mesh_refined.ply');
  const modelGlb = join(mvsDir, 'model.glb');

  const images = join(dense, 'images');
  // Long ceilings: dense depth-map estimation and mesh refinement are the two
  // stages that can legitimately run for many minutes on a large flight.
  const DENSIFY_TIMEOUT = 90 * 60 * 1000;
  // ReconstructMesh is a single-threaded CPU Delaunay + graph-cut whose cost grows
  // super-linearly with the dense-cloud size. On a long clip (150+ frames → 6 M+
  // points → 38 M+ tetrahedra) the graph-cut alone can run ~30 min and the whole
  // mesh 45 min+ BEFORE the multi-hundred-MB .ply is even serialised. A 45-min
  // ceiling killed such a run *at the finish line* (mesh computed, not yet written)
  // → no mesh.ply, whole pipeline errors out. 120 min gives genuine headroom so a
  // slow-but-healthy mesh is never aborted just short of completion. (Speed is
  // addressed separately upstream by thinning the cloud via minPointDistance.)
  const MESH_TIMEOUT = 120 * 60 * 1000;

  // ---- 1. Import COLMAP -> OpenMVS scene --------------------------------
  logger?.step('OpenMVS: importing COLMAP workspace into scene.mvs');
  await run(commands.interface, [
    '-i', dense,
    '-o', scene,
    '--image-folder', images,
    '--max-threads', String(mvs.threads),
    '-v', '2',
  ], mvsDir, job, 'Importing COLMAP scene (OpenMVS)', { stage: 'densify', total: imageCount });
  if (!produced(scene, 256)) throw new Error('InterfaceCOLMAP did not produce a scene.mvs.');

  // ---- 2. Dense depth maps + fusion on the GPU --------------------------
  // Stage reuse is opt-in (mvs.reuse). Depth-map estimation is the second most
  // expensive stage, so skipping it when re-running a job to re-texture or
  // re-mesh saves minutes — but only an explicit request can make that safe,
  // because the artefact on disk was produced with the PREVIOUS run's tuning
  // and nothing about the file records which knobs produced it.
  const reusedDense = mvs.reuse && upToDate(denseCloud, [scene]) && upToDate(denseScene, [scene], 256);
  if (reusedDense) {
    logger?.ok('OpenMVS: reusing the existing dense cloud (newer than scene.mvs, reuse requested)');
    tracker?.complete('densify');
  } else {
  logger?.step(`OpenMVS: densifying on GPU (resolution-level ${mvs.resolutionLevel}, max ${mvs.maxResolution}px)`);
  await run(commands.densify, [
    scene,
    '-o', denseScene,
    '--resolution-level', String(mvs.resolutionLevel),
    '--max-resolution', String(mvs.maxResolution),
    '--min-resolution', String(mvs.minResolution),
    '--number-views', String(mvs.numberViews),
    '--number-views-fuse', String(mvs.numberViewsFuse),
    '--sub-resolution-levels', String(mvs.subResolutionLevels),
    '--iters', String(mvs.densifyIters),
    '--geometric-iters', String(mvs.geometricIters),
    '--fusion-mode', '0',
    '--estimate-colors', '2',
    '--estimate-normals', '2',
    // Depth maps are 100s of MB per frame-pair and only fusion consumes them.
    '--remove-dmaps', '1',
    // OpenMVS defaults tower-mode to 4, which injects a synthetic cylinder of
    // points down the middle of the region of interest. That exists for
    // scans of a single tall object on a turntable; on an aerial pass over a
    // town it fabricates a column of geometry through the scene centre and
    // costs real time in fusion and then again in the Delaunay. Off.
    '--tower-mode', String(mvs.towerMode ?? 0),
    '--cuda-device', String(mvs.cudaDevice),
    '--max-threads', String(mvs.threads),
    '-v', '2',
  ], mvsDir, job, 'Dense depth maps on GPU (OpenMVS)', {
    stage: 'densify',
    total: imageCount,
    timeoutMs: DENSIFY_TIMEOUT,
  });
  if (!produced(denseCloud)) throw new Error('DensifyPointCloud produced no dense point cloud.');
  tracker?.complete('densify');
  }

  // ---- 3. Surface reconstruction (Delaunay + graph-cut) -----------------
  if (mvs.reuse && upToDate(meshPly, [denseCloud, denseScene])) {
    logger?.ok('OpenMVS: reusing the existing mesh (newer than the dense cloud, reuse requested)');
  } else {
  logger?.step(
    `OpenMVS: reconstructing surface (min-point-distance ${mvs.minPointDistance}`
    + `${mvs.targetFaceNum ? `, target ${mvs.targetFaceNum.toLocaleString()} faces` : ''})`,
  );
  await run(commands.reconstructMesh, [
    '-i', denseScene,
    '-p', denseCloud,
    '-o', meshPly,
    '--min-point-distance', String(mvs.minPointDistance),
    '--smooth', String(mvs.meshSmooth),
    '--remove-spurious', String(mvs.removeSpurious),
    '--remove-spikes', '1',
    '--close-holes', String(mvs.closeHoles),
    '--decimate', String(mvs.meshDecimate),
    // Cap the face count here rather than letting the graph-cut emit 868k
    // faces and paying for them twice — once in view assignment and again in
    // atlas packing, which together were 68% of a measured run. 0 disables.
    ...(mvs.targetFaceNum ? ['--target-face-num', String(mvs.targetFaceNum)] : []),
    '--cuda-device', String(mvs.cudaDevice),
    '--max-threads', String(mvs.threads),
    '-v', '2',
  ], mvsDir, job, 'Reconstructing surface mesh (OpenMVS)', {
    stage: 'mesh',
    total: imageCount,
    timeoutMs: MESH_TIMEOUT,
  });
  if (!produced(meshPly)) throw new Error('ReconstructMesh produced no mesh.');
  }

  // ---- Optional photo-consistent refinement (shared by BOTH output paths) ----
  // RefineMesh carves real geometric relief (roof/facade edges) into the surface by
  // optimising it against the photographs — the strongest lever for "it looks like
  // a 3D model, not a draped photo". It is also the slowest, most VRAM-hungry stage
  // and the first to fall over on an 8 GB card, so it is OPT-IN (mvs.refine, set via
  // SKYFORGE_MESH_REFINE upstream) and any failure is downgraded to a warning — the
  // run continues with the unrefined mesh. Running it HERE (before the vertex-colour
  // early-return) means the default colour path benefits from the sharper geometry
  // too, not just the atlas path.
  let meshForTexture = meshPly;
  let refined = false;
  if (mvs.refine) {
    try {
      logger?.step(`OpenMVS: refining mesh against photographs (scales ${mvs.refineScales})`);
      await run(commands.refineMesh, [
        '-i', denseScene,
        '-m', meshPly,
        '-o', refinedPly,
        '--resolution-level', String(mvs.refineResolutionLevel),
        '--max-views', String(mvs.refineMaxViews),
        '--scales', String(mvs.refineScales),
        '--reduce-memory', '1',
        '--close-holes', String(mvs.closeHoles),
        '--cuda-device', String(mvs.refineCudaDevice),
        '--max-threads', String(mvs.threads),
        '-v', '2',
      ], mvsDir, job, 'Refining mesh against photographs (OpenMVS)', {
        stage: 'mesh',
        total: imageCount,
        timeoutMs: MESH_TIMEOUT,
      });
      if (produced(refinedPly)) {
        meshForTexture = refinedPly;
        refined = true;
        logger?.ok('OpenMVS: mesh refinement complete');
      } else {
        logger?.warn('OpenMVS: refinement produced no output; using the unrefined mesh.');
      }
    } catch (error) {
      // A refine timeout is not fatal either: the unrefined mesh is right here.
      if (isCancellation(error)) throw error;
      logger?.warn(`OpenMVS: mesh refinement failed (${error.message}); using the unrefined mesh.`);
    }
  }

  // Default path: stop here and let the caller vertex-colour this (dense,
  // un-decimated, possibly refined) mesh straight from the coloured point cloud.
  // OpenMVS's own TextureMesh atlas is unreliable on a single forward aerial pass —
  // most faces see no clean view and bake to flat "empty-colour", so the model
  // renders blank. Vertex colours from the cloud never fail that way. The atlas path
  // is kept behind `mvs.textureAtlas` for scenes with good multi-view coverage.
  if (!mvs.textureAtlas) {
    logger?.ok(`OpenMVS mesh ready${refined ? ' (photo-refined)' : ''}; colouring from the dense cloud (atlas skipped)`);
    return {
      modelGlb: null,
      denseCloud: produced(denseCloud) ? denseCloud : null,
      meshPly: produced(meshForTexture) ? meshForTexture : null,
      refined,
    };
  }

  // ---- 5. Texture atlas -> GLB -----------------------------------------
  // This is where real photographic detail (roof tiles, road markings, signage)
  // gets onto the surface at a resolution the MESH ITSELF does not have to carry —
  // which is why we can keep the mesh light/fast and still look sharp.
  //
  // The whole stage is best-effort: on any failure, an empty-looking bake, or a
  // timeout, we return the mesh WITHOUT a GLB and the caller vertex-colours it from
  // the dense cloud exactly as before. The atlas can only ever be an upgrade.
  try {
    // How much of the mesh to throw away before texturing.
    //
    // TextureMesh's two phases both scale with face count: assigning a source
    // view to every face, then packing the resulting patches into the atlas.
    // On a measured run those were 6m36s and 14m59s over 868k faces and 36k
    // patches - 68% of the whole pipeline. Texture detail comes from the
    // ATLAS resolution, not the triangle count, so a mesh decimated to a few
    // hundred thousand faces textures several times faster and looks much the
    // same. 1 is "disabled"; the factor is a keep-ratio, so it is computed
    // from the mesh actually on disk rather than guessed per profile.
    let decimate = 1;
    if (mvs.textureTargetFaces) {
      const faces = plyFaceCount(meshForTexture);
      if (faces > mvs.textureTargetFaces) {
        // Floor at 0.1: below that the surface stops being recognisable.
        decimate = Math.max(0.1, Number((mvs.textureTargetFaces / faces).toFixed(4)));
        logger?.info(
          `OpenMVS: decimating ${faces.toLocaleString()} faces to about `
          + `${mvs.textureTargetFaces.toLocaleString()} for texturing (factor ${decimate})`,
        );
      }
    }
    // Texture from scene.mvs, not scene_dense.mvs.
    //
    // DensifyPointCloud runs at --resolution-level 1, which halves every image
    // before matching AND writes that working resolution into the scene it
    // saves. So scene_dense.mvs describes 960x534 images even though
    // dense/images holds 1920x1068 ones, and texturing from it bakes the atlas
    // from a quarter of the pixels that were actually captured -- measured on a
    // 127-frame run, OpenMVS reported "0.49 MPixels/image" at the texture
    // stage. The result is a soft, muddy atlas for no reason: the downscale
    // exists to make depth-map search affordable, and texturing does no depth
    // search at all.
    //
    // scene.mvs carries the same 127 calibrated poses at full resolution, and
    // the mesh comes in separately via -m, so nothing here needs the dense
    // cloud. Falls back to the dense scene if the pre-densify scene is missing
    // (a reused run that only kept the dense products).
    const textureScene = produced(scene, 256) ? scene : denseScene;
    logger?.step(
      `OpenMVS: baking texture atlas up to ${mvs.maxTextureSize}px -> GLB`
      + `${textureScene === scene ? ' (full-resolution images)' : ' (dense scene; downscaled images)'}`,
    );
    await run(commands.textureMesh, [
      '-i', textureScene,
      '-m', meshForTexture,
      '-o', modelGlb,
      '--export-type', 'glb',
      '--resolution-level', String(mvs.textureResolutionLevel),
      '--max-texture-size', String(mvs.maxTextureSize),
      '--close-holes', String(mvs.closeHoles),
      ...(decimate < 1 ? ['--decimate', String(decimate)] : []),
      // Virtual faces are OFF, and must stay off until someone re-measures.
      //
      // The idea is to merge coplanar triangles sharing views into one "virtual
      // face" before patch generation, so a flat roof becomes one patch instead
      // of hundreds. It was introduced here as "the principled attack on the 36k
      // patches". Measured on a 127-frame scene, the same mesh both ways, it
      // does the exact opposite and corrupts the output as well:
      //
      //            virtual-face-images 3      virtual-face-images 0
      //   patches  149,665                    35,046
      //   atlases  2                          1
      //   UVs      -9.82 .. 14.53 (broken)    0.0006 .. 0.9998
      //   time     8m28s                      3m02s
      //
      // So it multiplied patches by 4.3x, spilled into a second atlas, and
      // generated out-of-range UVs for every one of the 120,947 texcoords in
      // that overflow primitive -- 15.5% of the mesh sampling nothing. Note the
      // UV damage is a multi-atlas effect, so it hides on scenes small enough
      // to fit one atlas and reappears on larger ones.
      '--virtual-face-images', String(mvs.virtualFaceImages ?? 0),
      // Compactness preference; see costSmoothnessRatio in reconstruct.mjs for
      // why this sits at the OpenMVS default rather than being tuned down.
      '--cost-smoothness-ratio', String(mvs.costSmoothnessRatio ?? 0.1),
      // Packing heuristic: 0 best fit, 3 good speed, 100 best speed. Packing
      // was the single largest cost in the run above, and the only thing a
      // looser heuristic costs is wasted atlas area.
      '--patch-packing-heuristic', String(mvs.patchPackingHeuristic ?? 3),
      // Faces no camera ever saw. The OpenMVS default is a loud orange; a mid
      // grey reads as honestly-unobserved instead of as a texturing error, and
      // never gets mistaken for a black render.
      '--empty-color', String(0x808080),
      // Seam levelling solves for a per-vertex colour correction that makes
      // neighbouring patches match where they meet. Without it every patch
      // keeps the exposure of whichever frame it was sampled from, and since
      // a drone's auto-exposure drifts across a flight, the model comes out a
      // visible patchwork of mismatched brightness. This is the single biggest
      // lever on how the texture actually looks.
      //
      // It was switched off on a hypothesis: a bake here had ~37% of faces
      // sampling solid black while the padding around those same patches held
      // correct photography, and a levelling solve returning extreme or
      // non-finite corrections would damage exactly the face interiors and
      // spare exactly the padding. Plausible, but wrong. The cause was
      // --virtual-face-images 3, which fragmented the atlas into 149,665
      // patches, spilled into a second texture, and emitted out-of-range UVs
      // for every texcoord in the overflow primitive; see the table at the
      // --virtual-face-images line above. With virtual faces off the bake is
      // a single atlas with UVs in [0,1] and no black-face defect at all.
      //
      // So levelling is back on, which is also the OpenMVS default. Set
      // SKYFORGE_SEAM_LEVELING=0 to disable it.
      '--global-seam-leveling', String(mvs.seamLeveling),
      '--local-seam-leveling', String(mvs.seamLeveling),
      '--cuda-device', String(mvs.textureCudaDevice ?? mvs.cudaDevice),
      '--max-threads', String(mvs.threads),
      '-v', '2',
    ], mvsDir, job, 'Baking texture atlas to GLB (OpenMVS)', {
      stage: 'texture',
      total: imageCount,
      timeoutMs: MESH_TIMEOUT,
    });
    if (!produced(modelGlb)) throw new Error('TextureMesh produced no GLB.');

    const atlas = atlasLooksPhotographic(mvsDir, modelGlb);
    if (!atlas.ok) {
      const density = atlas.bytesPerPixel == null
        ? 'dimensions unreadable'
        : `${atlas.bytesPerPixel.toFixed(4)} B/px over ${(atlas.pixels / 1e6).toFixed(1)} MPixels`;
      logger?.warn(
        `OpenMVS: the baked atlas looks blank (${(atlas.textureBytes / 1024).toFixed(0)} KB across `
        + `${atlas.found} image(s), ${density}); falling back to vertex colours from the dense cloud.`,
      );
    } else {
      logger?.ok(
        `OpenMVS: textured atlas baked (${(atlas.textureBytes / 1e6).toFixed(1)} MB, `
        + `${atlas.bytesPerPixel ? atlas.bytesPerPixel.toFixed(3) : '?'} B/px)`,
      );
      return {
        modelGlb,
        denseCloud: produced(denseCloud) ? denseCloud : null,
        meshPly: produced(meshForTexture) ? meshForTexture : null,
        refined,
      };
    }
  } catch (error) {
    if (isCancellation(error)) throw error;
    logger?.warn(`OpenMVS: texture atlas failed (${error.message}); falling back to vertex colours.`);
  }

  return {
    modelGlb: null,
    denseCloud: produced(denseCloud) ? denseCloud : null,
    meshPly: produced(meshForTexture) ? meshForTexture : null,
    refined,
  };
}
