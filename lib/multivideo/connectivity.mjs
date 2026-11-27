/**
 * Overlap and connectivity from geometrically verified correspondences.
 *
 * Nodes are videos (and images), edges are image pairs for which COLMAP's
 * RANSAC fitted a two-view geometry (essential, fundamental or homography).
 */

import { UnionFind } from '../meshdata.mjs';
import { videoOfImage } from './manifest.mjs';

// COLMAP TwoViewGeometry::ConfigurationType
const CONFIG = { CALIBRATED: 2, UNCALIBRATED: 3, PLANAR: 4, PANORAMIC: 5, PLANAR_OR_PANORAMIC: 6, WATERMARK: 7, MULTIPLE: 8 };
const ACCEPTED = new Set([2, 3, 4, 5, 6, 8]);

export const THRESHOLDS = Object.freeze({
  minInliers: 15,
  reliableInliers: 30,
  strong: { pairs: 8, framesPerSide: 3, inliers: 500 },
  weak: { pairs: 3, framesPerSide: 2 },
  suspectInlierRatio: 0.2,
});

export const isVerified = (g) => ACCEPTED.has(g.config) && g.inliers >= THRESHOLDS.minInliers;

const median = (values) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};

/** Group ids into components given `[a, b]` links. Largest first. */
export function componentsOf(ids, links, weight = () => 1) {
  const index = new Map(ids.map((id, i) => [id, i]));
  const sets = new UnionFind(ids.length);
  for (const [a, b] of links) sets.union(index.get(a), index.get(b));
  const groups = new Map();
  ids.forEach((id, i) => {
    const root = sets.find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(id);
  });
  const size = (group) => group.reduce((sum, id) => sum + weight(id), 0);
  return [...groups.values()].sort((a, b) => size(b) - size(a));
}

/**
 * @param {Map<number,{name:string}>} images  COLMAP images
 * @param {Array<{a:number,b:number,inliers:number,config:number,matches:number}>} geometries
 * @param {Array<{id:string,label:string,frames:Array}>} videos
 */
export function analyzeConnectivity(images, geometries, videos) {
  const edges = new Map();
  const intra = new Map(videos.map((v) => [v.id, 0]));
  const degree = new Map([...images.keys()].map((id) => [id, 0]));
  const imageLinks = [];
  let verifiedCross = 0;
  let crossMatches = 0;

  for (const g of geometries) {
    const a = images.get(g.a);
    const b = images.get(g.b);
    if (!a || !b) continue;
    let va = videoOfImage(a.name);
    let vb = videoOfImage(b.name);
    let [na, nb] = [a.name, b.name];
    if (va > vb) { [va, vb] = [vb, va]; [na, nb] = [nb, na]; }
    const cross = va !== vb;
    let edge = null;
    if (cross) {
      const key = `${va}|${vb}`;
      edge = edges.get(key);
      if (!edge) {
        edge = { a: va, b: vb, attemptedPairs: 0, verifiedPairs: 0, reliablePairs: 0, inlierMatches: 0, framesA: new Set(), framesB: new Set(), ratios: [], planar: 0, watermark: 0 };
        edges.set(key, edge);
      }
      edge.attemptedPairs += 1;
      if (g.config === CONFIG.WATERMARK) edge.watermark += 1;
    }
    if (!isVerified(g)) continue;
    const reliable = g.inliers >= THRESHOLDS.reliableInliers;
    if (reliable) {
      imageLinks.push([g.a, g.b]);
      degree.set(g.a, degree.get(g.a) + 1);
      degree.set(g.b, degree.get(g.b) + 1);
    }
    if (!cross) { intra.set(va, (intra.get(va) || 0) + 1); continue; }
    verifiedCross += 1;
    crossMatches += g.inliers;
    edge.verifiedPairs += 1;
    edge.inlierMatches += g.inliers;
    if (reliable) edge.reliablePairs += 1;
    edge.framesA.add(na);
    edge.framesB.add(nb);
    if (g.matches > 0) edge.ratios.push(Math.min(1, g.inliers / g.matches));
    if (g.config === CONFIG.PLANAR || g.config === CONFIG.PANORAMIC || g.config === CONFIG.PLANAR_OR_PANORAMIC) edge.planar += 1;
  }

  const edgeList = [...edges.values()].map((e) => {
    const side = Math.min(e.framesA.size, e.framesB.size);
    const { strong, weak } = THRESHOLDS;
    let strength = 'none';
    if (e.reliablePairs >= strong.pairs && side >= strong.framesPerSide && e.inlierMatches >= strong.inliers) strength = 'strong';
    else if (e.reliablePairs >= weak.pairs && side >= weak.framesPerSide) strength = 'weak';
    else if (e.verifiedPairs > 0) strength = 'insufficient';
    const ratio = median(e.ratios);
    const warnings = [];
    if (e.verifiedPairs && ratio != null && ratio < THRESHOLDS.suspectInlierRatio) {
      warnings.push('Low inlier ratio: possible false matches between similar-looking structures.');
    }
    // A homography-only link is also what two photographs of one flat, repeated
    // facade produce, so it deserves a second look.
    if (e.verifiedPairs >= 3 && e.planar / e.verifiedPairs > 0.85) {
      warnings.push('Overlap is almost entirely planar or rotation-only; verify the alignment visually.');
    }
    if (e.watermark) warnings.push('Some matches follow a static overlay (watermark or timestamp) and were rejected.');
    if (strength === 'insufficient') warnings.push('Too few verified pairs to treat these videos as overlapping.');
    return {
      a: e.a,
      b: e.b,
      strength,
      attemptedPairs: e.attemptedPairs,
      verifiedPairs: e.verifiedPairs,
      reliablePairs: e.reliablePairs,
      inlierMatches: e.inlierMatches,
      linkedFramesA: e.framesA.size,
      linkedFramesB: e.framesB.size,
      medianInlierRatio: ratio == null ? null : Math.round(ratio * 1000) / 1000,
      warnings,
    };
  }).sort((x, y) => y.inlierMatches - x.inlierMatches);

  const frameCount = new Map(videos.map((v) => [v.id, v.frames.length]));
  const components = componentsOf(
    videos.map((v) => v.id),
    edgeList.filter((e) => e.strength === 'strong' || e.strength === 'weak').map((e) => [e.a, e.b]),
    (id) => frameCount.get(id) || 0,
  );
  const imageIds = [...images.keys()];
  const imageComponents = componentsOf(imageIds, imageLinks);
  const total = imageIds.length || 1;

  return {
    method: 'COLMAP two-view geometric verification (RANSAC)',
    thresholds: THRESHOLDS,
    videos: videos.map((v) => {
      const own = imageIds.filter((id) => videoOfImage(images.get(id).name) === v.id);
      return {
        id: v.id,
        label: v.label,
        frames: v.frames.length,
        verifiedIntraPairs: intra.get(v.id) || 0,
        isolatedFrames: own.filter((id) => degree.get(id) === 0).length,
        weakFrames: own.filter((id) => degree.get(id) === 1).length,
        component: components.findIndex((c) => c.includes(v.id)) + 1,
      };
    }),
    edges: edgeList,
    components,
    disconnected: components.slice(1).flat(),
    fullyConnected: components.length <= 1,
    verifiedCrossPairs: verifiedCross,
    crossInlierMatches: crossMatches,
    imageGraph: {
      images: imageIds.length,
      components: imageComponents.length,
      largestComponentFraction: Math.round(((imageComponents[0]?.length || 0) / total) * 1000) / 1000,
      isolatedImages: imageIds.filter((id) => degree.get(id) === 0).length,
      weakImages: imageIds.filter((id) => degree.get(id) === 1).length,
    },
  };
}
