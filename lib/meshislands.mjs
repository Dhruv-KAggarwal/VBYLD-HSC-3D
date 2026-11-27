/**
 * Drop disconnected islands from a triangle mesh.
 *
 * OpenMVS leaves a small number of free-floating fragments behind: reprojection
 * blobs up in the sky, specks of triangulated cloud, shards under the terrain.
 * `ReconstructMesh --remove-spurious` does not catch them, and they are easy to
 * overlook because they are a rounding error by face count -- on a measured
 * 127-frame scene, 582 faces out of 499,933 (0.12%) spread over 17 islands.
 *
 * They are worth removing anyway, for parity rather than for geometry. The
 * vertex-colour export already strips them, and on the scene above this pass
 * finds exactly the same 774 faces that path reports, so leaving them in means
 * the textured model ships with sky blobs floating over it that the
 * vertex-coloured model of the same mesh does not have.
 *
 * What this does NOT do is change the model's bounding box. That was the
 * original reason for writing it, and the reason was wrong: the measurement
 * said the box was 1364 m tall against an expected 233 m, but applying the
 * scene transform to all 446,592 vertices by hand gives 233.4 m, matching the
 * vertex-coloured export exactly. The 1364 m came from a faulty measurement
 * script, not from the mesh. These islands sit inside the scene envelope and
 * removing them moves no extent at all -- verified before and after on the same
 * file. Do not re-justify this function on bounding boxes.
 *
 * The vertex-colour path already cleans this up in `cleanColoredMesh`, but its
 * first pass classifies sky by vertex colour, which a UV-textured mesh does not
 * carry. This is that function's second pass on its own, and on the measured
 * scene the topology pass alone was sufficient -- it found the same 774 faces
 * the two-pass version does. Weld coincident
 * vertices (so UV seams are not mistaken for topological breaks), union-find the
 * faces, and keep only components at or above `minFaceRatio` of the largest one.
 *
 * @param {{positions: Float32Array, indices: Uint32Array}} mesh
 * @param {{minFaceRatio?: number}} [options] fraction of the largest component a
 *   component must reach to survive; the default keeps anything within 1%.
 * @returns {{positions: Float32Array, indices: Uint32Array, removedFaces: number,
 *   removedVertices: number, islands: number, components: number}}
 */
export function dropMeshIslands({ positions, indices }, options = {}) {
  const minFaceRatio = options.minFaceRatio ?? 0.01;
  const vertexCount = positions.length / 3;
  const faceCount = indices.length / 3;
  if (!faceCount) {
    return {
      positions, indices, removedFaces: 0, removedVertices: 0, islands: 0, components: 0,
    };
  }

  // Weld by exact position first. A textured mesh splits a vertex once per UV
  // island, so the same corner appears several times; without the weld every
  // texture seam reads as a topological break and the mesh shatters into
  // thousands of "components".
  const byPosition = new Map();
  const parent = new Int32Array(vertexCount);
  for (let i = 0; i < vertexCount; i += 1) {
    const key = `${positions[i * 3]},${positions[i * 3 + 1]},${positions[i * 3 + 2]}`;
    const seen = byPosition.get(key);
    if (seen === undefined) { byPosition.set(key, i); parent[i] = i; } else { parent[i] = seen; }
  }

  const find = (start) => {
    let x = start;
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };
  for (let f = 0; f < indices.length; f += 3) {
    union(indices[f], indices[f + 1]);
    union(indices[f + 1], indices[f + 2]);
  }

  const facesPerRoot = new Map();
  for (let f = 0; f < indices.length; f += 3) {
    const root = find(indices[f]);
    facesPerRoot.set(root, (facesPerRoot.get(root) || 0) + 1);
  }
  let largest = 0;
  for (const n of facesPerRoot.values()) if (n > largest) largest = n;
  const threshold = Math.max(1, Math.floor(largest * minFaceRatio));

  const keepRoot = new Set();
  for (const [root, n] of facesPerRoot) if (n >= threshold) keepRoot.add(root);
  if (keepRoot.size === facesPerRoot.size) {
    return {
      positions,
      indices,
      removedFaces: 0,
      removedVertices: 0,
      islands: 0,
      components: facesPerRoot.size,
    };
  }

  // Compact the vertices as well as the faces. Leaving orphans behind would
  // defeat the whole point: three.js computes a bounding box from the POSITION
  // attribute, so an unreferenced vertex 1.3 km up still blows out the box.
  const remap = new Int32Array(vertexCount).fill(-1);
  const keptIndices = new Uint32Array(indices.length);
  let writeIndex = 0;
  let nextVertex = 0;
  for (let f = 0; f < indices.length; f += 3) {
    if (!keepRoot.has(find(indices[f]))) continue;
    for (let c = 0; c < 3; c += 1) {
      const original = indices[f + c];
      if (remap[original] === -1) { remap[original] = nextVertex; nextVertex += 1; }
      keptIndices[writeIndex] = remap[original];
      writeIndex += 1;
    }
  }

  const outPositions = new Float32Array(nextVertex * 3);
  for (let i = 0; i < vertexCount; i += 1) {
    const target = remap[i];
    if (target === -1) continue;
    outPositions[target * 3] = positions[i * 3];
    outPositions[target * 3 + 1] = positions[i * 3 + 1];
    outPositions[target * 3 + 2] = positions[i * 3 + 2];
  }

  return {
    positions: outPositions,
    indices: keptIndices.subarray(0, writeIndex),
    removedFaces: faceCount - writeIndex / 3,
    removedVertices: vertexCount - nextVertex,
    islands: facesPerRoot.size - keepRoot.size,
    components: facesPerRoot.size,
  };
}
