#!/usr/bin/env python3
"""
Learned features + matching with hloc (SuperPoint + LightGlue), exported into a
COLMAP database that `colmap mapper` OR `glomap mapper` can consume directly.

SuperPoint + LightGlue is far more robust than COLMAP's SIFT on blurry, low-
texture, repetitive drone footage (SkyForge_Upgrade_Plan §3) — it is the single
biggest completeness/robustness win in the plan.

Pipeline: extract SuperPoint -> build sequential(+loop) pairs -> LightGlue match
-> create COLMAP db -> import images/features/matches -> geometric verification.
The result is `--database` with verified two-view geometries; run GLOMAP/COLMAP
mapper on it afterwards.

Usage:
  python hloc_match.py --images <frames_dir> --database <out.db> [--window 10] [--loop 0]

NOTE: hloc's module layout shifts between releases. The imports below match the
mainline `cvg/Hierarchical-Localization`; if your version differs, adjust the few
import lines at the top of run(). Everything else is version-stable.
"""
import argparse
from pathlib import Path


def build_sequential_pairs(names, window, loop_stride, out_path):
    """Each frame matched to the next `window` frames, plus optional loop links."""
    pairs = set()
    n = len(names)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + window, n)):
            pairs.add((names[i], names[j]))
    if loop_stride and loop_stride > 0:
        for i in range(0, n, loop_stride):
            for j in range(i + loop_stride, n, loop_stride):
                pairs.add((names[i], names[j]))
    with open(out_path, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")
    return len(pairs)


def run(images_dir, database_path, window, loop_stride):
    import pycolmap
    from hloc import extract_features, match_features
    from hloc.reconstruction import create_empty_db, import_images, get_image_ids
    from hloc.triangulation import (
        import_features,
        import_matches,
        estimation_and_geometric_verification,
    )

    images = Path(images_dir)
    database = Path(database_path)
    work = database.parent / "hloc"
    work.mkdir(parents=True, exist_ok=True)

    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]

    names = sorted([p.name for p in images.iterdir() if p.suffix.lower() in (".jpg", ".png")])
    print(f"[hloc] {len(names)} frames", flush=True)

    print("[hloc] extracting SuperPoint features...", flush=True)
    feature_path = extract_features.main(feature_conf, images, work)

    pairs_path = work / "pairs.txt"
    n_pairs = build_sequential_pairs(names, window, loop_stride, pairs_path)
    print(f"[hloc] {n_pairs} candidate pairs (window={window}, loop={loop_stride})", flush=True)

    print("[hloc] matching with LightGlue...", flush=True)
    match_path = match_features.main(matcher_conf, pairs_path, feature_conf["output"], work)

    if database.exists():
        database.unlink()
    print("[hloc] building COLMAP database + geometric verification...", flush=True)
    create_empty_db(database)
    import_images(images, database, pycolmap.CameraMode.SINGLE)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, feature_path)
    import_matches(image_ids, database, pairs_path, match_path,
                   min_match_score=None, skip_geometric_verification=True)
    estimation_and_geometric_verification(database, pairs_path, verbose=False)
    print(f"[hloc] done -> {database}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument("--window", type=int, default=10, help="match each frame to the next N")
    ap.add_argument("--loop", type=int, default=0, help="loop-closure stride (0 = off)")
    args = ap.parse_args()
    run(args.images, args.database, args.window, args.loop)


if __name__ == "__main__":
    main()
