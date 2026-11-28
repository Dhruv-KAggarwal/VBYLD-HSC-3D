#!/usr/bin/env python3
"""
Photoreal branch: train a 3D Gaussian Splat from the COLMAP reconstruction.

Why this exists
---------------
A textured mesh is measurable but it is still a surface: it has one colour per
point on a triangle, so thin structures (railings, foliage, signage), glass and
grazing-angle detail all get smeared or dropped. A Gaussian splat stores the
scene as millions of oriented, semi-transparent blobs fitted directly to the
source photographs, so it reproduces exactly what the camera saw from any
viewpoint. That is what makes it possible to drop to street level and read the
front of a building, which is the whole point of this branch.

It shares the SAME COLMAP poses as the mesh branch, so there is no second solve.

This replaces an earlier version that shelled out to Nerfstudio's `ns-train`.
Nerfstudio is a large dependency tree that does not install cleanly on Windows
with current Python; gsplat on its own is a single prebuilt wheel. Everything
Nerfstudio was doing for us -- read COLMAP, train, export PLY -- is done here
directly, which is a few hundred lines and removes the dependency entirely.

Output is a standard 3DGS .ply (the de-facto interchange format), readable by
three.js GaussianSplatPLYLoader, SuperSplat, Spark and the rest.

Usage:
  python train_splat.py --work <runtime/job-id> --out <.../products/splat> \
      [--iters 15000] [--cap 900000] [--max-size 1600]
"""
import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy

# Progress goes to stdout unbuffered and line-oriented so the Node side can
# parse it and the operator sees movement rather than a frozen bar.
def emit(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# COLMAP model reader (binary, with a text fallback)
# ---------------------------------------------------------------------------

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read(fid, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fid.read(size))


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as fid:
        (count,) = _read(fid, "<Q")
        for _ in range(count):
            cam_id, model_id, width, height = _read(fid, "<iiQQ")
            name, num_params = CAMERA_MODELS[model_id]
            params = _read(fid, "<" + "d" * num_params)
            cameras[cam_id] = {"model": name, "width": width, "height": height, "params": np.array(params)}
    return cameras


def read_images_bin(path):
    images = {}
    with open(path, "rb") as fid:
        (count,) = _read(fid, "<Q")
        for _ in range(count):
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read(fid, "<idddddddi")
            name = b""
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name += char
            (num_points,) = _read(fid, "<Q")
            fid.read(24 * num_points)  # x, y, point3D_id per 2D point - unused here
            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": cam_id,
                "name": name.decode(),
            }
    return images


def read_points3d_bin(path):
    xyz, rgb = [], []
    with open(path, "rb") as fid:
        (count,) = _read(fid, "<Q")
        for _ in range(count):
            _pid, x, y, z, r, g, b, _err = _read(fid, "<QdddBBBd")
            (track_len,) = _read(fid, "<Q")
            fid.read(8 * track_len)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, dtype=np.float64), np.array(rgb, dtype=np.uint8)


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def find_sparse(work: Path) -> Path:
    """Prefer the undistorted dense workspace: its cameras are PINHOLE, so the
    images need no further rectification and the intrinsics are exact."""
    candidates = [work / "dense" / "sparse"]
    sparse_root = work / "sparse"
    if sparse_root.exists():
        candidates += sorted([p for p in sparse_root.iterdir() if p.is_dir()])
    for candidate in candidates:
        if (candidate / "cameras.bin").exists() or (candidate / "cameras.txt").exists():
            return candidate
    raise SystemExit(f"No COLMAP sparse model found under {work}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Views:
    """All frames held on the CPU as uint8, moved to the GPU one at a time.

    An 8 GB card cannot hold 150 full-resolution frames as float32 alongside a
    million Gaussians, and streaming one frame per step costs far less than the
    training step itself.
    """

    def __init__(self, work: Path, max_size: int):
        from PIL import Image

        sparse = find_sparse(work)
        emit(f"splat: reading COLMAP model from {sparse}")
        cameras = read_cameras_bin(sparse / "cameras.bin")
        images = read_images_bin(sparse / "images.bin")

        image_dir = work / "dense" / "images"
        if not image_dir.exists():
            image_dir = work / "frames"
        emit(f"splat: image source {image_dir}")

        self.frames = []
        skipped = 0
        for image in sorted(images.values(), key=lambda entry: entry["name"]):
            path = image_dir / image["name"]
            if not path.exists():
                skipped += 1
                continue
            camera = cameras[image["camera_id"]]
            pixels = Image.open(path).convert("RGB")

            scale = 1.0
            if max_size and max(pixels.size) > max_size:
                scale = max_size / max(pixels.size)
                pixels = pixels.resize(
                    (max(1, int(pixels.width * scale)), max(1, int(pixels.height * scale))),
                    Image.LANCZOS,
                )

            params = camera["params"]
            if camera["model"] in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:  # SIMPLE_* models share one focal length
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            sx = pixels.width / camera["width"]
            sy = pixels.height / camera["height"]
            K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float32)

            # COLMAP stores world-to-camera directly, which is what the
            # rasteriser wants, so no inversion is needed.
            viewmat = np.eye(4, dtype=np.float32)
            viewmat[:3, :3] = qvec_to_rotmat(image["qvec"])
            viewmat[:3, 3] = image["tvec"]

            self.frames.append({
                # np.array (not asarray): PIL returns a read-only buffer and
                # torch warns loudly about wrapping non-writable memory.
                "image": torch.from_numpy(np.array(pixels, dtype=np.uint8)),
                "K": torch.from_numpy(K),
                "viewmat": torch.from_numpy(viewmat),
                "name": image["name"],
            })

        if skipped:
            emit(f"splat: {skipped} registered frame(s) had no image file and were skipped")
        if len(self.frames) < 8:
            raise SystemExit(f"Only {len(self.frames)} usable frames; need at least 8 to train a splat.")

        points, colors = read_points3d_bin(sparse / "points3D.bin")
        self.points = points
        self.colors = colors
        emit(f"splat: {len(self.frames)} frames, {len(points):,} seed points")

    def __len__(self):
        return len(self.frames)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def knn_spacing(points: np.ndarray, k: int = 4) -> np.ndarray:
    """Mean distance to the k nearest neighbours, used to size the initial
    Gaussians. Chunked so a large cloud does not allocate an N x N matrix."""
    tensor = torch.from_numpy(points).float().cuda()
    out = torch.empty(len(points), device="cuda")
    # Each block is a chunk x N distance matrix: size it to about 1 GB so a large
    # cloud does not exhaust a 6 GB card before training starts.
    chunk = max(64, min(4096, int(2.5e8 // max(1, len(points)))))
    for start in range(0, len(points), chunk):
        block = tensor[start:start + chunk]
        dist = torch.cdist(block, tensor)
        nearest = dist.topk(k + 1, largest=False).values[:, 1:]
        out[start:start + chunk] = nearest.mean(dim=1)
    return out.clamp(min=1e-7).cpu().numpy()


def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    return (rgb / 255.0 - 0.5) / 0.28209479177387814


def sh0_to_rgb(sh0: np.ndarray) -> np.ndarray:
    return sh0 * 0.28209479177387814 + 0.5


def build_model(views: Views, cap: int, sh_degree: int):
    points = views.points
    colors = views.colors
    if len(points) > cap // 2:
        keep = np.random.choice(len(points), cap // 2, replace=False)
        points, colors = points[keep], colors[keep]

    spacing = knn_spacing(points)
    scales = np.log(spacing)[:, None].repeat(3, axis=1)

    quats = np.zeros((len(points), 4), dtype=np.float32)
    quats[:, 0] = 1.0
    # inverse sigmoid of 0.1: start mostly transparent so the optimiser decides
    # what is real rather than starting with an opaque fog.
    opacities = np.full(len(points), math.log(0.1 / 0.9), dtype=np.float32)

    num_sh = (sh_degree + 1) ** 2
    sh = np.zeros((len(points), num_sh, 3), dtype=np.float32)
    sh[:, 0, :] = rgb_to_sh0(colors.astype(np.float32))

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(points).float().cuda()),
        "scales": torch.nn.Parameter(torch.from_numpy(scales).float().cuda()),
        "quats": torch.nn.Parameter(torch.from_numpy(quats).cuda()),
        "opacities": torch.nn.Parameter(torch.from_numpy(opacities).cuda()),
        "sh0": torch.nn.Parameter(torch.from_numpy(sh[:, :1]).cuda()),
        "shN": torch.nn.Parameter(torch.from_numpy(sh[:, 1:]).cuda()),
    }).cuda()

    # Scene scale sets the learning rate for positions; without it a town-sized
    # scene learns far too slowly and a room-sized one diverges.
    centre = points.mean(axis=0)
    scene_scale = float(np.linalg.norm(points - centre, axis=1).mean())

    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}], eps=1e-15)
        for name, lr in lrs.items()
    }
    return params, optimizers, scene_scale


def render(params, viewmat, K, width, height, sh_degree, background):
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    return rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=True,
        absgrad=False,
        rasterize_mode="antialiased",
        backgrounds=None,
    )


# ---------------------------------------------------------------------------
# PLY export (standard 3DGS layout)
# ---------------------------------------------------------------------------

def write_ply(path: Path, params):
    means = params["means"].detach().cpu().numpy()
    scales = params["scales"].detach().cpu().numpy()
    quats = F.normalize(params["quats"].detach(), dim=-1).cpu().numpy()
    opacities = params["opacities"].detach().cpu().numpy().reshape(-1, 1)
    sh0 = params["sh0"].detach().cpu().numpy()          # (N, 1, 3)
    shN = params["shN"].detach().cpu().numpy()          # (N, M, 3)

    count = len(means)
    # The reference 3DGS format stores f_rest channel-major: all M coefficients
    # of red, then of green, then of blue. Writing it interleaved loads without
    # error but tints the model, so the transpose matters.
    rest = shN.transpose(0, 2, 1).reshape(count, -1)

    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {count}\n"
    header += "".join(f"property float {name}\n" for name in names)
    header += "end_header\n"

    data = np.concatenate([
        means,
        np.zeros((count, 3), dtype=np.float32),  # normals: unused, kept for compatibility
        sh0.reshape(count, 3),
        rest,
        opacities,
        scales,
        quats,
    ], axis=1).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fid:
        fid.write(header.encode())
        fid.write(data.tobytes())
    return count, path.stat().st_size


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, help="runtime/<job-id>")
    parser.add_argument("--out", required=True, help="output folder for splat.ply")
    parser.add_argument("--iters", type=int, default=15000)
    parser.add_argument("--cap", type=int, default=900000, help="maximum Gaussians (VRAM bound)")
    parser.add_argument("--max-size", type=int, default=1600, help="longest training image edge")
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; splat training needs the GPU.")

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    emit(f"splat: {name} ({vram_gb:.1f} GB VRAM), gsplat MCMC, cap {args.cap:,}")

    work = Path(args.work)
    out = Path(args.out)
    views = Views(work, args.max_size)
    params, optimizers, scene_scale = build_model(views, args.cap, args.sh_degree)
    emit(f"splat: initialised {len(params['means']):,} Gaussians, scene scale {scene_scale:.2f}")

    strategy = MCMCStrategy(cap_max=args.cap, refine_stop_iter=int(args.iters * 0.8), verbose=False)
    state = strategy.initialize_state()
    background = torch.zeros(1, 3, device="cuda")

    # Spherical-harmonic degree is raised gradually: fitting view-dependent
    # colour before the geometry has settled bakes errors into the appearance.
    sh_interval = max(1, args.iters // (args.sh_degree + 1))

    started = time.time()
    order = np.random.permutation(len(views))
    cursor = 0
    last_report = 0.0

    for step in range(args.iters):
        if cursor >= len(order):
            order = np.random.permutation(len(views))
            cursor = 0
        frame = views.frames[order[cursor]]
        cursor += 1

        target = frame["image"].cuda(non_blocking=True).float() / 255.0
        height, width = target.shape[:2]
        sh_degree = min(args.sh_degree, step // sh_interval)

        rendered, _alpha, info = render(
            params, frame["viewmat"].cuda(), frame["K"].cuda(),
            width, height, sh_degree, background,
        )
        image = rendered[0].clamp(0, 1)

        l1 = F.l1_loss(image, target)
        # A little SSIM-like structural term via local means keeps edges crisp
        # without pulling in an extra dependency for a full SSIM.
        blur_pred = F.avg_pool2d(image.permute(2, 0, 1)[None], 11, 1, 5)
        blur_true = F.avg_pool2d(target.permute(2, 0, 1)[None], 11, 1, 5)
        structure = F.l1_loss(image.permute(2, 0, 1)[None] - blur_pred, target.permute(2, 0, 1)[None] - blur_true)
        loss = 0.8 * l1 + 0.2 * structure
        # MCMC regularisers: keep Gaussians small and mostly transparent so the
        # cap is spent on detail rather than on a few huge blobs.
        loss = loss + 1e-2 * torch.sigmoid(params["opacities"]).mean()
        loss = loss + 1e-2 * torch.exp(params["scales"]).mean()

        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        strategy.step_post_backward(
            params, optimizers, state, step, info, lr=optimizers["means"].param_groups[0]["lr"],
        )

        now = time.time()
        if step % 50 == 0 or step == args.iters - 1:
            # Throttled so a fast GPU does not flood the log, but frequent
            # enough that the operator always sees movement.
            if now - last_report > 1.0 or step == args.iters - 1:
                last_report = now
                elapsed = now - started
                rate = (step + 1) / max(elapsed, 1e-6)
                eta = (args.iters - step - 1) / max(rate, 1e-6)
                emit(
                    f"splat: step {step + 1}/{args.iters} "
                    f"({100 * (step + 1) / args.iters:.1f}%) "
                    f"loss {loss.item():.4f} "
                    f"gaussians {len(params['means']):,} "
                    f"vram {torch.cuda.max_memory_allocated() / 1e9:.1f}GB "
                    f"eta {eta / 60:.1f}min"
                )

    ply_path = out / "splat.ply"
    count, size = write_ply(ply_path, params)
    elapsed = time.time() - started
    emit(f"splat: wrote {count:,} Gaussians to {ply_path} ({size / 1e6:.1f} MB) in {elapsed / 60:.1f} min")

    (out / "splat.json").write_text(json.dumps({
        "gaussians": count,
        "iterations": args.iters,
        "frames": len(views),
        "shDegree": args.sh_degree,
        "trainingSeconds": round(elapsed, 1),
        "maxImageEdge": args.max_size,
        "device": name,
        "format": "3DGS PLY (binary little endian)",
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        emit("splat: cancelled")
        sys.exit(130)
