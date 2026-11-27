# Place FFmpeg, COLMAP, and optional OpenMVS here

Skyforge looks for Windows binaries in this folder. **Do not commit the binaries** — they exceed GitHub’s 100 MB file limit (`ffmpeg.exe` is ~101 MB).

After unzipping, this tree must exist:

```
engines/
  ffmpeg/bin/ffmpeg.exe
  ffmpeg/bin/ffprobe.exe
  colmap/bin/colmap.exe
  openmvs/                    (optional)
```

## FFmpeg 8.x

1. Download **ffmpeg-release-essentials.zip** from https://www.gyan.dev/ffmpeg/builds/
2. Extract so `ffmpeg.exe` and `ffprobe.exe` sit in `engines/ffmpeg/bin/`

## COLMAP 4.2 CUDA (required)

CPU-only COLMAP will crawl. Use a **CUDA** build.

1. Open https://github.com/colmap/colmap/releases and download the Windows CUDA zip for **4.2.x**
2. Extract the archive
3. Copy the `bin` folder (and any sibling `lib` folder the zip includes) to `engines/colmap/`
4. Confirm `engines/colmap/bin/colmap.exe` exists
5. First launch may trigger Windows Defender — allow it

If `colmap mapper -h` does **not** list `--Mapper.ba_global_frames_ratio`, you have COLMAP 3.x. This project is written for 4.2 (`frames_ratio`, not `images_ratio`).

## OpenMVS 2.3.0 (needed for the textured 3D mesh)

1. Download **OpenMVS_Windows_x64.7z** (about 11 MB) from
   https://github.com/cdcseacave/openMVS/releases/tag/v2.3.0
2. Extract the `.exe` and `.dll` files into `engines/openmvs/` (Windows 11:
   `tar -xf OpenMVS_Windows_x64.7z`)
3. Optional check: `Get-FileHash OpenMVS_Windows_x64.7z` printed
   `E1B7AE31AD078160059005B1BFED10280C14E99EE2579F5F806CCAFF05C784CA` for the
   copy this project was verified with (the release publishes no checksum).

It uses CUDA through the NVIDIA driver, so no CUDA Toolkit is needed.

**Do not use the official 2.4.0 CUDA download** unless you have an RTX 40-series
card. It contains GPU code for that architecture only, and on any other GPU
densification stops with `CUDA error ... named symbol not found (code 500)`
(upstream issue 1315).

Without OpenMVS the pipeline falls back to COLMAP dense stereo and a
vertex-coloured Delaunay mesh: slower to mesh, and no texture atlas.

## Custom install locations

You can skip this folder and point at system installs:

```bash
export FFMPEG_PATH="/c/ffmpeg/bin/ffmpeg.exe"
export FFPROBE_PATH="/c/ffmpeg/bin/ffprobe.exe"
export COLMAP_PATH="/c/COLMAP/bin/colmap.exe"
```

On PowerShell:

```powershell
$env:FFMPEG_PATH = "C:\ffmpeg\bin\ffmpeg.exe"
$env:FFPROBE_PATH = "C:\ffmpeg\bin\ffprobe.exe"
$env:COLMAP_PATH = "C:\COLMAP\bin\colmap.exe"
```
