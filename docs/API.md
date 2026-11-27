# Skyforge HTTP API

The server (`server.mjs`, default `http://localhost:8787`) speaks JSON. Job and
upload ids are validated before they touch the filesystem. Errors are
`{ "error": "...", "stale"?: true, "reason"?: "..." }`.

## Reconstruction

| Method | Path | Notes |
|---|---|---|
| POST | `/api/reconstruct` | **Single video** (unchanged): raw body, headers `X-File-Name`, `X-Quality`, `X-Altitude`, `X-Frames`, `X-Threads`, `X-Splat`, `X-Dsm`, `X-Mask-Dynamic`. Optional `X-Enhance`: URI-encoded JSON, see below. |
| POST | `/api/reconstruct` | **Multi-video**: `Content-Type: application/json`, body `{ "uploadId", "options" }`. |
| GET | `/api/reconstruct/:id` | Job status. |
| GET | `/api/reconstruct/:id/stream` | Server-sent events: `progress`, `log`, `done`. |
| GET | `/api/reconstruct/:id/log?since=n` | Log polling fallback. |
| POST | `/api/reconstruct/:id/cancel` | Cancel a running job. |
| POST | `/api/reconstruct/:id/turntable` | Render the turntable video. |
| GET/HEAD | `/api/reconstruct/:id/file?kind=` | `glb splat terrain full cloud mesh ortho meta video poster report original repaired removed repair-report confidence-glb confidence-ply confidence-bin confidence-report lod-report` |
| GET | `/api/reconstruct/:id/files` | Which kinds exist, with sizes. |
| GET | `/api/jobs`, `/api/health`, `/api/telemetry` | |

### `options` (multi-video) and `X-Enhance`

```json
{
  "quality": "fast | balanced | detailed",
  "altitude": 82,
  "frames": 300,
  "threads": 8,
  "splat": false,
  "dsm": false,
  "maskDynamic": false,
  "matching": "auto | exhaustive",
  "separateComponents": true,
  "enhance": {
    "analysis": true,
    "repair": false,
    "confidence": true,
    "lod": true,
    "repairSettings": {},
    "confidenceSettings": {},
    "lodSettings": {}
  }
}
```

`frames` is the total frame budget across all videos (24 to 2000). `X-Enhance`
carries the `enhance` object. Without it a single-video job runs exactly the
original pipeline.

## Staged uploads

| Method | Path | Notes |
|---|---|---|
| POST | `/api/uploads` | Create a session. Returns `{id, videos, limits, estimate}`. |
| GET | `/api/uploads/:id?quality=&frames=` | Session and requirement estimate. |
| PATCH | `/api/uploads/:id` | `{ "order": ["video_002", …], "settings": { "video_001": { … } } }` |
| DELETE | `/api/uploads/:id` | Discard the session and its files. |
| POST | `/api/uploads/:id/videos[?replace=video_001]` | Raw body, header `X-File-Name` (URI-encoded). 409 duplicate content, 413 over a limit, 415 not a video, 507 low disk. |
| DELETE | `/api/uploads/:id/videos/:videoId` | Remove one video. |
| GET | `/api/uploads/:id/videos/:videoId/thumbnail` | JPEG. |

Per-video settings: `label`, `altitudeM`, `cameraModel`
(`SIMPLE_RADIAL RADIAL OPENCV OPENCV_FISHEYE SIMPLE_PINHOLE`),
`horizontalFovDeg`, `weight` (0.25 to 4), `trimStart`, `trimEnd`, `enabled`,
`cameraName`, `flightNotes`.

Sessions idle for 24 h are deleted (`SKYFORGE_UPLOAD_TTL_HOURS`).

## Job products

| Method | Path | Returns |
|---|---|---|
| GET | `/api/reconstruct/:id/views` | Camera positions along the flight (viewer coordinates) and the filmed area. |
| GET | `/api/reconstruct/:id/videos` | Source videos with frame and registration figures. |
| GET | `/api/reconstruct/:id/videos/:videoId/thumbnail` | JPEG. |
| GET | `/api/reconstruct/:id/connectivity` | Verified overlap graph, registration, altitude check. |
| GET | `/api/reconstruct/:id/frames` | `frames.json`. |
| GET | `/api/reconstruct/:id/artifacts` | Which derived products exist and are current, the active model, compression support. |
| POST | `/api/reconstruct/:id/active-model` | `{ "model": "original" \| "repaired" }` |
| GET | `/api/reconstruct/:id/mesh-quality` | Analysis; `202 {task}` if it had to be started. |
| POST | `/api/reconstruct/:id/mesh-quality` | Re-run analysis. |
| POST | `/api/reconstruct/:id/repair` | Body: repair settings. `202 {task}`. |
| GET | `/api/reconstruct/:id/repair` | Repaired GLB. |
| GET | `/api/reconstruct/:id/repair-report` | Repair statistics. |
| GET | `/api/reconstruct/:id/provenance` | Binary `SFPV`. |
| POST | `/api/reconstruct/:id/lod/generate` | Body: LOD settings. `202 {task}`. |
| GET | `/api/reconstruct/:id/lod` | Manifest. |
| GET | `/api/reconstruct/:id/lod/:level` | GLB. |
| GET | `/api/reconstruct/:id/lod/:level/confidence`, `/provenance` | Per-face data for that level. |
| POST | `/api/reconstruct/:id/confidence/analyze` | Body: confidence settings. `202 {task}`. |
| GET | `/api/reconstruct/:id/confidence[?model=]` | Summary and schema. |
| GET | `/api/reconstruct/:id/confidence/regions` | Low-confidence regions. |
| GET | `/api/reconstruct/:id/confidence/data` | Binary `SFCF`. |
| GET | `/api/reconstruct/:id/confidence/coverage` | Camera and video coverage. |
| GET | `/api/reconstruct/:id/tasks`, `/tasks/stream` | The enhancement task; SSE. |
| POST | `/api/reconstruct/:id/tasks/cancel` | Cancel it. |
| GET | `/api/reconstruct/:id/components/:n/model` | A separately reconstructed component. |

A product whose source files have changed answers `409` with `stale: true`
instead of being served.

### Settings objects

**Repair:** `preset` (`conservative balanced aggressive`), `removeFloating`,
`floatingAggressiveness`, `removeDuplicates`, `repairNonManifold`,
`fixOrientation`, `fillHoles`, `maxHoleSize`, `maxHoleUnit` (`m percent`),
`holeStrategy` (`auto planar boundary curvature`), `preserveOpenings`,
`smoothing` (`off light moderate strong custom`), `smoothingIterations`,
`edgePreservation` (0 to 1), `preserveTextures`.

**Confidence:** `weights` (`coverage diversity reprojection support quality
depth`), `lowThreshold`, `minAvailability`, `resolution`, `exportPly`.

**LOD:** `levels` (`[{level, ratio, texture, enabled}]`), `textureFormat`
(`auto jpeg png webp`), `textureQuality`, `quantize`, `errorScale`.

## Binary schemas

All little-endian, indexed by face in the order of the matching GLB: primitives
in node order, faces in index order. Files written by Skyforge also carry
`primitive.extras.skyforgeFaceOffset`.

**`SFCF` confidence** — 16-byte header: `"SFCF"`, u16 version, u16 record size
(16), u32 face count, u32 flags. Then one record per face:

| Offset | Type | Field |
|---|---|---|
| 0 | u8 | score 0–100, 255 unknown |
| 1 | u8 | evidence availability, percent |
| 2–7 | u8 ×6 | coverage, diversity, reprojection, support, quality, depth (0–100, 255 unavailable) |
| 8 | u16 | supporting cameras, 65535 unavailable |
| 10 | u16 | median reprojection error ×100 px, 65535 unavailable |
| 12 | u8 | distinct supporting videos |
| 13 | u8 | provenance code |
| 14 | u8 | mean viewing angle off the normal, degrees |
| 15 | u8 | angular spread of viewing rays, degrees |

If flags bit 0 is set, `u32 videoMask[faceCount]` follows; bit *i* is the *i*-th
id in `confidence.json` → `binary.videoMaskBits`.

**`SFPV` provenance** — `"SFPV"`, u16 version, u16 reserved, u32 face count,
`u8 code[faceCount]` padded to 4 bytes, `i32 sourceFace[faceCount]` (face of the
source mesh, −1 for generated). Codes: 0 original, 1 cleaned, 2 smoothed,
3 interpolated, 4 hole filled, 5 unsupported.
