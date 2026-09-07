# The Intelligence Age Reconstruction

A source-assisted, frame-by-frame reconstruction of the 60-second film
**ChatGPT | The Intelligence Age**, using Python, OpenCV, and FFmpeg.

This repository contains the original reconstruction pipeline only: contour
extraction, rendering, visual comparison, and audio verification.

## What this project does

The extractor decodes a reference frame and converts its grayscale levels into
integer-grid contour layers. Color frames use separate B, G, and R contour
planes. The renderer fills those contours to reconstruct the frame without
opening the reference video. It then encodes a lossless master and an MP4.

This is digital tracing (reference-assisted rotoscoping). The shapes and motion
come from the reference, one frame at a time. It is not an independently
animated remake, an AI video generation model, or a procedural system that
invents the original scenes. The original audio is reused, not synthesized.

The goal is exact pixel reconstruction of the **decoded reference**, using a
consistent OpenCV decoder. That does not mean access to the original production
files or studio master. The MP4 is lossy; only the lossless master is checked
for exact equality. Different decoder versions may produce different pixels.

## Requirements

- Python 3.14 (the development environment).
- FFmpeg and FFprobe on PATH, with FFV1, libx264, AAC, and PCM support.
- Python dependencies pinned in `requirements.txt`.
- A locally supplied reference video. The default configuration expects
  1920 x 1080, 24 fps, at least 1,440 frames, and an audio stream.
- At least 15 GB of free disk space for the default project. The measured
  contour assets alone occupied approximately 11 GB. Detailed per-frame tracing
  can take substantial time; try the synthetic smoke test first.

## Quick start

```sh
make setup
make test
make smoke

mkdir -p reference
cp /path/to/your/reference.mp4 reference/original-60s.mp4
make all
```

`make smoke` creates a tiny synthetic clip in a temporary directory, extracts
and renders it, compares every frame, checks audio, and verifies rendering
without the reference. It needs no third-party film and does not change your
project configuration.

The reference video, soundtrack, extracted geometry, and rendered films are
intentionally excluded from the source distribution. See
[Third-party materials](THIRD_PARTY_NOTICES.md).

## Pipeline

```sh
make extract    # Trace geometry and extract the reference audio.
make render     # Render from geometry and audio only.
make compare    # Check the lossless master and measure MP4 error.
make audio      # Compare decoded PCM audio byte for byte.
make review     # Generate a side-by-side review and error contact sheet.
make isolated   # Re-render in a separate project without the reference video.
```

`make isolated` replaces the current rendered files with the independently
rendered copies after successful rendering. Generated outputs are local and
ignored by Git.

For a single reconstructed frame:

```sh
.venv/bin/python scripts/pipeline.py inspect --frame 1332
```

To limit tracing concurrency:

```sh
.venv/bin/python scripts/pipeline.py extract --workers 2
```

To work on a shorter segment, edit `start_frame` and `frame_count` in
`project.json`, then rerun extraction. Keep dimensions and frame rate consistent
with your input. Rendering rejects stale geometry when these settings change.

## Files

| Path | Purpose |
| --- | --- |
| `project.json` | Reference path, frame range, dimensions, and output paths |
| `src/contours.py` | Contour extraction, storage, and rasterization |
| `scripts/pipeline.py` | Extract, render, compare, and inspect commands |
| `scripts/make_review.py` | Side-by-side review and error heatmaps |
| `scripts/check_audio.py` | PCM audio comparison |
| `scripts/verify_isolation.py` | Rendering without a reference video |
| `tests/test_contours.py` | Holes, edges, grayscale, and color round trips |

After rendering, the main outputs are:

- `renders/recreated-full60.mp4`: high-quality H.264/AAC delivery.
- `renders/recreated-full60-lossless.mkv`: FFV1/PCM lossless master.
- `reports/full60/metrics.json`: per-frame and aggregate visual metrics.
- `reports/full60/audio-check.json`: soundtrack verification.

## Geometry and validation

NPZ files contain vertices, contour offsets, layer offsets, sizes, and shades.
They do not contain image textures or encoded video frames. They nevertheless
encode the source image content and are not a way to remove source-media rights.

Geometry format 1 stores neutral grayscale frames. Format 2 adds separate color
planes; both are needed by the original full-film reconstruction. These are file
format versions, not alternative edits of the film.

Extraction immediately rasterizes each traced frame and requires exact BGR
pixel equality. After rendering, comparison checks all configured frames against
both the reference and extraction hashes. MP4 quality is reported separately
with channel error, PSNR, and grayscale SSIM.

The renderer relies on OpenCV's integer-grid even-odd contour filling. Keep the
pinned dependencies when reproducing results. The assets are large because the
pipeline prioritizes exact frame appearance over geometric simplification.

## Reference and license

- [Builders Club production page](https://builders-club.com/projects/openai-the-intelligence-age)
- [Production video on Vimeo](https://vimeo.com/1055260932/1225af8c69)

Code and documentation: [MIT](LICENSE).
Reference film, soundtrack, trademarks, and derived media: separate rights;
see [Third-party materials](THIRD_PARTY_NOTICES.md).
