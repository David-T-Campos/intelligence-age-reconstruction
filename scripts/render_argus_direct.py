#!/usr/bin/env python3
"""Render the ARGUS Engineer edit directly from the public reference film.

This path intentionally avoids the very large contour archive. It preserves every
reference frame until the brand transform is applied, writes one high-quality MP4,
and then validates every decoded output frame against the deterministic target.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from argus_brand import _blue_fraction, apply_argus_brand


DEFAULT_OUTPUT = ROOT / "renders" / "argus-engineer-intelligence-age.mp4"
DEFAULT_REPORT = ROOT / "reports" / "argus-direct"


def _font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def probe(reference: Path):
    capture = cv2.VideoCapture(str(reference))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open reference video: {reference}")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width < 1 or height < 1 or fps <= 0 or count < 1:
        raise RuntimeError("Reference video metadata is invalid")
    return width, height, fps, count


def find_outro_start(reference: Path, fps: float, count: int) -> int:
    """Find the first sustained saturated-blue run in the final eight seconds."""
    scan = max(1, min(count, round(fps * 8)))
    first = count - scan
    capture = cv2.VideoCapture(str(reference))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    run = 0
    result = None
    for index in range(first, count):
        ok, frame = capture.read()
        if not ok:
            break
        if _blue_fraction(frame) >= 0.012:
            run += 1
            if run >= 3:
                result = index - run + 1
                break
        else:
            run = 0
    capture.release()
    return result if result is not None else max(0, count - round(fps * 3.5))


def render(reference: Path, output: Path, width: int, height: int,
           fps: float, count: int, outro_start: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", f"{fps:.8f}",
        "-i", "pipe:0", "-i", str(reference),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "slow", "-crf", "4",
        "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart",
        "-shortest", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    capture = cv2.VideoCapture(str(reference))
    rendered = 0
    try:
        for index in range(count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Reference stopped decoding at frame {index}")
            target = apply_argus_brand(frame, index, count, outro_start)
            process.stdin.write(target.tobytes())
            rendered += 1
            if rendered % 120 == 0 or rendered == count:
                print(f"render {rendered}/{count}", flush=True)
        process.stdin.close()
        process.stdin = None
        if process.wait() != 0:
            raise RuntimeError("ffmpeg render failed")
    except BaseException:
        if process.stdin is not None:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    finally:
        capture.release()


def _thumbnail(frame: np.ndarray, width: int = 320) -> np.ndarray:
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, max(1, round(frame.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


def _select_transitions(scores, limit=30, separation=8):
    chosen = []
    for score, index in sorted(scores, reverse=True):
        if all(abs(index - other) >= separation for other in chosen):
            chosen.append(index)
            if len(chosen) >= limit:
                break
    return sorted(chosen)


def _make_sheet(video: Path, indices, destination: Path, fps: float,
                columns: int = 6, thumb_width: int = 320):
    indices = sorted(set(int(i) for i in indices if i >= 0))
    if not indices:
        return
    capture = cv2.VideoCapture(str(video))
    cells = []
    font = _font(19)
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        thumb = _thumbnail(frame, thumb_width)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        label = f"{index:04d}  {index / fps:05.2f}s"
        box = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((7, 7, box[2] + 17, box[3] + 15), fill=(255, 255, 255))
        draw.text((12, 10), label, font=font, fill=(0, 0, 0))
        cells.append(np.asarray(image))
    capture.release()
    if not cells:
        return
    cell_h, cell_w = cells[0].shape[:2]
    rows = math.ceil(len(cells) / columns)
    canvas = np.full((rows * cell_h, columns * cell_w, 3), 255, dtype=np.uint8)
    for n, cell in enumerate(cells):
        row, col = divmod(n, columns)
        canvas[row * cell_h:(row + 1) * cell_h,
               col * cell_w:(col + 1) * cell_w] = cell
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 94])


def validate(reference: Path, output: Path, report_dir: Path, width: int,
             height: int, fps: float, count: int, outro_start: int) -> dict:
    source = cv2.VideoCapture(str(reference))
    delivery = cv2.VideoCapture(str(output))
    if not source.isOpened() or not delivery.isOpened():
        raise RuntimeError("Cannot open source or rendered video for validation")

    out_width = round(delivery.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = round(delivery.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = float(delivery.get(cv2.CAP_PROP_FPS))
    out_count = round(delivery.get(cv2.CAP_PROP_FRAME_COUNT))

    per_frame = []
    transition_scores = []
    previous_gray = None
    min_ssim = 1.0
    worst_index = 0
    sum_ssim = 0.0
    max_mae = 0.0

    for index in range(count):
        ok1, original = source.read()
        ok2, decoded = delivery.read()
        if not (ok1 and ok2):
            raise AssertionError(f"Missing decoded frame at {index}")
        expected = apply_argus_brand(original, index, count, outro_start)

        small_expected = cv2.resize(expected, (480, 270), interpolation=cv2.INTER_AREA)
        small_decoded = cv2.resize(decoded, (480, 270), interpolation=cv2.INTER_AREA)
        gray_expected = cv2.cvtColor(small_expected, cv2.COLOR_BGR2GRAY)
        gray_decoded = cv2.cvtColor(small_decoded, cv2.COLOR_BGR2GRAY)
        ssim = float(structural_similarity(gray_expected, gray_decoded, data_range=255))
        mae = float(np.abs(small_expected.astype(np.int16) -
                           small_decoded.astype(np.int16)).mean())
        min_ssim = min(min_ssim, ssim)
        if ssim <= min_ssim:
            worst_index = index
        sum_ssim += ssim
        max_mae = max(max_mae, mae)
        if previous_gray is not None:
            transition_scores.append((float(cv2.absdiff(gray_expected, previous_gray).mean()), index))
        previous_gray = gray_expected
        per_frame.append({"frame": index, "ssim_480p": round(ssim, 7),
                          "mae_480p": round(mae, 5)})
        if (index + 1) % 120 == 0 or index + 1 == count:
            print(f"qa {index + 1}/{count}  min_ssim={min_ssim:.6f}", flush=True)

    extra_frame = delivery.read()[0]
    source.release()
    delivery.release()

    mean_ssim = sum_ssim / count
    general_indices = list(range(0, count, max(1, round(fps * 2))))
    transitions = _select_transitions(transition_scores)
    outro_indices = list(range(max(0, outro_start - round(fps)), count,
                               max(1, round(fps / 6))))
    worst_indices = list(range(max(0, worst_index - 3), min(count, worst_index + 4)))

    report_dir.mkdir(parents=True, exist_ok=True)
    _make_sheet(output, general_indices, report_dir / "timeline-contact-sheet.jpg", fps)
    _make_sheet(output, transitions, report_dir / "transition-contact-sheet.jpg", fps)
    _make_sheet(output, outro_indices, report_dir / "outro-contact-sheet.jpg", fps)
    _make_sheet(output, worst_indices, report_dir / "worst-frame-neighborhood.jpg", fps,
                columns=7)

    passed = (
        out_width == width and out_height == height and
        abs(out_fps - fps) < 0.02 and out_count == count and not extra_frame and
        min_ssim >= 0.975 and mean_ssim >= 0.993 and max_mae <= 4.5
    )
    report = {
        "passed": passed,
        "reference": str(reference),
        "output": str(output),
        "dimensions": [width, height],
        "fps": fps,
        "frame_count": count,
        "output_dimensions": [out_width, out_height],
        "output_fps": out_fps,
        "output_frame_count": out_count,
        "extra_decoded_frame": bool(extra_frame),
        "argus_blue": "#2C55E6",
        "outro_start_frame": outro_start,
        "outro_start_seconds": outro_start / fps,
        "all_frames_compared": True,
        "mean_ssim_480p": mean_ssim,
        "min_ssim_480p": min_ssim,
        "worst_frame": worst_index,
        "max_frame_mae_480p": max_mae,
        "transition_frames_reviewed": transitions,
        "per_frame": per_frame,
    }
    (report_dir / "qa.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_frame"}, indent=2))
    if not passed:
        raise AssertionError("Rendered MP4 failed frame-by-frame quality gates")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "reference" / "original-60s.mp4")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    width, height, fps, count = probe(args.reference)
    if (width, height) != (1920, 1080):
        raise AssertionError(f"Expected 1920x1080 reference, got {width}x{height}")
    if abs(fps - 24.0) > 0.02:
        raise AssertionError(f"Expected 24 fps reference, got {fps}")
    if count < 1400 or count > 1450:
        raise AssertionError(f"Unexpected reference frame count: {count}")

    outro_start = find_outro_start(args.reference, fps, count)
    print(f"reference {width}x{height} {fps:.6f}fps {count} frames")
    print(f"ARGUS outro begins at frame {outro_start} ({outro_start / fps:.3f}s)")
    render(args.reference, args.output, width, height, fps, count, outro_start)
    validate(args.reference, args.output, args.report_dir,
             width, height, fps, count, outro_start)


if __name__ == "__main__":
    main()
