#!/usr/bin/env python3
"""Trace, independently render, and compare the reference-assisted scene."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
from contours import trace_frame, rasterize_frame, save, load, dot_summary
from argus_brand import apply_argus_brand, detect_outro_start


def configuration():
    return json.loads((ROOT / "project.json").read_text())


def output_path(config, suffix=".mp4"):
    return ROOT / "renders" / (config.get("output_name", "recreated-first10") + suffix)


def report_path(config, name):
    directory = ROOT / config.get("report_dir", "reports")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def extract_one(task):
    index, reference, target, start = task
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(reference)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start + index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode reference frame {start + index}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    geometry = trace_frame(frame)
    reconstructed = rasterize_frame(geometry)
    if not np.array_equal(frame, reconstructed):
        raise AssertionError(f"Contour reconstruction mismatch in frame {index}")
    path = Path(target) / f"{index:06d}.npz"
    save(path, geometry)
    return {
        "frame": index,
        "source_frame": start + index,
        "gray_sha256": hashlib.sha256(gray.tobytes()).hexdigest(),
        "bgr_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "color": int(geometry["version"][0]) == 2,
        "geometry_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "layers": sum(len(v) for k, v in geometry.items() if k.endswith("shades")),
        "contours": sum(len(v) - 1 for k, v in geometry.items() if k.endswith("contour_ends")),
        "vertices": sum(len(v) for k, v in geometry.items() if k.endswith("vertices")),
        "shapes": dot_summary(gray),
        "max_pixel_error": 0,
    }


def extract(args):
    config = configuration()
    target = ROOT / config["frames"]
    target.mkdir(parents=True, exist_ok=True)
    reference = ROOT / config["reference"]
    count = config["frame_count"]
    capture = cv2.VideoCapture(str(reference))
    if not capture.isOpened():
        raise ValueError("Reference video cannot be opened")
    if (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (config["width"], config["height"]):
        raise ValueError("Configured dimensions differ from reference")
    if abs(capture.get(cv2.CAP_PROP_FPS) - config["fps"]) > 1e-6:
        raise ValueError("Configured frame rate differs from reference")
    if count < 1 or config["start_frame"] < 0 or config["start_frame"] + count > capture.get(cv2.CAP_PROP_FRAME_COUNT):
        raise ValueError("Requested frame range exceeds reference")
    capture.release()
    tasks = [(i, str(reference), str(target), config["start_frame"]) for i in range(count)]
    begin = time.monotonic()
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(extract_one, tasks, chunksize=3):
            frames.append(result)
            if len(frames) % 24 == 0 or len(frames) == count:
                print(f"trace {len(frames)}/{count}, all completed frames pixel-exact, "
                      f"{time.monotonic() - begin:.1f}s", flush=True)
    manifest = {
        "format": "color-contour-sequence-v2",
        "method": "source-assisted contour tracing; motion is a per-frame geometry sequence",
        "config": config,
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "frame_count": count,
        "color_frame_count": sum(f["color"] for f in frames),
        "geometry_bytes": sum(frame["bytes"] for frame in frames),
        "frames": frames,
    }
    (target.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # This is an explicitly reused audio asset, not generated music.
    run(["ffmpeg", "-v", "error", "-y", "-ss", str(config["start_frame"] / config["fps"]),
         "-i", str(reference), "-t", str(count / config["fps"]), "-map", "0:a:0",
         "-c:a", "pcm_s24le", str(ROOT / config["audio"])])
    print("Geometry and original audio asset ready", flush=True)


def frame_at(index, config):
    return rasterize_frame(load(ROOT / config["frames"] / f"{index:06d}.npz"))


def brand_outro_start(config):
    return detect_outro_start(config["frame_count"], config["fps"],
                              lambda index: frame_at(index, config))


def render(args):
    # Intentionally never opens config['reference']; this is also verified by a
    # complete render in a separate project with no reference video.
    config = configuration()
    width, height = config["width"], config["height"]
    count, fps = config["frame_count"], config["fps"]
    manifest = json.loads(((ROOT / config["frames"]).parent / "manifest.json").read_text())
    for key in ["width", "height", "fps", "frame_count", "start_frame"]:
        if manifest["config"][key] != config[key]:
            raise ValueError(f"Stale geometry manifest: {key} changed; run extract first")
    for index in range(count):
        if not (ROOT / config["frames"] / f"{index:06d}.npz").is_file():
            raise ValueError(f"Missing geometry frame {index}; run extract first")
    outro_start = brand_outro_start(config)
    print(f"ARGUS outro begins at frame {outro_start} ({outro_start / fps:.3f}s)", flush=True)
    destination = output_path(config, "-lossless.mkv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
               "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
               "-i", str(ROOT / config["audio"]), "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-c:a", "pcm_s24le",
               "-t", str(count / fps), str(destination)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i in range(count):
            frame = apply_argus_brand(frame_at(i, config), i, count, outro_start)
            if frame.shape != (height, width, 3):
                raise ValueError(f"Frame {i} has wrong size")
            process.stdin.write(frame.tobytes())
            if (i + 1) % 24 == 0:
                print(f"render {i + 1}/{count} from geometry + ARGUS brand layer", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg lossless render failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    run(["ffmpeg", "-v", "error", "-y", "-i", str(destination), "-map", "0:v:0", "-map", "0:a:0",
         "-vf", "scale=in_range=pc:out_range=pc:out_color_matrix=bt709,format=yuvj420p",
         "-c:v", "libx264", "-preset", "slow", "-crf", "1", "-pix_fmt", "yuvj420p",
         "-x264-params", "aq-mode=0:psy=0", "-color_range", "pc",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
         "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart",
         str(output_path(config))])
    print("Lossless ARGUS master and compatible MP4 ready", flush=True)


def compare(args):
    from skimage.metrics import structural_similarity
    config = configuration()
    source = cv2.VideoCapture(str(ROOT / config["reference"]))
    source.set(cv2.CAP_PROP_POS_FRAMES, config["start_frame"])
    master = cv2.VideoCapture(str(output_path(config, "-lossless.mkv")))
    delivery = cv2.VideoCapture(str(output_path(config)))
    manifest = json.loads(((ROOT / config["frames"]).parent / "manifest.json").read_text())
    outro_start = brand_outro_start(config)
    per_frame = []
    for i in range(config["frame_count"]):
        ok1, original = source.read()
        ok2, rebuilt = master.read()
        ok3, mp4 = delivery.read()
        if not (ok1 and ok2 and ok3):
            raise AssertionError(f"Missing decoded frame {i}")

        base = frame_at(i, config)
        matches_extraction = hashlib.sha256(base.tobytes()).hexdigest() == manifest["frames"][i]["bgr_sha256"]
        source_difference = np.abs(original.astype(np.int16) - base.astype(np.int16))
        geometry_identical = bool(matches_extraction and
                                  manifest["frames"][i]["max_pixel_error"] == 0 and
                                  source_difference.max() == 0)

        expected = apply_argus_brand(base, i, config["frame_count"], outro_start)
        gray = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)
        master_difference = np.abs(expected.astype(np.int16) - rebuilt.astype(np.int16))
        delivery_difference = np.abs(expected.astype(np.int16) - mp4.astype(np.int16))
        mse = float(np.mean(delivery_difference.astype(np.float32) ** 2))
        score = float(structural_similarity(gray, cv2.cvtColor(mp4, cv2.COLOR_BGR2GRAY), data_range=255))
        per_frame.append({
            "frame": i, "time": i / config["fps"],
            "geometry_identical": geometry_identical,
            "master_max_error": int(master_difference.max()),
            "master_different_pixels": int(np.count_nonzero(master_difference.max(axis=2))),
            "mp4_max_channel_error": int(delivery_difference.max()),
            "mp4_mae": float(delivery_difference.mean()), "mp4_mse": mse,
            "mp4_psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
            "mp4_ssim": score,
        })
        if (i + 1) % 24 == 0:
            print(f"compare {i + 1}/{config['frame_count']}", flush=True)
    if master.read()[0] or delivery.read()[0]:
        raise AssertionError("Render contains extra video frames")
    source.release()
    master.release()
    delivery.release()
    mse = float(np.mean([f["mp4_mse"] for f in per_frame]))
    results = {
        "frame_count": len(per_frame), "fps": config["fps"],
        "argus_blue": "#0B1F3A",
        "argus_outro_start_frame": outro_start,
        "argus_outro_start_seconds": outro_start / config["fps"],
        "geometry_all_frames_identical": all(f["geometry_identical"] for f in per_frame),
        "master_all_frames_identical_to_argus_target": all(f["master_max_error"] == 0 for f in per_frame),
        "master_max_pixel_error": max(f["master_max_error"] for f in per_frame),
        "delivery_mean_ssim": float(np.mean([f["mp4_ssim"] for f in per_frame])),
        "delivery_min_ssim": min(f["mp4_ssim"] for f in per_frame),
        "delivery_psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
        "delivery_mean_absolute_channel_error": float(np.mean([f["mp4_mae"] for f in per_frame])),
        "delivery_max_channel_error": max(f["mp4_max_channel_error"] for f in per_frame),
        "frames": per_frame,
    }
    report_path(config, "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "frames"}, indent=2), flush=True)
    if not results["geometry_all_frames_identical"] or not results["master_all_frames_identical_to_argus_target"]:
        raise AssertionError("ARGUS render did not meet exact-match acceptance")


def inspect(args):
    config = configuration()
    index = args.frame
    if index < 0 or index >= config["frame_count"]:
        raise ValueError("Frame outside configured range")
    outro_start = brand_outro_start(config)
    destination = report_path(config, f"reconstructed-{index:06d}.png")
    frame = apply_argus_brand(frame_at(index, config), index,
                              config["frame_count"], outro_start)
    cv2.imwrite(str(destination), frame)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "render", "compare", "inspect"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frame", type=int, default=168)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    globals()[args.command](args)
