#!/usr/bin/env python3
"""Build review media from source and the source-assisted reconstructed master."""
from pathlib import Path
import json
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "project.json").read_text())
stem = config.get("output_name", "recreated-first10")
report_dir = ROOT / config.get("report_dir", "reports")
report_dir.mkdir(parents=True, exist_ok=True)
source = cv2.VideoCapture(str(ROOT / config["reference"]))
source.set(cv2.CAP_PROP_POS_FRAMES, config["start_frame"])
master = cv2.VideoCapture(str(ROOT / "renders" / (stem + "-lossless.mkv")))
delivery = cv2.VideoCapture(str(ROOT / "renders" / (stem + ".mp4")))
(ROOT / "renders").mkdir(parents=True, exist_ok=True)
process = subprocess.Popen([
    "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
    "-video_size", "1920x624", "-framerate", str(config["fps"]), "-i", "pipe:0",
    "-i", str(ROOT / config["audio"]), "-map", "0:v:0", "-map", "1:a:0",
    "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart",
    "-t", str(config["frame_count"] / config["fps"]),
    str(ROOT / "renders" / (stem + "-comparison.mp4"))
], stdin=subprocess.PIPE)
samples = set(np.linspace(0, config["frame_count"] - 1, 12, dtype=int).tolist())
if config["frame_count"] == 1440:
    samples = {0, 156, 288, 432, 576, 680, 792, 960, 1104, 1248, 1332, 1439}
sheet = Image.new("RGB", (1440, len(samples) * 300 + 48), "#15171b")
draw = ImageDraw.Draw(sheet)
font = ImageFont.load_default(size=19)
for x, label in [(12, "REFERENCE"), (492, "RECONSTRUCTION"), (972, "MP4 DIFFERENCE x16")]:
    draw.text((x, 12), label, font=font, fill="white")
row = 0
for index in range(config["frame_count"]):
    ok1, original = source.read()
    ok2, reconstructed = master.read()
    ok3, mp4 = delivery.read()
    assert ok1 and ok2 and ok3
    canvas = np.full((624, 1920, 3), 22, np.uint8)
    canvas[60:600, :960] = cv2.resize(original, (960, 540), interpolation=cv2.INTER_AREA)
    canvas[60:600, 960:] = cv2.resize(reconstructed, (960, 540), interpolation=cv2.INTER_AREA)
    title_left = "REFERENCE / ORIGINAL"
    title_right = f"RECONSTRUCTED / FRAME {index:03d} / {index / config['fps']:.3f}s"
    cv2.putText(canvas, title_left, (24, 39), cv2.FONT_HERSHEY_SIMPLEX, .8, (238, 238, 238), 1, cv2.LINE_AA)
    cv2.putText(canvas, title_right, (984, 39), cv2.FONT_HERSHEY_SIMPLEX, .8, (238, 238, 238), 1, cv2.LINE_AA)
    process.stdin.write(canvas.tobytes())
    if index in samples:
        error = np.abs(original.astype(np.int16) - mp4.astype(np.int16)).max(axis=2)
        amplified = np.clip(error * 16, 0, 255).astype(np.uint8)
        error_rgb = cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)
        y = 48 + row * 300
        for column, frame in enumerate([original, reconstructed, error_rgb]):
            thumb = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            sheet.paste(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)), (column * 480, y))
        draw.text((12, y + 274), f"frame {index:03d} / {index / config['fps']:.3f}s", fill="white", font=font)
        row += 1
process.stdin.close()
assert process.wait() == 0
source.release()
master.release()
delivery.release()
sheet.save(report_dir / "visual-comparison.jpg", quality=94)
print("Review video and contact sheet ready", flush=True)
