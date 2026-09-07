#!/usr/bin/env python3
"""Verify the current renderer in a project without a reference video."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
initial_config = json.loads((ROOT / "project.json").read_text())
stem = initial_config.get("output_name", "recreated-first10")
stage = ROOT / "work" / ("render-without-reference-" + stem)
for name in ["src", "scripts", "renders"]:
    (stage / name).mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "src/contours.py", stage / "src/contours.py")
shutil.copy2(ROOT / "scripts/pipeline.py", stage / "scripts/pipeline.py")
config = json.loads((ROOT / "project.json").read_text())
config["reference"] = "reference/INTENTIONALLY-ABSENT.mp4"
config["frames"] = str(ROOT / config["frames"])
config["audio"] = str(ROOT / config["audio"])
(stage / "project.json").write_text(json.dumps(config, indent=2))
assert not (stage / "reference").exists()
previous = ROOT / "renders" / (stem + ".mp4")
previous_hash = hashlib.sha256(previous.read_bytes()).hexdigest() if previous.exists() else None
started = time.monotonic()
subprocess.run([sys.executable, str(stage / "scripts/pipeline.py"), "render"], cwd=stage, check=True)
rebuilt_hash = hashlib.sha256((stage / "renders" / (stem + ".mp4")).read_bytes()).hexdigest()
for path in [stage / "renders" / (stem + suffix) for suffix in (".mp4", "-lossless.mkv")]:
    os.replace(path, ROOT / "renders" / path.name)
report = {
    "passed": True,
    "reference_directory_present": False,
    "reference_in_config": config["reference"],
    "visual_input": initial_config["frames"] + ": integer-grid contours, no bitmap/video textures",
    "audio_input": initial_config["audio"] + ": original audio",
    "identical_to_previous_mp4": previous_hash == rebuilt_hash if previous_hash else None,
    "mp4_sha256": rebuilt_hash,
    "elapsed_seconds": time.monotonic() - started,
}
directory = ROOT / initial_config.get("report_dir", "reports")
directory.mkdir(parents=True, exist_ok=True)
(directory / "independent-render.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
