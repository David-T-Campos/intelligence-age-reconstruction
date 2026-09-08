#!/usr/bin/env python3
"""Make the connected-circle breakup flow continuously into the ARGUS wordmark.

Runs after patch_argus_outro.py. The approved connected-circle appearance stays
unchanged; this only removes the dead beat where the dots had separated but the
wordmark travel had not visibly begun yet.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "argus_brand.py"


def sub_once(text, pattern, replacement, label):
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not patch {label}; matches={count}")
    return text


def main():
    text = PATH.read_text()

    # Start the actual shape travel while the circles are still separating, so
    # the viewer never lands on a stationary dotted disk.
    text = sub_once(
        text,
        r"^(\s*)morph_start\s*=\s*[0-9.]+\s*$",
        r"\1morph_start = 0.045",
        "morph_start",
    )
    text = sub_once(
        text,
        r"^(\s*)morph_end\s*=\s*[0-9.]+\s*$",
        r"\1morph_end = 0.820",
        "morph_end",
    )

    # Keep only a very small center-out phase offset. The earlier delay range
    # was long enough that the separated field looked parked for a moment.
    delay_pattern = (
        r"delay\s*=\s*\(0\.010\s*\+\s*0\.046\s*\*\s*radial\s*\+\s*\n"
        r"\s*0\.005\s*\*\s*\(0\.5\s*\+\s*0\.5\s*\*\s*np\.sin\(target_theta\s*\*\s*3\.0\)\)\)"
        r"\.astype\(np\.float32\)"
    )
    delay_replacement = (
        "delay = (0.001 + 0.010 * radial +\n"
        "             0.0015 * (0.5 + 0.5 * np.sin(target_theta * 3.0))).astype(np.float32)"
    )
    text = sub_once(text, delay_pattern, delay_replacement, "particle delay")

    # Quintic smootherstep has an intentionally very slow launch. For this one
    # handoff use cubic smoothstep, then feed a little of the breakup progress
    # directly into travel. Motion therefore begins before the dots finish
    # opening and never drops to an apparent zero-speed plateau.
    motion_pattern = (
        r"eased\s*=\s*_smootherstep\(local\)\s*\n"
        r"\s*position\s*=\s*source\s*\+\s*\(target\s*-\s*source\)\s*\*\s*eased\[:,\s*None\]"
    )
    motion_replacement = (
        "eased = local * local * (3.0 - 2.0 * local)\n"
        "    early_flow = 0.045 * breakup\n"
        "    motion = early_flow + (1.0 - early_flow) * eased\n"
        "    position = source + (target - source) * motion[:, None]"
    )
    text = sub_once(text, motion_pattern, motion_replacement, "continuous position motion")

    text = sub_once(
        text,
        r"^(\s*)radii\s*=\s*source_radius_now\s*\*\s*\(1\.0\s*-\s*eased\)\s*\+\s*dot_radius\s*\*\s*eased\s*$",
        r"\1radii = source_radius_now * (1.0 - motion) + dot_radius * motion",
        "radius motion",
    )

    PATH.write_text(text)
    print("Applied seamless ARGUS breakup-to-morph continuity patch")


if __name__ == "__main__":
    main()
