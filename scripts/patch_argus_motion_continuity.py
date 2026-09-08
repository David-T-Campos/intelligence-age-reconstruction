#!/usr/bin/env python3
"""Make the connected-circle breakup flow continuously into the ARGUS wordmark.

This runs after patch_argus_outro.py. It keeps the approved connected-circle
look unchanged and only removes the brief dead zone where the dots had already
separated but had not yet started travelling toward the wordmark.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "argus_brand.py"


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match, found {count}: {old!r}")
    return text.replace(old, new, 1)


def main():
    text = PATH.read_text()

    # Start coherent travel while the circles are still opening up. Previously
    # morph_start=0.118 meant the visible separation mostly finished before the
    # point field moved, which created the awkward pause the user identified.
    text = replace_once(text,
        "    morph_start = 0.118\n",
        "    morph_start = 0.052\n")

    # Keep the center-out character, but drastically reduce per-dot waiting so
    # every part of the field is already in motion during the breakup instead of
    # outer dots lingering after the gaps have appeared.
    text = replace_once(text,
        "    delay = (0.010 + 0.046 * radial +\n"
        "             0.005 * (0.5 + 0.5 * np.sin(target_theta * 3.0))).astype(np.float32)\n",
        "    delay = (0.001 + 0.014 * radial +\n"
        "             0.002 * (0.5 + 0.5 * np.sin(target_theta * 3.0))).astype(np.float32)\n")

    # Slightly overlap the two motions even more: as the source circles shrink,
    # feed a small amount of the same breakup progress into their travel. This
    # guarantees non-zero motion throughout the handoff while preserving the
    # connected-circle silhouette at the start and the dotted lockup at the end.
    old = (
        "    eased = _smootherstep(local)\n"
        "    position = source + (target - source) * eased[:, None]\n"
    )
    new = (
        "    eased = _smootherstep(local)\n"
        "    early_flow = 0.030 * breakup\n"
        "    motion = early_flow + (1.0 - early_flow) * eased\n"
        "    position = source + (target - source) * motion[:, None]\n"
    )
    text = replace_once(text, old, new)

    # Radius interpolation follows the same continuous motion parameter so there
    # is no visual beat where position and dot-size animation disagree.
    text = replace_once(text,
        "    radii = source_radius_now * (1.0 - eased) + dot_radius * eased\n",
        "    radii = source_radius_now * (1.0 - motion) + dot_radius * motion\n")

    PATH.write_text(text)
    print("Applied seamless ARGUS breakup-to-morph continuity patch")


if __name__ == "__main__":
    main()
