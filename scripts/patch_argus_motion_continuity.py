#!/usr/bin/env python3
"""Make the ARGUS outro move like the original film instead of a rigid interpolation.

The repository reconstruction is frame-by-frame rotoscoping, so there is no
single procedural 'original motion function' to copy. The first few seconds and
the animal sequence reveal the useful motion language directly: neighboring
points travel together, paths bend, timing varies by region, dot sizes breathe,
and the image keeps changing velocity instead of moving on straight synchronized
tracks. This patch preserves the approved connected-circle start and dotted
wordmark finish while reproducing those traits procedurally for the custom outro.
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

    # The circle begins moving while it is still separating. The original film
    # almost never pauses on a completed intermediate state.
    text = sub_once(
        text,
        r"^(\s*)morph_start\s*=\s*[0-9.]+\s*$",
        r"\1morph_start = 0.040",
        "morph_start",
    )
    text = sub_once(
        text,
        r"^(\s*)morph_end\s*=\s*[0-9.]+\s*$",
        r"\1morph_end = 0.835",
        "morph_end",
    )

    # Keep a short *spatial* phase difference rather than a random independent
    # delay. Adjacent dots therefore launch in coherent patches, like the horse /
    # animal and early morph sequences, instead of every point leaving at once.
    delay_pattern = (
        r"delay\s*=\s*\(0\.010\s*\+\s*0\.046\s*\*\s*radial\s*\+\s*\n"
        r"\s*0\.005\s*\*\s*\(0\.5\s*\+\s*0\.5\s*\*\s*np\.sin\(target_theta\s*\*\s*3\.0\)\)\)"
        r"\.astype\(np\.float32\)"
    )
    delay_replacement = (
        "delay = (0.002 + 0.020 * radial +\n"
        "             0.006 * (0.5 + 0.5 * np.sin(target_theta * 2.35 + radial * 8.0))).astype(np.float32)"
    )
    text = sub_once(text, delay_pattern, delay_replacement, "regional particle delay")

    # Replace the straight source->target interpolation and tiny ripple with a
    # curved, spatially correlated flow. The two Bezier controls are derived
    # from smooth functions of source position, so neighboring dots bend in the
    # same broad direction while still having local variation. A second moving
    # flow field adds the small continuous reversals / waviness visible in the
    # film. Both effects vanish exactly at the start and final wordmark.
    block_start = text.index("    morph = np.clip((p - morph_start)")
    block_end = text.index("    return _round_dot_layer", block_start)
    replacement = r'''    morph = np.clip((p - morph_start) / (morph_end - morph_start), 0.0, 1.0)

    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    target_offset = target - center
    target_distance = np.linalg.norm(target_offset, axis=1)
    target_theta = np.arctan2(target_offset[:, 1], target_offset[:, 0])

    # Regional arrival variation prevents the type from snapping into place as
    # one mathematical sheet. Every point is still fully settled before the end.
    arrival_wave = 0.5 + 0.5 * np.sin(target_theta * 2.1 + target_distance * 0.016)
    arrival = 0.885 + 0.095 * arrival_wave
    local = np.clip((morph - delay) / np.maximum(0.001, arrival - delay), 0.0, 1.0)
    eased = local * local * (3.0 - 2.0 * local)

    # Feed a small amount of the breakup directly into travel. This preserves
    # the seamless handoff approved in the previous version.
    early_flow = 0.040 * breakup
    motion = np.clip(early_flow + (1.0 - early_flow) * eased, 0.0, 1.0)

    delta = target - source
    distance = np.linalg.norm(delta, axis=1)
    safe_distance = np.maximum(distance, 1e-6)
    travel_direction = delta / safe_distance[:, None]
    travel_perp = np.column_stack([-travel_direction[:, 1], travel_direction[:, 0]])

    # Smooth, low-frequency spatial phases are the key difference from the old
    # rigid version. No per-particle random jitter: nearby points share motion.
    min_dim = float(min(width, height))
    sx = (source[:, 0] - center[0]) / min_dim
    sy = (source[:, 1] - center[1]) / min_dim
    bend_wave = (0.62 * np.sin(sx * 8.2 + sy * 5.1 + phase * 0.30) +
                 0.38 * np.sin(sx * 3.4 - sy * 9.0 + phase * 0.17 + 1.35))
    bend = bend_wave * np.minimum(distance * 0.145, min_dim * 0.105)

    control1 = source + delta * 0.18 + travel_perp * bend[:, None]
    control2 = source + delta * 0.72 - travel_perp * (bend * 0.58)[:, None]

    t = motion[:, None]
    one = 1.0 - t
    position = (one ** 3 * source +
                3.0 * one ** 2 * t * control1 +
                3.0 * one * t ** 2 * control2 +
                t ** 3 * target)

    # Film-like local advection: correlated waves pass through the dot field as
    # it travels. The amplitude is intentionally visible (the old ripple was
    # only about one pixel) but fades to zero at both endpoints.
    travel_envelope = np.sin(np.pi * motion)
    flow_phase_x = source[:, 1] * 0.015 + target[:, 0] * 0.006 + p * np.pi * 5.0
    flow_phase_y = source[:, 0] * 0.013 - target[:, 1] * 0.005 - p * np.pi * 4.2
    flow_x = (np.sin(flow_phase_x) + 0.45 * np.sin(flow_phase_y * 0.71 + 1.2)) * step * 1.55
    flow_y = (np.cos(flow_phase_y) + 0.45 * np.cos(flow_phase_x * 0.67 - 0.8)) * step * 1.55
    position = position + np.column_stack([flow_x, flow_y]) * travel_envelope[:, None]

    # The source dots shrink apart while the same points begin travelling.
    # Radius variation is also spatially coherent and disappears at settlement.
    radii = source_radius_now * (1.0 - motion) + dot_radius * motion
    pulse = (1.0 + 0.085 * np.sin(np.pi * local) *
             np.sin(phase * 0.72 + p * np.pi * 4.4) +
             0.030 * np.sin(np.pi * motion) * np.sin(sx * 7.0 + sy * 5.0 + p * np.pi * 3.0))
    radii = radii * pulse
'''
    text = text[:block_start] + replacement + text[block_end:]

    PATH.write_text(text)
    print("Applied film-style organic ARGUS dot motion patch")


if __name__ == "__main__":
    main()
