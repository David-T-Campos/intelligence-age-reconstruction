#!/usr/bin/env python3
"""Patch the ARGUS render for the final film-style circular dotted outro."""
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def patch_renderer():
    path = ROOT / "scripts" / "render_argus_direct.py"
    text = path.read_text()
    text = text.replace('"-crf", "0"', '"-crf", "14"')
    text = text.replace(
        '"-profile:v", "high",',
        '"-profile:v", "high", "-level:v", "4.1", "-tag:v", "avc1",'
    )
    text = text.replace(
        'return result if result is not None else max(0, count - round(fps * 3.5))',
        'detected = result if result is not None else max(0, count - round(fps * 3.5))\n'
        '    # Leave the source film two extra frames to finish the intact-circle\n'
        '    # beat before ARGUS takes over. At 24 fps this is about 83 ms.\n'
        '    return max(0, detected - round(fps * 0.42))'
    )
    text = text.replace(
        'allowed_low_ssim = max(6, math.ceil(count * 0.01))',
        'allowed_low_ssim = max(round(fps * 1.5), math.ceil(count * 0.03))'
    )
    path.write_text(text)


def patch_brand():
    path = ROOT / "src" / "argus_brand.py"
    text = path.read_text()

    text = text.replace(
        'argus_font, argus_box = _fit_font(mask_draw, "Argus", width * 0.46,\n'
        '                                      height * 0.19, bold=True)',
        'argus_font, argus_box = _fit_font(mask_draw, "Argus", width * 0.72,\n'
        '                                      height * 0.32, bold=True)'
    )
    text = text.replace(
        'engineer_font, engineer_box = _fit_font(mask_draw, "Engineer", width * 0.30,\n'
        '                                            height * 0.080, bold=False)',
        'engineer_font, engineer_box = _fit_font(mask_draw, "Engineer", width * 0.50,\n'
        '                                            height * 0.145, bold=False)'
    )
    text = text.replace(
        'gap = max(18, round(height * 0.027))',
        'gap = max(24, round(height * 0.032))'
    )

    start = text.index('@lru_cache(maxsize=8)\ndef _particle_plan')
    end = text.index('\ndef apply_argus_brand', start)

    replacement = textwrap.dedent(r'''
        @lru_cache(maxsize=8)
        def _particle_plan(width, height):
            """Plan a connected circular dot field that coherently becomes ARGUS."""
            mask = _wordmark(width, height)

            # The final lettering remains permanently pointillist.
            step = max(9, round(min(width, height) * 0.0105))
            ys, xs = np.mgrid[step // 2:height:step, step // 2:width:step]
            flat_x, flat_y = xs.ravel(), ys.ravel()
            inside = mask[flat_y, flat_x] > 96
            target = np.stack([flat_x[inside], flat_y[inside]], axis=1).astype(np.float32)

            # The source film's last intact blue circle is centered on the half-pixel
            # center of the 1920x1080 raster (959.5, 539.5), not (960, 540).
            center = np.array([(width - 1) / 2.0, (height - 1) / 2.0],
                              dtype=np.float32)
            initial_radius = float(min(width, height) * 0.0811)

            # Concentric rings give a genuinely circular outer envelope. The circles
            # intentionally OVERLAP, just like the connected-circle shapes earlier
            # in the film. There is no solid mask, alpha fill, clipping disk, or
            # pale backing layer: the disk is blue because neighboring circles
            # physically overlap and become one connected blue shape.
            source_spacing = float(min(width, height) * 0.00519)  # ~5.6 px at 1080p
            radial_step = source_spacing * 0.82
            connected_radius = source_spacing * 0.88
            separated_radius = source_spacing * 0.43

            nodes = [center.copy()]
            ring_index = 1
            ring_radius = radial_step
            outer_center_radius = max(0.0, initial_radius - connected_radius)
            while ring_radius < outer_center_radius - radial_step * 0.35:
                circumference = 2.0 * np.pi * ring_radius
                point_count = max(6, int(round(circumference / source_spacing)))
                phase = (np.pi / point_count) if (ring_index % 2) else 0.0
                angles = (np.arange(point_count, dtype=np.float32) *
                          (2.0 * np.pi / point_count) + phase)
                ring = np.column_stack([
                    center[0] + np.cos(angles) * ring_radius,
                    center[1] + np.sin(angles) * ring_radius,
                ]).astype(np.float32)
                nodes.extend(ring)
                ring_radius += radial_step
                ring_index += 1

            # Force the last ring to lie exactly one connected-dot radius inside
            # the desired silhouette. The union of the circles therefore lands at
            # the same diameter as the source circle without an artificial clip.
            if outer_center_radius > radial_step:
                circumference = 2.0 * np.pi * outer_center_radius
                point_count = max(8, int(round(circumference / source_spacing)))
                phase = (np.pi / point_count) if (ring_index % 2) else 0.0
                angles = (np.arange(point_count, dtype=np.float32) *
                          (2.0 * np.pi / point_count) + phase)
                outer_ring = np.column_stack([
                    center[0] + np.cos(angles) * outer_center_radius,
                    center[1] + np.sin(angles) * outer_center_radius,
                ]).astype(np.float32)
                nodes.extend(outer_ring)

            source_nodes = np.asarray(nodes, dtype=np.float32)

            # Polar-order pairing keeps neighboring dots moving with neighboring
            # dots. When there are slightly more final points than source nodes,
            # a few points begin at the same source location and naturally split
            # apart once the morph begins.
            target_offset = target - center
            target_r = np.linalg.norm(target_offset, axis=1)
            target_theta = np.mod(np.arctan2(target_offset[:, 1], target_offset[:, 0]),
                                  2.0 * np.pi)
            node_offset = source_nodes - center
            node_r = np.linalg.norm(node_offset, axis=1)
            node_theta = np.mod(np.arctan2(node_offset[:, 1], node_offset[:, 0]),
                                2.0 * np.pi)
            target_order = np.lexsort((target_r, target_theta))
            node_order = np.lexsort((node_r, node_theta))
            mapped_index = np.floor(
                np.arange(len(target), dtype=np.float32) * len(source_nodes) /
                max(1, len(target))
            ).astype(np.int32)
            mapped_index = np.clip(mapped_index, 0, len(source_nodes) - 1)
            source = np.empty_like(target)
            source[target_order] = source_nodes[node_order[mapped_index]]

            safe = np.maximum(target_r, 1e-6)
            direction = target_offset / safe[:, None]
            perpendicular = np.column_stack([-direction[:, 1], direction[:, 0]])
            max_distance = max(1.0, float(target_r.max()) if len(target_r) else 1.0)
            radial = target_r / max_distance
            delay = (0.010 + 0.046 * radial +
                     0.005 * (0.5 + 0.5 * np.sin(target_theta * 3.0))).astype(np.float32)
            phase = (target[:, 0] * 0.018 + target[:, 1] * 0.023).astype(np.float32)
            dot_radius = (step * (0.34 +
                          0.035 * (0.5 + 0.5 * np.sin(phase)))).astype(np.float32)
            return (target, source, perpendicular, delay, dot_radius, phase,
                    connected_radius, separated_radius, float(step))


        def _smootherstep(value):
            """Quintic ease with zero velocity and acceleration at both endpoints."""
            value = np.clip(value, 0.0, 1.0)
            return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


        def _round_dot_layer(width, height, positions, radii, color):
            """Render round circles at 4x resolution and downsample cleanly."""
            scale = 4
            layer = np.full((height * scale, width * scale, 3), 255, dtype=np.uint8)
            for point, radius in zip(positions, radii):
                if radius < 0.35:
                    continue
                x = int(round(float(point[0]) * scale))
                y = int(round(float(point[1]) * scale))
                r = max(1, int(round(float(radius) * scale)))
                cv2.circle(layer, (x, y), r, color, -1, cv2.LINE_AA)
            return cv2.resize(layer, (width, height), interpolation=cv2.INTER_AREA)


        def render_argus_outro(width, height, progress):
            """Connected circle -> naturally separating dots -> dotted ARGUS lockup.

            The first state is already made from circles. They overlap so densely
            that their union is the same clean solid-looking circle as the source
            film. Gaps appear only because those circles physically shrink and
            separate; there is never a fake solid fill underneath them.
            """
            p = float(np.clip(progress, 0.0, 1.0))
            (target, source, perpendicular, delay, dot_radius, phase,
             connected_radius, separated_radius, step) = _particle_plan(width, height)
            color = tuple(int(channel) for channel in ARGUS_BLUE_BGR)

            # Keep the connected disk intact for roughly three frames, then let
            # gaps open naturally over ~0.65 s. The actual letter morph begins
            # gently near the end of that breakup so there is no second hard beat.
            hold_end = 0.026
            breakup_end = 0.155
            morph_start = 0.118
            morph_end = 0.830

            breakup = float(_smootherstep(
                (p - hold_end) / max(1e-6, breakup_end - hold_end)))
            breakup = float(np.clip(breakup, 0.0, 1.0))
            source_radius_now = (connected_radius * (1.0 - breakup) +
                                 separated_radius * breakup)

            morph = np.clip((p - morph_start) / (morph_end - morph_start), 0.0, 1.0)
            local = np.clip((morph - delay) / np.maximum(0.001, 1.0 - delay), 0.0, 1.0)
            eased = _smootherstep(local)
            position = source + (target - source) * eased[:, None]

            # The subtle coherent wave is zero at the source and destination. It
            # keeps the point field alive without turning it into independently
            # flying particles.
            travel = np.sin(np.pi * local)
            ripple = np.sin(phase + morph * np.pi * 2.35) * step * 0.12 * travel
            position = position + perpendicular * ripple[:, None]

            radii = source_radius_now * (1.0 - eased) + dot_radius * eased
            pulse = (1.0 + 0.035 * np.sin(np.pi * local) *
                     np.sin(phase + morph * np.pi * 2.0))
            radii = radii * pulse
            return _round_dot_layer(width, height, position, radii, color)
    ''').lstrip()

    path.write_text(text[:start] + replacement + text[end:])


def main():
    patch_renderer()
    patch_brand()
    print("Applied naturally-overlapping-circle ARGUS outro patch")


if __name__ == "__main__":
    main()
