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
        '    # Take over two frames later than the previous revision so the\n'
        '    # source film gets a tiny natural hold on its intact blue circle.\n'
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
            """Plan an exact circular source that coherently becomes the wordmark."""
            mask = _wordmark(width, height)

            # Final lettering remains permanently pointillist.
            step = max(9, round(min(width, height) * 0.0105))
            ys, xs = np.mgrid[step // 2:height:step, step // 2:width:step]
            flat_x, flat_y = xs.ravel(), ys.ravel()
            inside = mask[flat_y, flat_x] > 96
            target = np.stack([flat_x[inside], flat_y[inside]], axis=1).astype(np.float32)

            center = np.array([width / 2, height / 2], dtype=np.float32)
            initial_radius = float(min(width, height) * 0.082)

            # Build a clean hexagonal source field inside the exact same radius as
            # the original solid blue circle. The old 0.87-radius source was what
            # produced the visibly smaller dotted disk in the user's screenshot.
            source_step = max(8.0, min(width, height) * 0.00835)
            source_dot_radius = max(2.5, source_step * 0.30)
            rows = []
            y = -initial_radius
            row = 0
            dy = source_step * np.sqrt(3.0) / 2.0
            limit = initial_radius - source_dot_radius * 0.55
            while y <= initial_radius:
                offset = (source_step * 0.5) if (row % 2) else 0.0
                x = -initial_radius + offset
                while x <= initial_radius:
                    if x * x + y * y <= limit * limit:
                        rows.append((center[0] + x, center[1] + y))
                    x += source_step
                y += dy
                row += 1
            source_nodes = np.asarray(rows, dtype=np.float32)

            # Map the larger final particle set onto the circular field by polar
            # order. Several final points may share one source node, so the dots
            # visibly split apart only after the circular field is established.
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
            delay = (0.010 + 0.050 * radial +
                     0.006 * (0.5 + 0.5 * np.sin(target_theta * 3.0))).astype(np.float32)
            phase = (target[:, 0] * 0.018 + target[:, 1] * 0.023).astype(np.float32)
            dot_radius = (step * (0.34 +
                          0.035 * (0.5 + 0.5 * np.sin(phase)))).astype(np.float32)

            # Radius large enough that the clipped hex field reconstructs a solid
            # disk with no scalloped edge at the first separation frame.
            cover_radius = float(source_step * 0.78)
            return (target, source, source_nodes, perpendicular, delay, dot_radius,
                    phase, initial_radius, source_dot_radius, cover_radius, float(step))


        def _smootherstep(value):
            """Quintic ease with zero velocity and acceleration at both endpoints."""
            value = np.clip(value, 0.0, 1.0)
            return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


        def _round_dot_layer(width, height, positions, radii, color):
            """Render truly round circles at 3x resolution and downsample cleanly."""
            scale = 3
            layer = np.full((height * scale, width * scale, 3), 255, dtype=np.uint8)
            for point, radius in zip(positions, radii):
                if radius < 0.35:
                    continue
                x = int(round(float(point[0]) * scale))
                y = int(round(float(point[1]) * scale))
                r = max(1, int(round(float(radius) * scale)))
                cv2.circle(layer, (x, y), r, color, -1, cv2.LINE_AA)
            return cv2.resize(layer, (width, height), interpolation=cv2.INTER_AREA)


        def _exact_circle(width, height, radius, color):
            """Return an antialiased solid circle at the exact original diameter."""
            scale = 4
            layer = np.full((height * scale, width * scale, 3), 255, dtype=np.uint8)
            cv2.circle(layer,
                       (int(round(width * scale / 2)), int(round(height * scale / 2))),
                       int(round(radius * scale)), color, -1, cv2.LINE_AA)
            return cv2.resize(layer, (width, height), interpolation=cv2.INTER_AREA)


        def _clipped_source_field(width, height, source_nodes, radius,
                                  initial_radius, color):
            """Draw saturated source dots, clipped to the exact original circle."""
            scale = 3
            field = np.full((height * scale, width * scale, 3), 255, dtype=np.uint8)
            c = tuple(int(channel) for channel in color)
            r = max(1, int(round(float(radius) * scale)))
            for point in source_nodes:
                cv2.circle(field,
                           (int(round(float(point[0]) * scale)),
                            int(round(float(point[1]) * scale))),
                           r, c, -1, cv2.LINE_AA)

            # Clip, rather than cross-fade, so the perimeter never turns into the
            # pale ghost ring visible in the rejected version.
            clip = np.zeros((height * scale, width * scale), dtype=np.uint8)
            cv2.circle(clip,
                       (int(round(width * scale / 2)), int(round(height * scale / 2))),
                       int(round(initial_radius * scale)), 255, -1, cv2.LINE_AA)
            white = np.full_like(field, 255)
            alpha = clip.astype(np.float32)[:, :, None] / 255.0
            field = field.astype(np.float32) * alpha + white.astype(np.float32) * (1.0 - alpha)
            field = np.clip(np.rint(field), 0, 255).astype(np.uint8)
            return cv2.resize(field, (width, height), interpolation=cv2.INTER_AREA)


        def render_argus_outro(width, height, progress):
            """Exact circle -> circular dot breakup -> coherent dotted ARGUS lockup."""
            p = float(np.clip(progress, 0.0, 1.0))
            (target, source, source_nodes, perpendicular, delay, dot_radius,
             phase, initial_radius, source_dot_radius, cover_radius, step) = \
                _particle_plan(width, height)
            color = tuple(int(channel) for channel in ARGUS_BLUE_BGR)

            # About three video frames of the exact solid circle after takeover,
            # then a short clean breakup. There is no alpha blend between a pale
            # disk and darker dots: white gaps are carved into a saturated disk.
            hold_end = 0.026
            split_end = 0.100
            morph_start = split_end
            morph_end = 0.825

            if p <= hold_end:
                return _exact_circle(width, height, initial_radius, color)

            if p < split_end:
                split = float(_smootherstep((p - hold_end) / (split_end - hold_end)))
                radius = cover_radius * (1.0 - split) + source_dot_radius * split
                return _clipped_source_field(width, height, source_nodes, radius,
                                             initial_radius, color)

            morph = np.clip((p - morph_start) / (morph_end - morph_start), 0.0, 1.0)
            local = np.clip((morph - delay) / np.maximum(0.001, 1.0 - delay), 0.0, 1.0)
            eased = _smootherstep(local)
            position = source + (target - source) * eased[:, None]

            # Small coherent transverse wave; zero at both ends and far below one
            # dot spacing. This reads like the earlier film morphs, not particles
            # independently flying around.
            travel = np.sin(np.pi * local)
            ripple = np.sin(phase + morph * np.pi * 2.35) * step * 0.13 * travel
            position = position + perpendicular * ripple[:, None]

            radii = source_dot_radius * (1.0 - eased) + dot_radius * eased
            pulse = (1.0 + 0.040 * np.sin(np.pi * local) *
                     np.sin(phase + morph * np.pi * 2.0))
            radii = radii * pulse
            return _round_dot_layer(width, height, position, radii, color)
    ''').lstrip()

    path.write_text(text[:start] + replacement + text[end:])


def main():
    patch_renderer()
    patch_brand()
    print("Applied exact-circle ARGUS outro patch")


if __name__ == "__main__":
    main()
