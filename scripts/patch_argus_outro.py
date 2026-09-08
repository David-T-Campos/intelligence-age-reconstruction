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
        '    return max(0, detected - round(fps * 0.50))'
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
            """Plan a coherent film-style dot-matrix expansion into the wordmark."""
            mask = _wordmark(width, height)

            # Use the same visual grammar as the film: clearly separated round
            # points on a structured field, not tiny pixel-like particles.
            step = max(9, round(min(width, height) * 0.0105))
            ys, xs = np.mgrid[step // 2:height:step, step // 2:width:step]
            flat_x, flat_y = xs.ravel(), ys.ravel()
            inside = mask[flat_y, flat_x] > 96
            target = np.stack([flat_x[inside], flat_y[inside]], axis=1).astype(np.float32)

            center = np.array([width / 2, height / 2], dtype=np.float32)
            offset = target - center
            distance = np.linalg.norm(offset, axis=1)
            safe = np.maximum(distance, 1e-6)
            direction = offset / safe[:, None]
            perpendicular = np.column_stack([-direction[:, 1], direction[:, 0]])
            max_distance = max(1.0, float(distance.max()) if len(distance) else 1.0)
            radial = distance / max_distance
            angle = np.arctan2(offset[:, 1], offset[:, 0])

            # Start as a tiny version of the final geometry inside the source dot.
            # That preserves topology and makes the transition behave like the
            # grow/morph sequences earlier in the film rather than a particle spray.
            source = center + offset * 0.045

            # Center-out reveal with a restrained wave. Neighboring dots remain
            # neighbors instead of taking unrelated paths across the frame.
            delay = (0.015 + 0.125 * radial +
                     0.012 * (0.5 + 0.5 * np.sin(angle * 3.0))).astype(np.float32)
            arrival = (0.70 + 0.10 * radial).astype(np.float32)

            phase = (target[:, 0] * 0.018 + target[:, 1] * 0.023).astype(np.float32)
            dot_radius = (step * (0.34 +
                          0.035 * (0.5 + 0.5 * np.sin(phase)))).astype(np.float32)
            initial_radius = float(min(width, height) * 0.082)
            return (target, source, perpendicular, delay, arrival,
                    dot_radius, phase, initial_radius, float(step))


        def _smootherstep(value):
            """Quintic ease with zero velocity and acceleration at both endpoints."""
            value = np.clip(value, 0.0, 1.0)
            return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


        def _round_dot_layer(width, height, positions, radii, color):
            """Render truly round circles at 2x resolution and downsample cleanly."""
            scale = 2
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
            """Morph the single dot into a large, permanently dotted ARGUS lockup.

            The motion deliberately follows the established film language: the
            point field stays coherent, expands outward, and changes dot size.
            There is no random Bezier flight and no solid-text phase.
            """
            p = float(np.clip(progress, 0.0, 1.0))
            (target, source, perpendicular, delay, arrival,
             dot_radius, phase, initial_radius, step) = _particle_plan(width, height)

            color = tuple(int(channel) for channel in ARGUS_BLUE_BGR)
            local = (p - delay) / np.maximum(0.001, arrival - delay)
            clipped = np.clip(local, 0.0, 1.0)
            eased = _smootherstep(clipped)

            # Geometry-preserving expansion plus a very small transverse ripple.
            # The ripple is intentionally below one dot spacing and vanishes at
            # both ends, matching the living pointillist movement elsewhere.
            position = source + (target - source) * eased[:, None]
            travel = np.sin(np.pi * clipped)
            ripple = np.sin(phase + p * np.pi * 3.0) * step * 0.42 * travel
            position = position + perpendicular * ripple[:, None]

            # Dots grow while the object grows, then settle into stable circles.
            appear = _smootherstep((p - np.maximum(0.0, delay - 0.035)) / 0.095)
            grow = 0.46 + 0.54 * eased
            pulse = (1.0 + 0.10 * np.sin(np.pi * clipped) *
                     np.sin(phase + p * np.pi * 2.0))
            radii = dot_radius * appear * grow * pulse
            frame = _round_dot_layer(width, height, position, radii, color)

            # The original source dot contracts directly into the structured field.
            circle_ease = float(_smootherstep(p / 0.155))
            circle_alpha = 1.0 - circle_ease
            if circle_alpha > 0.001:
                circle_radius = initial_radius * (1.0 - 0.17 * circle_ease)
                circle = _round_dot_layer(
                    width, height,
                    np.array([[width / 2, height / 2]], dtype=np.float32),
                    np.array([circle_radius], dtype=np.float32), color,
                )
                frame = cv2.addWeighted(circle, circle_alpha, frame,
                                        1.0 - circle_alpha, 0.0)

            return frame
    ''').lstrip()

    path.write_text(text[:start] + replacement + text[end:])


def main():
    patch_renderer()
    patch_brand()
    print("Applied circular film-style ARGUS outro patch")


if __name__ == "__main__":
    main()
