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
            """Plan a true circular source field that coherently becomes the wordmark."""
            mask = _wordmark(width, height)

            # Final lettering uses clearly separated round points, matching the
            # pointillist grammar of the rest of the film rather than pixel specks.
            step = max(9, round(min(width, height) * 0.0105))
            ys, xs = np.mgrid[step // 2:height:step, step // 2:width:step]
            flat_x, flat_y = xs.ravel(), ys.ravel()
            inside = mask[flat_y, flat_x] > 96
            target = np.stack([flat_x[inside], flat_y[inside]], axis=1).astype(np.float32)

            center = np.array([width / 2, height / 2], dtype=np.float32)
            count = len(target)
            initial_radius = float(min(width, height) * 0.082)

            # IMPORTANT: the previous revision started as a miniature copy of the
            # wordmark, which became the little "shirt" blob as soon as the solid
            # circle faded. Start from an actually circular, uniformly filled dot
            # field instead. A Vogel / sunflower disk gives an even circular fill
            # with no hidden word silhouette at the handoff.
            idx = np.arange(count, dtype=np.float32)
            golden_angle = np.float32(np.pi * (3.0 - np.sqrt(5.0)))
            source_angle = idx * golden_angle
            source_radius = initial_radius * 0.87 * np.sqrt((idx + 0.5) / max(1, count))
            source_raw = np.column_stack([
                center[0] + np.cos(source_angle) * source_radius,
                center[1] + np.sin(source_angle) * source_radius,
            ]).astype(np.float32)

            # Pair circular-source points with final targets by polar order. This
            # keeps neighbors moving with neighbors and avoids the random particle
            # flight that looked unlike the earlier morphs in the film.
            source_offset = source_raw - center
            target_offset = target - center
            source_r = np.linalg.norm(source_offset, axis=1)
            target_r = np.linalg.norm(target_offset, axis=1)
            source_theta = np.mod(np.arctan2(source_offset[:, 1], source_offset[:, 0]),
                                  2.0 * np.pi)
            target_theta = np.mod(np.arctan2(target_offset[:, 1], target_offset[:, 0]),
                                  2.0 * np.pi)
            source_order = np.lexsort((source_r, source_theta))
            target_order = np.lexsort((target_r, target_theta))
            source = np.empty_like(source_raw)
            source[target_order] = source_raw[source_order]

            safe = np.maximum(target_r, 1e-6)
            direction = target_offset / safe[:, None]
            perpendicular = np.column_stack([-direction[:, 1], direction[:, 0]])
            max_distance = max(1.0, float(target_r.max()) if count else 1.0)
            radial = target_r / max_distance

            # A restrained center-out phase offset recreates the film's broad
            # shape morphs without making each dot feel independently animated.
            delay = (0.015 + 0.075 * radial +
                     0.010 * (0.5 + 0.5 * np.sin(target_theta * 3.0))).astype(np.float32)
            phase = (target[:, 0] * 0.018 + target[:, 1] * 0.023).astype(np.float32)
            dot_radius = (step * (0.34 +
                          0.035 * (0.5 + 0.5 * np.sin(phase)))).astype(np.float32)
            source_dot_radius = float(max(1.8, step * 0.19))
            return (target, source, perpendicular, delay, dot_radius, phase,
                    initial_radius, source_dot_radius, float(step))


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
            """Circle -> circular dot field -> large permanently dotted ARGUS lockup.

            The source remains a real circle during the handoff. It first resolves
            into round dots *inside that same circle*, and only then does the whole
            coherent field stretch into the final lettering. There is no miniature
            wordmark, random particle spray, or solid-text phase.
            """
            p = float(np.clip(progress, 0.0, 1.0))
            (target, source, perpendicular, delay, dot_radius, phase,
             initial_radius, source_dot_radius, step) = _particle_plan(width, height)

            color = tuple(int(channel) for channel in ARGUS_BLUE_BGR)

            # Phase 1 (0.00-0.07): hold a perfect solid circle.
            # Phase 2 (0.07-0.20): resolve that circle into a circular field of dots.
            # Phase 3 (0.18-0.82): morph the coherent field into ARGUS / Engineer.
            hold_end = 0.070
            dots_end = 0.200
            morph_start = 0.180
            morph_end = 0.820

            morph = np.clip((p - morph_start) / (morph_end - morph_start), 0.0, 1.0)
            local = np.clip((morph - delay) / np.maximum(0.001, 1.0 - delay), 0.0, 1.0)
            eased = _smootherstep(local)

            position = source + (target - source) * eased[:, None]

            # A tiny transverse wave is strongest mid-morph and exactly zero at
            # both ends. This gives the living motion seen earlier in the film
            # while the overall object still moves as one coherent shape.
            travel = np.sin(np.pi * local)
            ripple = np.sin(phase + morph * np.pi * 2.5) * step * 0.16 * travel
            position = position + perpendicular * ripple[:, None]

            reveal = _smootherstep((p - hold_end) / (dots_end - hold_end))
            radii = (source_dot_radius * (1.0 - eased) + dot_radius * eased)
            pulse = (1.0 + 0.055 * np.sin(np.pi * local) *
                     np.sin(phase + morph * np.pi * 2.0))
            radii = radii * reveal * pulse
            frame = _round_dot_layer(width, height, position, radii, color)

            # Keep the solid disk fully intact at first, then dissolve it only
            # after the circular dot field underneath has become visible. This
            # guarantees a literal circle-to-dots transition with no blob frame.
            circle_alpha = 1.0 - float(_smootherstep(
                (p - hold_end) / (dots_end - hold_end)))
            if p <= hold_end:
                circle_alpha = 1.0
            if circle_alpha > 0.001:
                circle = _round_dot_layer(
                    width, height,
                    np.array([[width / 2, height / 2]], dtype=np.float32),
                    np.array([initial_radius], dtype=np.float32), color,
                )
                frame = cv2.addWeighted(circle, circle_alpha, frame,
                                        1.0 - circle_alpha, 0.0)

            return frame
    ''').lstrip()

    path.write_text(text[:start] + replacement + text[end:])


def main():
    patch_renderer()
    patch_brand()
    print("Applied true-circle film-style ARGUS outro patch")


if __name__ == "__main__":
    main()
