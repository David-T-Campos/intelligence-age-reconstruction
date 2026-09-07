"""Minimal ARGUS visual-brand transform for The Intelligence Age reconstruction."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ARGUS primary product blue: #2C55E6 (RGB). OpenCV stores BGR.
ARGUS_BLUE_RGB = (0x2C, 0x55, 0xE6)
ARGUS_BLUE_BGR = np.array((0xE6, 0x55, 0x2C), dtype=np.float32)


def _border_pixels(frame):
    height, width = frame.shape[:2]
    band = max(1, min(height, width) // 32)
    return np.concatenate([
        frame[:band].reshape(-1, 3),
        frame[-band:].reshape(-1, 3),
        frame[:, :band].reshape(-1, 3),
        frame[:, -band:].reshape(-1, 3),
    ], axis=0)


def _background_bgr(frame):
    """Estimate the current flat field from the image perimeter."""
    return np.median(_border_pixels(frame).astype(np.float32), axis=0)


def recolor_black_dots(frame):
    """Keep the pointillist foreground ARGUS blue through every inversion.

    The original film continuously flips between dark marks on a light field,
    bright marks on a dark field, and intermediate gray/color transition fields.
    A fixed black-only or white-only threshold therefore fails during inversions.

    Instead, estimate the current background from the frame perimeter and recolor
    neutral foreground marks according to their contrast from that background.
    The background itself stays untouched, while black dots, white dots and their
    antialiased gray edges all converge on the same ARGUS blue (#2C55E6).
    """
    source = frame.astype(np.float32)
    output = source.copy()
    background = _background_bgr(frame)

    # The pointillist artwork is essentially neutral before branding. Allow a
    # little chroma for H.264/upscale ringing so white dots do not slip through.
    spread = source.max(axis=2) - source.min(axis=2)
    neutral = spread <= 28.0

    b, g, r = [source[:, :, i] for i in range(3)]
    tone = (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
    bg_b, bg_g, bg_r = [float(background[i]) for i in range(3)]
    bg_tone = (0.114 * bg_b + 0.587 * bg_g + 0.299 * bg_r) / 255.0

    # Luma contrast handles white<->black inversions. RGB distance also catches
    # neutral white/black dots sitting on a colored (for example pale-blue) field.
    luma_contrast = np.abs(tone - bg_tone)
    rgb_contrast = np.linalg.norm(source - background[None, None, :], axis=2) / (255.0 * np.sqrt(3.0))
    contrast = np.maximum(luma_contrast, rgb_contrast)

    # Ignore tiny codec/background fluctuations. Above ~19% contrast the mark is
    # treated as a full foreground dot; lower-contrast antialiasing blends smoothly.
    strength = np.clip((contrast - 0.03) / 0.16, 0.0, 1.0)
    strength = strength * strength * (3.0 - 2.0 * strength)
    mask = neutral & (strength > 0.0)
    if not np.any(mask):
        return frame.copy()

    alpha = strength[mask, None]
    output[mask] = source[mask] * (1.0 - alpha) + ARGUS_BLUE_BGR[None, :] * alpha
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _blue_fraction(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue = ((hue >= 85) & (hue <= 125) &
            (saturation >= 55) & (value >= 70))
    return float(blue.mean())


def detect_outro_start(frame_count, fps, frame_loader):
    """Find the original saturated-blue ChatGPT outro inside the final 8 s."""
    scan = max(1, min(frame_count, round(fps * 8)))
    first = frame_count - scan
    run = 0
    for index in range(first, frame_count):
        if _blue_fraction(frame_loader(index)) >= 0.012:
            run += 1
            if run >= 3:
                return index - run + 1
        else:
            run = 0
    return max(0, frame_count - round(fps * 3.5))


def _font_candidates(bold):
    if bold:
        return (
            Path("C:/Windows/Fonts/segoeuib.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("/System/Library/Fonts/SFNS.ttf"),
            Path("/Library/Fonts/Arial Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        )
    return (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/System/Library/Fonts/SFNS.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )


def _font(size, bold=False):
    for path in _font_candidates(bold):
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    for name in (("DejaVuSans-Bold.ttf",) if bold else ("DejaVuSans.ttf",)):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_font(draw, text, max_width, initial_size, bold=False):
    size = max(10, int(initial_size))
    while size > 10:
        font = _font(size, bold)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= max_width:
            return font, box
        size -= max(1, size // 24)
    font = _font(size, bold)
    return font, draw.textbbox((0, 0), text, font=font)


@lru_cache(maxsize=8)
def _wordmark(width, height):
    image = Image.new("RGB", (width, height), "white")
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask_image)

    argus_font, argus_box = _fit_font(draw, "Argus", width * 0.48,
                                      height * 0.20, bold=True)
    engineer_font, engineer_box = _fit_font(draw, "Engineer", width * 0.26,
                                            height * 0.074, bold=False)
    argus_width = argus_box[2] - argus_box[0]
    argus_height = argus_box[3] - argus_box[1]
    engineer_width = engineer_box[2] - engineer_box[0]
    engineer_height = engineer_box[3] - engineer_box[1]
    gap = max(16, round(height * 0.025))
    total_height = argus_height + gap + engineer_height
    top = round(height * 0.51 - total_height / 2)

    argus_xy = ((width - argus_width) / 2 - argus_box[0],
                top - argus_box[1])
    engineer_xy = ((width - engineer_width) / 2 - engineer_box[0],
                   top + argus_height + gap - engineer_box[1])

    draw.text(argus_xy, "Argus", font=argus_font, fill=ARGUS_BLUE_RGB)
    draw.text(engineer_xy, "Engineer", font=engineer_font, fill=ARGUS_BLUE_RGB)
    mask_draw.text(argus_xy, "Argus", font=argus_font, fill=255)
    mask_draw.text(engineer_xy, "Engineer", font=engineer_font, fill=255)

    final_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    mask = np.asarray(mask_image)
    return final_bgr, mask


@lru_cache(maxsize=8)
def _particle_plan(width, height):
    final, mask = _wordmark(width, height)
    step = max(10, round(min(width, height) * 0.013))
    ys, xs = np.mgrid[step // 2:height:step, step // 2:width:step]
    flat_x, flat_y = xs.ravel(), ys.ravel()
    inside = mask[flat_y, flat_x] > 80
    target = np.stack([flat_x[inside], flat_y[inside]], axis=1).astype(np.float32)

    rng = np.random.default_rng(0xA69E)
    if len(target) > 2600:
        target = target[rng.choice(len(target), 2600, replace=False)]
    count = len(target)
    center = np.array([width / 2, height / 2], dtype=np.float32)
    angle = rng.uniform(0, 2 * np.pi, count)
    radius = rng.uniform(min(width, height) * 0.18,
                         min(width, height) * 0.78, count)
    source = np.column_stack([
        center[0] + np.cos(angle) * radius,
        center[1] + np.sin(angle) * radius,
    ]).astype(np.float32)
    source[:, 0] = np.clip(source[:, 0], 20, width - 20)
    source[:, 1] = np.clip(source[:, 1], 20, height - 20)

    # Keep the asynchronous feel, but reduce the delay spread and arc size so
    # adjacent 24 fps frames flow more continuously into the final lockup.
    delay = rng.uniform(0.0, 0.22, count).astype(np.float32)
    curve = rng.uniform(-min(width, height) * 0.10,
                        min(width, height) * 0.10, count).astype(np.float32)
    start_radius = rng.uniform(1.5, 4.5, count).astype(np.float32)
    end_radius = rng.uniform(max(2, step * 0.22),
                             max(3, step * 0.34), count).astype(np.float32)
    return final, mask, target, source, delay, curve, start_radius, end_radius


def _smootherstep(value):
    """Quintic ease with zero velocity and acceleration at both endpoints."""
    value = np.clip(value, 0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def render_argus_outro(width, height, progress):
    """Render fluid asynchronous dots converging into the Argus Engineer lockup."""
    p = float(np.clip(progress, 0.0, 1.0))
    final, mask, target, source, delay, curve, start_radius, end_radius = \
        _particle_plan(width, height)
    if p >= 0.95:
        return final.copy()

    white = np.full((height, width, 3), 255, dtype=np.uint8)
    frame = white.copy()
    color = tuple(int(channel) for channel in ARGUS_BLUE_BGR)

    # Each particle gets a slightly different start, but all settle gently by
    # roughly the same phase instead of stopping early and then snapping away.
    arrival = 0.80 + delay * 0.10
    local = (p - delay) / np.maximum(0.001, arrival - delay)
    eased = _smootherstep(local)
    visible = p > delay

    delta = target - source
    length = np.linalg.norm(delta, axis=1) + 1e-6
    perpendicular = np.column_stack([-delta[:, 1] / length,
                                      delta[:, 0] / length])
    arc = np.sin(np.pi * eased) * curve
    position = source + delta * eased[:, None] + perpendicular * arc[:, None]
    radius = start_radius + (end_radius - start_radius) * eased

    for particle in np.flatnonzero(visible):
        cv2.circle(frame,
                   (int(round(position[particle, 0])),
                    int(round(position[particle, 1]))),
                   max(1, int(round(radius[particle]))),
                   color, -1, cv2.LINE_AA)

    # Fade the free particles away only as the complete wordmark resolves.
    # This removes the previous hard disappearance near the end of the morph.
    particle_fade = float(_smootherstep((p - 0.76) / 0.19))
    if particle_fade > 0.0:
        frame = np.clip(frame.astype(np.float32) * (1.0 - particle_fade) +
                        white.astype(np.float32) * particle_fade,
                        0, 255).astype(np.uint8)

    # Every letter resolves together (not character-by-character / typing).
    merge = float(_smootherstep((p - 0.72) / 0.23))
    if merge > 0.0:
        alpha = (mask.astype(np.float32) / 255.0 * merge)[:, :, None]
        frame = np.clip(frame.astype(np.float32) * (1.0 - alpha) +
                        final.astype(np.float32) * alpha,
                        0, 255).astype(np.uint8)
    return frame


def apply_argus_brand(frame, index, frame_count, outro_start):
    if index >= outro_start:
        denominator = max(1, frame_count - 1 - outro_start)
        return render_argus_outro(frame.shape[1], frame.shape[0],
                                  (index - outro_start) / denominator)
    return recolor_black_dots(frame)
