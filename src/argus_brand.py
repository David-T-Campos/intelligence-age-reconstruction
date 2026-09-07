"""Minimal ARGUS visual-brand transform for The Intelligence Age reconstruction."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ARGUS product dark blue: #0B1F3A (RGB). OpenCV stores BGR.
ARGUS_BLUE_RGB = (0x0B, 0x1F, 0x3A)
ARGUS_BLUE_BGR = np.array((0x3A, 0x1F, 0x0B), dtype=np.float32)
_WHITE_BGR = np.array((255.0, 255.0, 255.0), dtype=np.float32)


def _border_luma(frame):
    height, width = frame.shape[:2]
    band = max(1, min(height, width) // 32)
    border = np.concatenate([
        frame[:band].reshape(-1, 3),
        frame[-band:].reshape(-1, 3),
        frame[:, :band].reshape(-1, 3),
        frame[:, -band:].reshape(-1, 3),
    ], axis=0)
    b, g, r = [border[:, i].astype(np.float32) for i in range(3)]
    return float(np.median(0.114 * b + 0.587 * g + 0.299 * r))


def recolor_black_dots(frame):
    """Map neutral dark marks on light scenes from black to ARGUS blue.

    Dark-background scenes are deliberately left alone so white dots remain white
    on black, matching the request to replace the black dots rather than globally
    tinting the whole film.
    """
    output = frame.copy()
    if _border_luma(frame) < 150:
        return output

    signed = frame.astype(np.int16)
    neutral = (signed.max(axis=2) - signed.min(axis=2)) <= 10
    tone = frame.mean(axis=2).astype(np.float32) / 255.0
    mask = neutral & (tone < 0.93)
    if not np.any(mask):
        return output

    t = tone[mask, None]
    mapped = ARGUS_BLUE_BGR[None, :] * (1.0 - t) + _WHITE_BGR[None, :] * t
    output[mask] = np.clip(np.rint(mapped), 0, 255).astype(np.uint8)
    return output


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

    argus_font, argus_box = _fit_font(draw, "ARGUS", width * 0.48,
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

    draw.text(argus_xy, "ARGUS", font=argus_font, fill=ARGUS_BLUE_RGB)
    draw.text(engineer_xy, "Engineer", font=engineer_font, fill=ARGUS_BLUE_RGB)
    mask_draw.text(argus_xy, "ARGUS", font=argus_font, fill=255)
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

    delay = rng.uniform(0.0, 0.34, count).astype(np.float32)
    curve = rng.uniform(-min(width, height) * 0.14,
                        min(width, height) * 0.14, count).astype(np.float32)
    start_radius = rng.uniform(1.5, 4.5, count).astype(np.float32)
    end_radius = rng.uniform(max(2, step * 0.22),
                             max(3, step * 0.34), count).astype(np.float32)
    return final, mask, target, source, delay, curve, start_radius, end_radius


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def render_argus_outro(width, height, progress):
    """Render asynchronous dots converging into the ARGUS Engineer lockup."""
    p = float(np.clip(progress, 0.0, 1.0))
    final, mask, target, source, delay, curve, start_radius, end_radius = \
        _particle_plan(width, height)
    if p >= 0.92:
        return final.copy()

    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    color = tuple(int(channel) for channel in ARGUS_BLUE_BGR)
    local = (p - delay) / np.maximum(0.001, 0.67 - delay)
    eased = _smoothstep(local)
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

    # Every letter resolves together: particles fade into the final wordmark,
    # rather than revealing characters sequentially like typing.
    merge = float(_smoothstep((p - 0.72) / 0.20))
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
