"""Minimal ARGUS visual-brand transform for The Intelligence Age reconstruction."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ARGUS primary product blue: #2C55E6 (RGB). OpenCV stores BGR.
ARGUS_BLUE_RGB = (0x2C, 0x55, 0xE6)
ARGUS_BLUE_BGR = np.array((0xE6, 0x55, 0x2C), dtype=np.float32)
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


def _dot_components(core_mask):
    """Keep bounded pointillist components while rejecting page/field components."""
    binary = core_mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(core_mask, dtype=bool)

    height, width = core_mask.shape
    max_area = int(height * width * 0.24)
    tiny_border_area = int(height * width * 0.008)

    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))
    touches_border = np.zeros(count, dtype=bool)
    touches_border[border_labels] = True

    areas = stats[:, cv2.CC_STAT_AREA]
    eligible = (areas > 0) & (areas <= max_area)
    eligible &= (~touches_border) | (areas <= tiny_border_area)
    eligible[0] = False
    return eligible[labels]


def recolor_black_dots(frame):
    """Keep both black and white pointillist marks ARGUS blue through inversions.

    Light scenes preserve the original broad black/gray-to-blue transform that
    already covered all of the black dot fields. Dark scenes map white/gray marks
    to the same ARGUS blue. A bounded bright-component pass catches white or
    slightly tinted white dots inside colored transition frames without tinting
    the large white page/background component.
    """
    source = frame.astype(np.float32)
    spread = source.max(axis=2) - source.min(axis=2)
    neutral = spread <= 32.0

    b, g, r = [source[:, :, i] for i in range(3)]
    tone = (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
    output = source.copy()

    if _border_luma(frame) < 150:
        # Dark mode: keep the black field black and map every white/gray mark to
        # ARGUS blue, including antialiased and slightly tinted pointillist edges.
        mask = neutral & (tone > 0.07)
        if np.any(mask):
            t = tone[mask, None]
            output[mask] = ARGUS_BLUE_BGR[None, :] * t
        return np.clip(np.rint(output), 0, 255).astype(np.uint8)

    # Light mode: preserve the original full black/gray-dot recolor rather than
    # filtering by connected-component size. This avoids regressing large dot
    # fields that visually merge into one illustration.
    dark_marks = neutral & (tone < 0.93)
    if np.any(dark_marks):
        t = tone[dark_marks, None]
        mapped = ARGUS_BLUE_BGR[None, :] * (1.0 - t) + _WHITE_BGR[None, :] * t
        output[dark_marks] = mapped

    # White dots can appear over blue/colored transition fields even while the
    # page border remains light. Reject the giant background component, retain
    # bounded bright marks, and color their antialiased edges smoothly.
    bright_core = neutral & (tone >= 0.72)
    bright_marks = _dot_components(bright_core)
    if np.any(bright_marks):
        core_f = bright_marks.astype(np.float32)
        soft = cv2.GaussianBlur(core_f, (0, 0), sigmaX=1.25, sigmaY=1.25)
        alpha = np.maximum(core_f, np.clip(soft * 0.90, 0.0, 0.82))[:, :, None]
        output = output * (1.0 - alpha) + ARGUS_BLUE_BGR[None, None, :] * alpha

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
    """Prefer a modern Lato lockup; retain platform-native fallbacks."""
    if bold:
        return (
            Path("/usr/share/fonts/truetype/lato/Lato-Heavy.ttf"),
            Path("/usr/share/fonts/truetype/lato/Lato-Semibold.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("/System/Library/Fonts/SFNS.ttf"),
            Path("/Library/Fonts/Arial Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        )
    return (
        Path("/usr/share/fonts/truetype/lato/Lato-Medium.ttf"),
        Path("/usr/share/fonts/truetype/lato/Lato-Regular.ttf"),
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
    """Build only a typography mask; the final logo is always rendered as dots."""
    mask_image = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask_image)

    argus_font, argus_box = _fit_font(mask_draw, "Argus", width * 0.46,
                                      height * 0.19, bold=True)
    engineer_font, engineer_box = _fit_font(mask_draw, "Engineer", width * 0.25,
                                            height * 0.068, bold=False)
    argus_width = argus_box[2] - argus_box[0]
    argus_height = argus_box[3] - argus_box[1]
    engineer_width = engineer_box[2] - engineer_box[0]
    engineer_height = engineer_box[3] - engineer_box[1]
    gap = max(18, round(height * 0.027))
    total_height = argus_height + gap + engineer_height
    top = round(height * 0.51 - total_height / 2)

    argus_xy = ((width - argus_width) / 2 - argus_box[0],
                top - argus_box[1])
    engineer_xy = ((width - engineer_width) / 2 - engineer_box[0],
                   top + argus_height + gap - engineer_box[1])

    mask_draw.text(argus_xy, "Argus", font=argus_font, fill=255)
    mask_draw.text(engineer_xy, "Engineer", font=engineer_font, fill=255)
    return np.asarray(mask_image)


@lru_cache(maxsize=8)
def _particle_plan(width, height):
    """Plan a deterministic dot-cloud burst from the center into the wordmark."""
    mask = _wordmark(width, height)
    step = max(6, round(min(width, height) * 0.0065))
    ys, xs = np.mgrid[step // 2:height:step, step // 2:width:step]
    flat_x, flat_y = xs.ravel(), ys.ravel()
    inside = mask[flat_y, flat_x] > 96
    target = np.stack([flat_x[inside], flat_y[inside]], axis=1).astype(np.float32)

    rng = np.random.default_rng(0xA69E)
    if len(target) > 3600:
        target = target[rng.choice(len(target), 3600, replace=False)]
    count = len(target)

    center = np.array([width / 2, height / 2], dtype=np.float32)
    initial_radius = min(width, height) * 0.082

    # All particles begin *inside* the last blue dot. As the dot dissolves,
    # these points become visible and peel away from the same physical location.
    start_angle = rng.uniform(0.0, 2.0 * np.pi, count)
    start_radius = initial_radius * np.sqrt(rng.uniform(0.0, 1.0, count))
    source = np.column_stack([
        center[0] + np.cos(start_angle) * start_radius,
        center[1] + np.sin(start_angle) * start_radius,
    ]).astype(np.float32)

    # First control point throws particles outward in a broad, asymmetric bloom.
    burst_angle = start_angle + rng.normal(0.0, 0.34, count)
    burst_radius = rng.uniform(min(width, height) * 0.16,
                               min(width, height) * 0.34, count)
    control1 = np.column_stack([
        center[0] + np.cos(burst_angle) * burst_radius,
        center[1] + np.sin(burst_angle) * burst_radius,
    ]).astype(np.float32)

    # Second control point bends those trajectories back toward their letter
    # destinations, creating a flowing magnetic-field / data-stream motion.
    inward = target - center
    length = np.linalg.norm(inward, axis=1) + 1e-6
    perpendicular = np.column_stack([-inward[:, 1] / length,
                                      inward[:, 0] / length])
    lateral = rng.uniform(-min(width, height) * 0.12,
                           min(width, height) * 0.12, count)
    control2 = (target - inward * rng.uniform(0.04, 0.13, count)[:, None] +
                perpendicular * lateral[:, None]).astype(np.float32)

    delay = rng.uniform(0.0, 0.11, count).astype(np.float32)
    arrival = rng.uniform(0.82, 0.90, count).astype(np.float32)
    dot_radius = rng.uniform(max(1.35, step * 0.18),
                             max(2.1, step * 0.29), count).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, count).astype(np.float32)
    return (mask, target, source, control1, control2, delay, arrival,
            dot_radius, phase, float(initial_radius))


def _smootherstep(value):
    """Quintic ease with zero velocity and acceleration at both endpoints."""
    value = np.clip(value, 0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _bezier(source, control1, control2, target, t):
    """Vectorized cubic Bezier positions for one parameter per particle."""
    t = np.asarray(t, dtype=np.float32)[:, None]
    one = 1.0 - t
    return (one ** 3 * source +
            3.0 * one ** 2 * t * control1 +
            3.0 * one * t ** 2 * control2 +
            t ** 3 * target)


def _brand_blue_source(frame):
    """Normalize the source outro's saturated blue to ARGUS blue for handoff."""
    output = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = ((hue >= 85) & (hue <= 125) &
            (saturation >= 55) & (value >= 70))
    if np.any(mask):
        output[mask] = np.clip(np.rint(ARGUS_BLUE_BGR), 0, 255).astype(np.uint8)
    return output


def render_argus_outro(width, height, progress, source_frame=None):
    """Dissolve the last blue dot into a fluid cloud that remains a dotted logo.

    There is deliberately no solid-wordmark stage. The final Argus / Engineer
    lockup is itself a high-density pointillist object, so the visual language
    stays technological all the way through the last frame.
    """
    p = float(np.clip(progress, 0.0, 1.0))
    (mask, target, source, control1, control2, delay, arrival,
     dot_radius, phase, initial_radius) = _particle_plan(width, height)

    white = np.full((height, width, 3), 255, dtype=np.uint8)
    frame = white.copy()
    color = tuple(int(channel) for channel in ARGUS_BLUE_BGR)

    # Each point starts inside the original dot, blooms outward, then settles
    # independently into the letter matrix. Arrival is finished early enough to
    # leave a clean dotted lockup on screen instead of morphing into solid text.
    local = (p - delay) / np.maximum(0.001, arrival - delay)
    eased = _smootherstep(local)
    position = _bezier(source, control1, control2, target, eased)

    # A very short ghost-dot trail makes 24 fps motion read substantially more
    # fluidly without introducing line graphics that break the pointillist style.
    moving = 1.0 - float(_smootherstep((p - 0.79) / 0.10))
    if moving > 0.001:
        previous = np.clip(eased - 0.030, 0.0, 1.0)
        trail_position = _bezier(source, control1, control2, target, previous)
        trail = frame.copy()
        for particle in range(len(target)):
            radius = max(1, int(round(dot_radius[particle] * 0.72)))
            cv2.circle(trail,
                       (int(round(trail_position[particle, 0])),
                        int(round(trail_position[particle, 1]))),
                       radius, color, -1, cv2.LINE_AA)
        frame = cv2.addWeighted(trail, 0.16 * moving, frame,
                                1.0 - 0.16 * moving, 0.0)

    # Subtle radius breathing while particles are in flight. It disappears as
    # the wordmark settles, so the final dotted type is crisp and stable.
    flight = 1.0 - _smootherstep((p - 0.72) / 0.16)
    radii = dot_radius * (1.0 + 0.08 * np.sin(phase + p * np.pi * 5.0) * flight)
    for particle in range(len(target)):
        cv2.circle(frame,
                   (int(round(position[particle, 0])),
                    int(round(position[particle, 1]))),
                   max(1, int(round(radii[particle]))),
                   color, -1, cv2.LINE_AA)

    # Keep the single dot continuous at the handoff, then let it visibly break
    # apart as the already-present internal particles escape from its boundary.
    circle_alpha = 1.0 - float(_smootherstep(p / 0.18))
    if circle_alpha > 0.001:
        dot_layer = frame.copy()
        shrink = 1.0 - 0.10 * float(_smootherstep(p / 0.18))
        cv2.circle(dot_layer, (round(width / 2), round(height / 2)),
                   max(1, round(initial_radius * shrink)), color, -1, cv2.LINE_AA)
        frame = cv2.addWeighted(dot_layer, circle_alpha, frame,
                                1.0 - circle_alpha, 0.0)

    # For only the first few frames, crossfade from the actual source frame so
    # the previous blue-dot shot and our generated breakup are temporally joined
    # rather than separated by a hard edit.
    if source_frame is not None and p < 0.06:
        handoff = 1.0 - float(_smootherstep(p / 0.06))
        branded_source = _brand_blue_source(source_frame)
        frame = np.clip(branded_source.astype(np.float32) * handoff +
                        frame.astype(np.float32) * (1.0 - handoff),
                        0, 255).astype(np.uint8)

    return frame


def apply_argus_brand(frame, index, frame_count, outro_start):
    if index >= outro_start:
        denominator = max(1, frame_count - 1 - outro_start)
        return render_argus_outro(frame.shape[1], frame.shape[0],
                                  (index - outro_start) / denominator,
                                  source_frame=frame)
    return recolor_black_dots(frame)
