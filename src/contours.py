"""Lossless tracing of neutral and color frames to integer-grid contour geometry.

A frame contains XY vertices, contour offsets and grayscale layer offsets.
It contains no bitmap, encoded frame, source-video path or image texture.
Neutral frames use v1; color frames use v2 with three independent v1 planes.
OpenCV's integer-grid even/odd fill convention is part of the file contract.
Motion is captured as discrete frame geometry at the source frame rate.
This is reference-assisted rotoscoping, not procedural motion synthesis.
"""
from pathlib import Path
import cv2
import numpy as np

FORMAT_VERSION = 1


def trace_frame(frame: np.ndarray) -> dict:
    """Keep neutral frames compact; trace independent B/G/R contours for color."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected an 8-bit BGR frame")
    if np.array_equal(frame[:, :, 0], frame[:, :, 1]) and np.array_equal(frame[:, :, 1], frame[:, :, 2]):
        return trace(frame[:, :, 0])
    geometry = {"version": np.array([2], dtype=np.uint8),
                "size": np.array(frame.shape[1::-1], dtype=np.uint16)}
    for channel, prefix in enumerate(("b", "g", "r")):
        for key, value in trace(frame[:, :, channel]).items():
            geometry[f"{prefix}_{key}"] = value
    return geometry


def rasterize_frame(geometry: dict) -> np.ndarray:
    """Always return BGR, including for the accepted v1 monochrome assets."""
    version = int(geometry["version"][0])
    if version == 1:
        return cv2.cvtColor(rasterize(geometry), cv2.COLOR_GRAY2BGR)
    if version != 2:
        raise ValueError("Unknown frame geometry version")
    channels = []
    for prefix in ("b", "g", "r"):
        channel = {key[2:]: value for key, value in geometry.items() if key.startswith(prefix + "_")}
        channels.append(rasterize(channel))
    return np.stack(channels, axis=2)


def trace(gray: np.ndarray) -> dict:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected an 8-bit grayscale frame")
    values = np.unique(gray)
    background = int(values[-1])
    vertices = []
    contour_ends = [0]
    layer_ends = [0]
    shades = []
    for shade in values[-2::-1]:
        mask = np.asarray(gray <= shade, dtype=np.uint8)
        paths, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for path in paths:
            points = path.reshape(-1, 2)
            vertices.append(points)
            contour_ends.append(contour_ends[-1] + len(points))
        layer_ends.append(layer_ends[-1] + len(paths))
        shades.append(int(shade))
    return {
        "version": np.array([FORMAT_VERSION], dtype=np.uint8),
        "size": np.array(gray.shape[::-1], dtype=np.uint16),
        "background": np.array([background], dtype=np.uint8),
        "vertices": (np.concatenate(vertices).astype(np.int16) if vertices
                     else np.empty((0, 2), dtype=np.int16)),
        "contour_ends": np.array(contour_ends, dtype=np.uint32),
        "layer_ends": np.array(layer_ends, dtype=np.uint32),
        "shades": np.array(shades, dtype=np.uint8),
    }


def rasterize(geometry: dict) -> np.ndarray:
    if int(geometry["version"][0]) != FORMAT_VERSION:
        raise ValueError("Unknown geometry format")
    w, h = map(int, geometry["size"])
    dst = np.full((h, w), int(geometry["background"][0]), dtype=np.uint8)
    points = geometry["vertices"].astype(np.int32)
    ends = geometry["contour_ends"]
    paths = [points[a:b].reshape(-1, 1, 2) for a, b in zip(ends[:-1], ends[1:])]
    layer_ends = geometry["layer_ends"]
    for shade, a, b in zip(geometry["shades"], layer_ends[:-1], layer_ends[1:]):
        # A whole layer is drawn in one operation so interior holes are retained.
        # LINE_AA must not be used: edge gray values already have their own layers.
        cv2.drawContours(dst, paths[a:b], -1, int(shade), cv2.FILLED,
                         lineType=cv2.LINE_8)
    return dst


def save(path: Path, geometry: dict) -> None:
    np.savez_compressed(path, **geometry)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def dot_summary(gray: np.ndarray) -> list:
    """Midtone silhouettes for inspection; not used to substitute ideal circles."""
    background_is_white = np.count_nonzero(gray > 127) > gray.size // 2
    mask = (gray < 128) if background_is_white else (gray >= 128)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2:
            continue
        moments = cv2.moments(contour)
        perimeter = cv2.arcLength(contour, True)
        shapes.append({
            "center": [round(moments["m10"] / moments["m00"], 3),
                       round(moments["m01"] / moments["m00"], 3)],
            "area": area,
            "bbox": list(cv2.boundingRect(contour)),
            "circularity": round(4 * np.pi * area / perimeter ** 2, 5),
        })
    return shapes
