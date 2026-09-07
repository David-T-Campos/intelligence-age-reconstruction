"""Meaningful edge cases for the lossless contour format, not video fixtures."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import numpy as np
from contours import trace, rasterize, trace_frame, rasterize_frame, save, load


class ContourContractTests(unittest.TestCase):
    def test_color_channels_and_neutral_frame_compatibility(self):
        image = np.random.default_rng(52).integers(0, 256, (24, 40, 3), dtype=np.uint8)
        geometry = trace_frame(image)
        self.assertEqual(int(geometry['version'][0]), 2)
        np.testing.assert_array_equal(image, rasterize_frame(geometry))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'color.npz'
            save(path, geometry)
            np.testing.assert_array_equal(image, rasterize_frame(load(path)))
        neutral = np.repeat(image[:, :, :1], 3, axis=2)
        gray_geometry = trace_frame(neutral)
        self.assertEqual(int(gray_geometry['version'][0]), 1)
        np.testing.assert_array_equal(neutral, rasterize_frame(gray_geometry))

    def assert_roundtrip(self, original):
        geometry = trace(original)
        np.testing.assert_array_equal(original, rasterize(geometry))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.npz"
            save(path, geometry)
            np.testing.assert_array_equal(original, rasterize(load(path)))

    def test_nested_holes_touching_edges_and_single_pixels(self):
        image = np.full((80, 120), 255, np.uint8)
        cv2.circle(image, (60, 40), 30, 0, -1)
        cv2.circle(image, (60, 40), 20, 255, -1)
        cv2.circle(image, (60, 40), 8, 96, -1)
        image[:17, :12] = 63
        image[77, 118] = 0
        image[76, 117] = 127
        self.assert_roundtrip(image)

    def test_every_grayscale_value_and_disconnected_noise(self):
        image = np.random.default_rng(42).integers(0, 256, (40, 64), dtype=np.uint8)
        image[:4, :] = np.arange(256, dtype=np.uint8).reshape(4, 64)
        self.assert_roundtrip(image)

    def test_solid_frames_need_no_contours(self):
        for shade in (0, 17, 255):
            self.assert_roundtrip(np.full((16, 24), shade, np.uint8))

    def test_antialiased_deformed_point(self):
        image = np.full((160, 240), 255, np.uint8)
        cv2.ellipse(image, (119, 79), (46, 53), 17, 0, 360, 0, -1, cv2.LINE_AA)
        self.assert_roundtrip(image)


if __name__ == "__main__":
    unittest.main()
