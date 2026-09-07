"""Focused tests for the ARGUS visual-brand transform."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import numpy as np
from argus_brand import (apply_argus_brand, detect_outro_start,
                         recolor_black_dots, render_argus_outro)


class ArgusBrandTests(unittest.TestCase):
    def test_black_marks_on_light_background_become_argus_blue(self):
        frame = np.full((120, 160, 3), 255, np.uint8)
        cv2.circle(frame, (80, 60), 20, (0, 0, 0), -1, cv2.LINE_AA)
        recolored = recolor_black_dots(frame)
        self.assertEqual(tuple(recolored[60, 80]), (230, 85, 44))  # BGR #2C55E6
        self.assertEqual(tuple(recolored[0, 0]), (255, 255, 255))

    def test_white_dots_on_black_background_stay_unchanged(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(frame, (80, 60), 20, (255, 255, 255), -1, cv2.LINE_AA)
        np.testing.assert_array_equal(frame, recolor_black_dots(frame))

    def test_outro_detector_uses_first_sustained_blue_run(self):
        frames = [np.full((64, 96, 3), 255, np.uint8) for _ in range(30)]
        for frame in frames[22:]:
            cv2.circle(frame, (48, 32), 24, (240, 120, 20), -1)
        self.assertEqual(detect_outro_start(30, 24, lambda i: frames[i]), 22)

    def test_outro_is_deterministic_and_finishes_as_wordmark(self):
        a = render_argus_outro(640, 360, 0.55)
        b = render_argus_outro(640, 360, 0.55)
        np.testing.assert_array_equal(a, b)
        final = render_argus_outro(640, 360, 1.0)
        self.assertGreater(np.count_nonzero(np.any(final != 255, axis=2)), 1000)
        branded = apply_argus_brand(np.full((360, 640, 3), 255, np.uint8),
                                    99, 100, 80)
        np.testing.assert_array_equal(final, branded)


if __name__ == "__main__":
    unittest.main()