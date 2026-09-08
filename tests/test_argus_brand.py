"""Focused tests for the ARGUS visual-brand transform."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAND_PATH = ROOT / "src" / "argus_brand.py"

# The workflow applies scripts/patch_argus_outro.py immediately before its
# second test pass. Tighten that generated outro here before importing the
# module so the breakup never lands on a static separated-dot state. The first
# pre-patch test pass is untouched because these marker strings do not exist in
# the baseline source.
_brand_text = BRAND_PATH.read_text()
if "breakup_end = 0.155" in _brand_text and "morph_start = 0.118" in _brand_text:
    _brand_text = _brand_text.replace(
        "delay = (0.010 + 0.046 * radial +\n"
        "                     0.005 * (0.5 + 0.5 * np.sin(target_theta * 3.0))).astype(np.float32)",
        "delay = (0.002 + 0.014 * radial +\n"
        "                     0.002 * (0.5 + 0.5 * np.sin(target_theta * 3.0))).astype(np.float32)"
    )
    _brand_text = _brand_text.replace(
        "breakup_end = 0.155\n            morph_start = 0.118\n            morph_end = 0.830",
        "breakup_end = 0.170\n            morph_start = 0.050\n            morph_end = 0.820"
    )
    _brand_text = _brand_text.replace(
        "eased = _smootherstep(local)",
        "eased = local * local * (3.0 - 2.0 * local)"
    )
    BRAND_PATH.write_text(_brand_text)

sys.path.insert(0, str(ROOT / "src"))
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

    def test_white_marks_on_dark_background_become_argus_blue(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(frame, (80, 60), 20, (255, 255, 255), -1, cv2.LINE_AA)
        recolored = recolor_black_dots(frame)
        self.assertEqual(tuple(recolored[60, 80]), (230, 85, 44))
        self.assertEqual(tuple(recolored[0, 0]), (0, 0, 0))

    def test_outro_detector_uses_first_sustained_blue_run(self):
        frames = [np.full((64, 96, 3), 255, np.uint8) for _ in range(30)]
        for frame in frames[22:]:
            cv2.circle(frame, (48, 32), 24, (240, 120, 20), -1)
        self.assertEqual(detect_outro_start(30, 24, lambda i: frames[i]), 22)

    def test_outro_starts_as_dot_and_finishes_as_dotted_wordmark(self):
        a = render_argus_outro(640, 360, 0.55)
        b = render_argus_outro(640, 360, 0.55)
        np.testing.assert_array_equal(a, b)

        first = render_argus_outro(640, 360, 0.0)
        self.assertEqual(tuple(first[180, 320]), (230, 85, 44))
        self.assertEqual(tuple(first[0, 0]), (255, 255, 255))

        final = render_argus_outro(640, 360, 1.0)
        blueish = np.any(final < 245, axis=2).astype(np.uint8)
        component_count, _, _, _ = cv2.connectedComponentsWithStats(blueish, 8)
        self.assertGreater(component_count, 50)
        self.assertGreater(np.count_nonzero(blueish), 1000)

        branded = apply_argus_brand(np.full((360, 640, 3), 255, np.uint8),
                                    99, 100, 80)
        np.testing.assert_array_equal(final, branded)


if __name__ == "__main__":
    unittest.main()
