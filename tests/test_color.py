import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.color import classify_color


class ColorTests(unittest.TestCase):
    def test_solid_colors(self):
        cases = [
            ("red", (0, 0, 255)),
            ("blue", (255, 0, 0)),
            ("green", (0, 255, 0)),
            ("white", (255, 255, 255)),
            ("black", (0, 0, 0)),
            ("gray", (90, 90, 90)),
            ("silver", (165, 165, 165)),
            ("yellow", (0, 255, 255)),
        ]
        for name, bgr in cases:
            with self.subTest(name=name):
                crop = np.full((30, 30, 3), bgr, dtype=np.uint8)
                self.assertEqual(classify_color(crop), name)

    def test_empty(self):
        self.assertEqual(
            classify_color(np.zeros((0, 0, 3), dtype=np.uint8)), "unknown"
        )


if __name__ == "__main__":
    unittest.main()
