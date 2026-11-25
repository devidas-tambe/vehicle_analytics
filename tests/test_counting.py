import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.counting import LineCounter


class CountingTests(unittest.TestCase):
    def test_down_once(self):
        c = LineCounter(0.5, 0.02)
        result = [c.observe(1, y) for y in [0.3, 0.49, 0.51, 0.55, 0.3, 0.7]]
        self.assertEqual(result, [None, None, None, "down", None, None])

    def test_up(self):
        c = LineCounter()
        self.assertIsNone(c.observe(1, 0.8))
        self.assertEqual(c.observe(1, 0.3), "up")

    def test_first_observation(self):
        c = LineCounter()
        self.assertIsNone(c.observe(1, 0.55))
        self.assertIsNone(c.observe(1, 0.8))

    def test_multiple_sources(self):
        c = LineCounter()
        self.assertIsNone(c.observe(("a", 1), 0.2))
        self.assertIsNone(c.observe(("b", 1), 0.8))
        self.assertEqual(c.observe(("a", 1), 0.8), "down")
        self.assertEqual(c.observe(("b", 1), 0.2), "up")

    def test_invalid_boundary(self):
        with self.assertRaises(ValueError):
            LineCounter(1.0)

    def test_deadband_jitter(self):
        c = LineCounter(0.5, 0.02)
        result = [c.observe("x", y) for y in [0.4, 0.49, 0.51, 0.48, 0.47]]
        self.assertEqual(result, [None] * 5)


if __name__ == "__main__":
    unittest.main()
