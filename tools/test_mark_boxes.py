import unittest

from mark_boxes import to_regions


class ToRegions(unittest.TestCase):
    def test_scales_screen_boxes_back_to_source_pixels(self):
        # window shown at half size: a 100x50 box at (10, 20) is 200x100 at (20, 40) in the source
        self.assertEqual(to_regions([(10, 20, 100, 50)], 0.5), [[20, 40, 220, 140]])

    def test_drops_accidental_tiny_boxes_and_exact_repeats(self):
        self.assertEqual(to_regions([(10, 20, 100, 50), (5, 5, 3, 3), (10, 20, 100, 50)], 1.0),
                         [[10, 20, 110, 70]])


if __name__ == "__main__":
    unittest.main()
