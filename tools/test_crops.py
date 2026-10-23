import unittest

from crops import crop_box, pick_people

BOXES = {1: (0, 0, 100, 100), 2: (200, 0, 300, 100)}


class PickPeople(unittest.TestCase):
    def test_keeps_people_near_a_desk_with_a_sit_or_stand_guess(self):
        persons = [[10, 10, 60, 90, 0.9],      # fully inside box 1 -> guess sit
                   [180, -60, 260, 100, 0.8],  # half inside box 2 -> guess stand
                   [500, 500, 550, 600, 0.9]]  # nowhere near a desk -> skipped
        got = pick_people(persons, BOXES)
        self.assertEqual([(g["box"], g["guess"]) for g in got], [(1, "sit"), (2, "stand")])

    def test_ignores_low_confidence_detections(self):
        self.assertEqual(pick_people([[10, 10, 60, 90, 0.2]], BOXES, min_conf=0.35), [])


class CropBox(unittest.TestCase):
    def test_adds_a_margin_so_the_chair_and_desk_show(self):
        self.assertEqual(crop_box((100, 100, 200, 300), 2560, 1440, margin=0.2), (80, 60, 220, 340))

    def test_margin_is_clamped_to_the_frame(self):
        self.assertEqual(crop_box((0, 0, 100, 100), 2560, 1440, margin=0.2), (0, 0, 120, 120))


if __name__ == "__main__":
    unittest.main()
