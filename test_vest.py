import unittest

import numpy as np

from vest import is_visitor, vest_fraction

VEST_BGR = (40, 255, 200)    # fluorescent yellow-green, like the "VISITOR" vests
GREY_BGR = (128, 128, 128)


def person(colour, size=(400, 200)):
    img = np.zeros((*size, 3), np.uint8)
    img[:] = colour
    return img


class VestFraction(unittest.TestCase):
    def test_a_person_in_a_vest_scores_high(self):
        self.assertGreater(vest_fraction(person(VEST_BGR), (0, 0, 200, 400)), 0.9)

    def test_a_person_in_a_grey_shirt_scores_zero(self):
        self.assertEqual(vest_fraction(person(GREY_BGR), (0, 0, 200, 400)), 0.0)

    def test_only_the_torso_counts_not_the_background_beside_them(self):
        img = person(GREY_BGR, (400, 600))
        img[:, :100] = VEST_BGR          # a yellow-green cabinet to the left, outside the person
        self.assertEqual(vest_fraction(img, (200, 0, 400, 400)), 0.0)

    def test_empty_or_degenerate_box_is_not_a_visitor(self):
        self.assertFalse(is_visitor(person(VEST_BGR), (10, 10, 10, 10)))


class IsVisitor(unittest.TestCase):
    def test_threshold_sits_between_staff_and_vest_wearers(self):
        # measured: staff <= 0.005, vest wearers 0.047-0.30
        self.assertTrue(is_visitor(person(VEST_BGR), (0, 0, 200, 400)))
        self.assertFalse(is_visitor(person(GREY_BGR), (0, 0, 200, 400)))


if __name__ == "__main__":
    unittest.main()
