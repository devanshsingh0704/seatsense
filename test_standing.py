import unittest

from standing import leg_ratio, veto


def person(box, shoulder_y, hip_y, knee_y, conf=0.9):
    kp = [[0, 0, 0.0]] * 17
    kp = [list(p) for p in kp]
    for i in (5, 6):
        kp[i] = [100, shoulder_y, conf]
    for i in (11, 12):
        kp[i] = [100, hip_y, conf]
    for i in (13, 14):
        kp[i] = [100, knee_y, conf]
    return {"box": box, "kp": kp}


class LegRatio(unittest.TestCase):
    def test_standing_has_knees_far_below_hips(self):
        self.assertAlmostEqual(leg_ratio(person([0, 0, 100, 400], 100, 200, 300)), 1.0)

    def test_sitting_has_knees_near_hip_level(self):
        self.assertAlmostEqual(leg_ratio(person([0, 0, 100, 300], 100, 200, 220)), 0.2)

    def test_unknown_when_knees_are_not_visible(self):
        self.assertIsNone(leg_ratio(person([0, 0, 100, 300], 100, 200, 300, conf=0.2)))


class Veto(unittest.TestCase):
    BOX = (0, 0, 200, 400)

    def test_vetoes_when_everyone_in_the_box_is_standing(self):
        self.assertTrue(veto([person([10, 0, 100, 400], 100, 200, 300)], self.BOX, threshold=0.6))

    def test_keeps_the_desk_if_anyone_in_it_is_sitting(self):
        people = [person([10, 0, 100, 400], 100, 200, 300), person([110, 0, 190, 300], 100, 200, 220)]
        self.assertFalse(veto(people, self.BOX, threshold=0.6))

    def test_never_vetoes_on_an_unknown_posture_or_an_empty_box(self):
        self.assertFalse(veto([person([10, 0, 100, 300], 100, 200, 300, conf=0.2)], self.BOX, threshold=0.6))
        self.assertFalse(veto([], self.BOX, threshold=0.6))

    def test_people_mostly_outside_the_box_are_ignored(self):
        self.assertFalse(veto([person([500, 0, 600, 400], 100, 200, 300)], self.BOX, threshold=0.6))


if __name__ == "__main__":
    unittest.main()
