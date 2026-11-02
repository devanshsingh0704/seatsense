import unittest
from datetime import date

from scripts.prune import doomed, old_enough


class Doomed(unittest.TestCase):
    FRAMES = ["d20_34200.jpg", "d20_34260.jpg", "d20_34320.jpg", "d20_34380.jpg", "d20_34440.jpg"]

    def test_only_frames_never_picked_for_labelling_go(self):
        keep = {34200, 34320, 34440}        # labelled, in the labelling set, an activity crop
        self.assertEqual(doomed(self.FRAMES, keep), ["d20_34260.jpg", "d20_34380.jpg"])

    def test_a_day_nobody_picked_anything_from_loses_every_frame(self):
        self.assertEqual(doomed(self.FRAMES, set()), self.FRAMES)


class OldEnough(unittest.TestCase):
    def test_only_days_past_the_grace_period(self):
        today = date(2026, 11, 10)
        self.assertTrue(old_enough("20261027-log", today, days=14))
        self.assertFalse(old_enough("20261028-log", today, days=14))

    def test_folders_without_a_date_are_left_alone(self):
        self.assertFalse(old_enough("activity", date(2026, 11, 10), days=14))


if __name__ == "__main__":
    unittest.main()
