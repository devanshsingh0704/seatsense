import unittest

from review_set import select


def row(t, **desks):
    """desks: n=(state, p)"""
    return {"t": t, "desks": {k[1:]: list(v) for k, v in desks.items()}}


class Select(unittest.TestCase):
    CONFIDENT = dict(d1=("occupied", 1.0), d2=("vacant", 0.0), d3=("vacant", 0.0), d4=("vacant", 0.0), d5=("vacant", 0.0), d6=("vacant", 0.0))

    def test_every_third_minute_with_overlapping_neighbours_occupied_is_picked(self):
        both = {**Select.CONFIDENT, "d4": ("occupied", 1.0)}           # desks 1 and 4 together
        rows = [row(t, **both) for t in range(0, 360, 60)]
        # 0 and 180 are every third such minute; 60 is the first remaining minute, kept as the control
        self.assertEqual(select(rows, reported=[], every=1000), [0, 60, 180])

    def test_unsure_minutes_are_picked(self):
        rows = [row(100, **{**self.CONFIDENT, "d3": ("vacant", 0.4)})]
        self.assertEqual(select(rows, reported=[], every=1000), [100])

    def test_desk_5_occupied_is_picked(self):
        rows = [row(100, **{**self.CONFIDENT, "d5": ("occupied", 0.99)})]
        self.assertEqual(select(rows, reported=[], every=1000), [100])

    def test_reported_times_pick_the_nearest_minute(self):
        rows = [row(t, **self.CONFIDENT) for t in range(0, 600, 60)]
        # the reported minute 240 and two either side; 0 is the first confident minute, kept as the control sample
        self.assertEqual(select(rows, reported=[245], every=1000), [0, 120, 180, 240, 300, 360])

    def test_every_nth_confident_minute_is_kept_as_a_control(self):
        rows = [row(t, **self.CONFIDENT) for t in range(0, 600, 60)]
        self.assertEqual(select(rows, reported=[], every=5), [0, 300])


if __name__ == "__main__":
    unittest.main()
