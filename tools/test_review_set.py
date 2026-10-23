import unittest

from review_set import latest, select


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


class Latest(unittest.TestCase):
    def test_a_filled_in_answer_replaces_the_stale_one_for_that_minute(self):
        rows = [{"t": 120, "src": "stale"}, {"t": 60, "src": "live"}, {"t": 120, "src": "offline"}]
        self.assertEqual([(r["t"], r["src"]) for r in latest(rows)], [(60, "live"), (120, "offline")])


class Calibrated(unittest.TestCase):
    QUIET = dict(d1=("vacant", 0.0), d2=("vacant", 0.0), d3=("vacant", 0.0), d4=("vacant", 0.0), d5=("vacant", 0.0), d6=("vacant", 0.0))

    # minute 0 is calm and becomes the control sample (the first remaining minute), as in the tests above

    def test_an_over_confident_score_counts_as_unsure_once_calibrated(self):
        rows = [row(0, **self.QUIET), row(100, **{**self.QUIET, "d3": ("vacant", 0.01)})]   # raw 1% -> about 9% at T=2
        self.assertEqual(select(rows, reported=[], every=1000), [0])
        self.assertEqual(select(rows, reported=[], every=1000, temperature=2.0), [0, 100])

    def test_the_budget_keeps_the_most_uncertain_minutes(self):
        rows = [row(0, **self.QUIET)] + [row(t, **{**self.QUIET, "d3": ("vacant", p)})
                                         for t, p in ((60, 0.2), (120, 0.5), (180, 0.9), (240, 0.45))]
        self.assertEqual(select(rows, reported=[], every=1000, max_unsure=2), [0, 120, 240])


if __name__ == "__main__":
    unittest.main()
