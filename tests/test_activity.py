import unittest

from seatsense.activity import NOT_AT, Activity


def feed(a, seq):
    """seq: list of (t, p_not); returns the state after each"""
    return [a.update(p, t) for t, p in seq]


class ActivityState(unittest.TestCase):
    def test_a_newly_seated_person_starts_as_working(self):
        self.assertEqual(Activity().update(0.1, 0), "working")

    def test_not_working_only_after_20_seconds(self):
        a = Activity()
        states = feed(a, [(t, 0.9) for t in range(0, 21)])
        self.assertEqual(states[19], "working")            # 19 s of "not working" looks: still working
        self.assertEqual(states[20], "not")

    def test_a_short_misread_does_not_restart_the_20_seconds(self):
        a = Activity()
        feed(a, [(t, 0.9) for t in range(0, 10)])
        a.update(0.2, 10)                                   # one working-looking second
        self.assertEqual(feed(a, [(t, 0.9) for t in range(11, 21)])[-1], "not")

    def test_back_to_working_after_10_seconds(self):
        a = Activity()
        feed(a, [(t, 0.9) for t in range(0, 21)])
        states = feed(a, [(t, 0.1) for t in range(21, 32)])
        self.assertEqual(states[8], "not")                  # 9 s working-looking: still not working
        self.assertEqual(states[10], "working")

    def test_reset_when_the_desk_empties(self):
        a = Activity()
        feed(a, [(t, 0.9) for t in range(0, 21)])
        a.reset()
        self.assertEqual(a.update(0.9, 100), "working")


class Threshold(unittest.TestCase):
    def test_not_working_needs_a_score_of_0_7(self):
        self.assertEqual(NOT_AT, 0.7)
        a = Activity()
        self.assertEqual(feed(a, [(t, 0.65) for t in range(0, 40)])[-1], "working")    # unsure: no flag
        self.assertEqual(feed(a, [(t, 0.8) for t in range(40, 61)])[-1], "not")


if __name__ == "__main__":
    unittest.main()
