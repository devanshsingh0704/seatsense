import unittest

from daily import activity_score, missing, pick_activity, seat_control, seat_score


def seated(n, p):
    return [{"t": 60 * i, "box": 1, "p_not": p} for i in range(n)]


class PickActivity(unittest.TestCase):
    def test_unsure_crops_are_picked(self):
        rows = seated(5, 0.02) + [{"t": 999, "box": 2, "p_not": 0.5}]
        picked = pick_activity(rows, seed="d", control_n=0)
        self.assertEqual([(p["t"], p["unsure"]) for p in picked], [(999, True)])

    def test_too_many_unsure_keeps_those_closest_to_half(self):
        rows = [{"t": i, "box": 1, "p_not": p} for i, p in enumerate([0.2, 0.5, 0.8, 0.45])]
        picked = pick_activity(rows, seed="d", max_unsure=2, control_n=0)
        self.assertEqual(sorted(p["t"] for p in picked), [1, 3])

    def test_control_is_a_random_sample_of_all_seated_crops(self):
        rows = seated(100, 0.01)
        picked = pick_activity(rows, seed="d", control_n=10)
        self.assertEqual(len(picked), 10)
        self.assertTrue(all(p["control"] and not p["unsure"] for p in picked))
        self.assertEqual(picked, pick_activity(rows, seed="d", control_n=10))    # same day, same sample

    def test_an_unsure_crop_can_also_be_in_the_control(self):
        picked = pick_activity([{"t": 0, "box": 1, "p_not": 0.5}], seed="d", control_n=5)
        self.assertEqual(picked, [{"t": 0, "box": 1, "p_not": 0.5, "unsure": True, "control": True}])


class SeatControl(unittest.TestCase):
    def test_every_nth_logged_minute_regardless_of_what_the_model_said(self):
        self.assertEqual(seat_control([{"t": 60 * i} for i in range(10)], every=4), [0, 240, 480])


class Missing(unittest.TestCase):
    def test_frames_without_a_live_answer_in_time_order(self):
        self.assertEqual(missing(frames=[300, 60, 120, 180], rows=[{"t": 60}, {"t": 180}]), [120, 300])


class Scores(unittest.TestCase):
    def test_seat_score_counts_desk_minutes_on_the_control_only(self):
        pre = {"0": [1], "60": [1, 2], "120": [5]}
        labels = {"0": [1], "60": [1], "120": [1]}
        # minute 0: 6/6 right; minute 60: desk 2 wrong -> 5/6; minute 120 is not control
        self.assertEqual(seat_score(pre, labels, control=[0, 60], desks=6), (11, 12))

    def test_seat_score_skips_control_minutes_not_yet_labelled(self):
        self.assertEqual(seat_score({"0": [1]}, {}, control=[0], desks=6), (0, 0))

    def test_activity_score_uses_labelled_control_crops_and_drops_unclear(self):
        man = [{"file": "a", "control": True, "p_not": 0.9}, {"file": "b", "control": True, "p_not": 0.1},
               {"file": "c", "control": True, "p_not": 0.1}, {"file": "d", "control": False, "p_not": 0.9},
               {"file": "e", "control": True, "p_not": 0.9}]
        labels = {"a": "P", "b": "T", "c": "W", "d": "W", "e": "U"}
        # a right (not), b wrong (missed talking), c right; d not control; e unclear
        self.assertEqual(activity_score(man, labels), {"right": 2, "total": 3, "working": [1, 1], "not": [1, 2]})


if __name__ == "__main__":
    unittest.main()
