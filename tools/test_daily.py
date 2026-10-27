import unittest

from daily import activity_score, missing, reported_activity, reported_times, pick_activity, seat_control, seat_score, trim_manifest


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


class PickActivityCalibrated(unittest.TestCase):
    def test_temperature_widens_what_counts_as_unsure(self):
        rows = [{"t": 0, "box": 1, "p_not": 0.05}]                       # raw 5% -> about 25% at T=3
        self.assertEqual(pick_activity(rows, seed="d", control_n=0), [])
        self.assertEqual(len(pick_activity(rows, seed="d", control_n=0, temperature=3.0)), 1)


class SeatControl(unittest.TestCase):
    def test_n_evenly_spaced_logged_minutes_regardless_of_what_the_model_said(self):
        self.assertEqual(seat_control([{"t": 60 * i} for i in range(10)], n=3), [0, 180, 360])

    def test_a_short_day_keeps_every_minute(self):
        self.assertEqual(seat_control([{"t": 60 * i} for i in range(3)], n=30), [0, 60, 120])


class TrimManifest(unittest.TestCase):
    def test_a_day_behind_keeps_only_control_reported_and_already_labelled_crops(self):
        man = [{"file": "a", "dir": "d1", "control": False},
               {"file": "b", "dir": "d1", "control": True},
               {"file": "c", "dir": "d1", "control": False, "reported": True},
               {"file": "e", "dir": "d1", "control": False},
               {"file": "f", "dir": "d0", "control": False}]
        self.assertEqual([m["file"] for m in trim_manifest(man, "d1", labels={"e": "W"})], ["b", "c", "e", "f"])


class Missing(unittest.TestCase):
    def test_frames_without_a_live_answer_in_time_order(self):
        self.assertEqual(missing(frames=[300, 60, 120, 180], rows=[{"t": 60}, {"t": 180}]), [120, 300])

    def test_a_stale_or_offline_live_answer_counts_as_missing(self):
        rows = [{"t": 60, "age": 1.2, "online": True},
                {"t": 120, "age": 400.0, "online": True},     # detector frozen: old answer
                {"t": 180, "age": 1.0, "online": False},      # camera offline
                {"t": 240, "age": None, "online": True},      # just started, no answer yet
                {"t": 300}]                                   # logged before ages were recorded: trusted
        self.assertEqual(missing(frames=[60, 120, 180, 240, 300], rows=rows), [120, 180, 240])


class Reported(unittest.TestCase):
    def test_times_the_user_reported_one_per_line(self):
        self.assertEqual(reported_times("16:39 desk 3 standing\n\n# note\n09:05:30\n"), [16 * 3600 + 39 * 60, 9 * 3600 + 5 * 60 + 30])


class ReportedActivity(unittest.TestCase):
    REPORTS = [{"t": 100, "desk": 2, "kind": "not_working_shown_working"},
               {"t": 100, "desk": 3, "kind": "not_seated_shown_occupied"},    # a seat report: not an activity crop
               {"t": 5000, "desk": 1, "kind": "working_shown_not"}]           # no logged frame near it

    def test_activity_reports_become_crops_at_the_nearest_logged_frame(self):
        self.assertEqual(reported_activity(self.REPORTS, frame_ts=[60, 130, 200], have=set()), [(130, 2)])

    def test_a_crop_already_picked_is_not_added_twice(self):
        self.assertEqual(reported_activity(self.REPORTS, frame_ts=[60, 130], have={(130, 2)}), [])


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
