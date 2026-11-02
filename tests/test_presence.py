import json
import tempfile
import unittest
from pathlib import Path

from seatsense.presence import SEAT_ENTER, SEAT_STAY, Desk, assign, load_boxes, seat_present, seat_threshold


class LoadBoxes(unittest.TestCase):
    def test_drops_exact_duplicates_and_keeps_the_users_numbering(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "d3_focus.json"
            f.write_text(json.dumps({"source_size": [2560, 1440],
                                     "regions": [[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]]}))
            self.assertEqual(load_boxes(f), {1: (0, 0, 10, 10), 3: (20, 20, 30, 30)})


class Assign(unittest.TestCase):
    BOXES = {1: (0, 0, 100, 100), 2: (80, 0, 200, 100)}

    def test_person_fully_inside_a_box(self):
        self.assertEqual(assign([(10, 10, 50, 50)], self.BOXES), {1: [0], 2: []})

    def test_person_in_an_overlap_goes_to_the_box_covering_more_of_them(self):
        # 85..125: 15 px inside box 1, all 40 px inside box 2
        self.assertEqual(assign([(85, 10, 125, 50)], self.BOXES), {1: [], 2: [0]})

    def test_person_outside_every_box_is_dropped(self):
        self.assertEqual(assign([(300, 300, 350, 350)], self.BOXES), {1: [], 2: []})

    def test_standing_person_sticking_out_of_the_box_is_dropped(self):
        # measured on D3: people standing in front of a desk were at most 77% inside its box
        self.assertEqual(assign([(10, -20, 60, 80)], {1: (0, 0, 100, 100)}), {1: []})   # 80% inside

    def test_seated_person_almost_fully_inside_counts(self):
        # measured on D20: seated people were 89-100% inside their box
        self.assertEqual(assign([(10, -5, 60, 95)], {1: (0, 0, 100, 100)}), {1: [0]})   # 95% inside

    def test_person_mostly_outside_is_dropped(self):
        # only 30% of the person is inside box 1
        self.assertEqual(assign([(-70, 10, 30, 50)], {1: (0, 0, 100, 100)}), {1: []})


class SeatPresent(unittest.TestCase):
    def test_the_classifier_decides_when_no_visitor_is_involved(self):
        self.assertTrue(seat_present(0.9, visitor_inside=False, staff_inside=False))
        self.assertFalse(seat_present(0.2, visitor_inside=False, staff_inside=False))

    def test_a_visitor_alone_in_the_box_never_occupies_it(self):
        self.assertFalse(seat_present(0.95, visitor_inside=True, staff_inside=False))

    def test_staff_sitting_next_to_a_visitor_still_counts(self):
        self.assertTrue(seat_present(0.95, visitor_inside=True, staff_inside=True))


class DeskOccupancy(unittest.TestCase):
    def test_becomes_occupied_only_after_enter_seconds_of_presence(self):
        d = Desk(enter_s=3, leave_s=20)
        self.assertIsNone(d.update(True, 0))
        self.assertIsNone(d.update(True, 2))
        self.assertEqual(d.update(True, 3), ("occupied", 0, None))
        self.assertEqual(d.state, "occupied")

    def test_someone_walking_past_for_9_seconds_does_not_occupy_the_desk(self):
        # the real false alarm on D3 box 8: presence for ~9 s, then gone
        d = Desk()
        events = [d.update(True, t / 2) for t in range(0, 19)] + [d.update(False, t) for t in range(10, 60)]
        self.assertEqual([e for e in events if e], [])
        self.assertEqual(d.state, "vacant")

    def test_a_missed_detection_does_not_restart_the_entry_count(self):
        d = Desk(enter_s=15, gap_s=3)
        for t in range(0, 8):
            d.update(True, t)
        d.update(False, 8)            # one frame with no detection
        for t in range(9, 15):
            d.update(True, t)
        self.assertEqual(d.update(True, 15), ("occupied", 0, None))

    def test_a_gap_longer_than_gap_s_restarts_the_entry_count(self):
        d = Desk(enter_s=15, gap_s=3)
        for t in range(0, 8):
            d.update(True, t)
        d.update(False, 12)           # 5 s with no one
        for t in range(12, 20):
            self.assertIsNone(d.update(True, t))
        self.assertEqual(d.state, "vacant")

    def test_default_marks_vacant_10_seconds_after_last_seen(self):
        d = Desk()
        for t in range(0, 30):
            d.update(True, t)
        self.assertIsNone(d.update(False, 38))           # 9 s after last seen at 29: still occupied
        self.assertEqual(d.update(False, 39)[0], "vacant")

    def test_a_short_gap_does_not_end_occupancy(self):
        d = Desk(enter_s=3, leave_s=20)
        for t in range(0, 10):
            d.update(True, t)
        self.assertIsNone(d.update(False, 25))  # 16 s since last seen at 9
        self.assertEqual(d.state, "occupied")

    def test_vacant_after_leave_seconds_counts_a_departure_and_the_time(self):
        d = Desk(enter_s=3, leave_s=20)
        for t in range(0, 61):
            d.update(True, t)
        self.assertEqual(d.update(False, 81), ("vacant", 60, 60))
        self.assertEqual((d.state, d.departures, d.occupied_s), ("vacant", 1, 60))

    def test_running_total_includes_the_current_stretch(self):
        d = Desk(enter_s=3, leave_s=20)
        for t in range(0, 11):
            d.update(True, t)
        self.assertEqual(d.occupied_total(30), 30)


class SeatThresholds(unittest.TestCase):
    """Two thresholds: stronger evidence to take a desk (stops standing / leaning-in), the usual 0.5 to keep it."""

    def feed(self, desk, p, t0, t1):
        for t in range(t0, t1):
            desk.update(seat_present(p, False, False, seat_threshold(desk.state)), t)

    def test_values(self):
        self.assertEqual((seat_threshold("vacant"), seat_threshold("occupied")), (SEAT_ENTER, SEAT_STAY))
        self.assertEqual((SEAT_ENTER, SEAT_STAY), (0.75, 0.5))

    def test_a_middling_score_does_not_take_a_free_desk(self):
        d = Desk()
        self.feed(d, 0.6, 0, 60)
        self.assertEqual(d.state, "vacant")

    def test_a_confident_score_takes_it_and_a_middling_one_keeps_it(self):
        d = Desk()
        self.feed(d, 0.9, 0, 20)
        self.assertEqual(d.state, "occupied")
        self.feed(d, 0.6, 20, 80)                    # a seated person leaning or turning: stays occupied
        self.assertEqual(d.state, "occupied")
        self.feed(d, 0.1, 80, 100)                   # really gone
        self.assertEqual(d.state, "vacant")


if __name__ == "__main__":
    unittest.main()
