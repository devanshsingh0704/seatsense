import unittest

from report import CLOSE_S, MINUTES, OPEN_S, day_report, to_csv, week_dates


def at(minute):
    """second of day for a minute after office opening"""
    return OPEN_S + minute * 60


def samples(states_by_minute):
    """{minute: {desk: state}} -> the per-minute log format"""
    return [{"t": at(m) + 5, "desks": d} for m, d in sorted(states_by_minute.items())]


class OfficeHours(unittest.TestCase):
    def test_office_hours_are_0930_to_1900(self):
        self.assertEqual((OPEN_S, CLOSE_S, MINUTES), (9 * 3600 + 1800, 19 * 3600, 570))


class DayReport(unittest.TestCase):
    def test_counts_minutes_per_state_and_departures(self):
        s = samples({0: {"1": "occupied"}, 1: {"1": "occupied"}, 2: {"1": "occupied"}, 3: {"1": "vacant"}})
        r = day_report(s, ["1"])["1"]
        self.assertEqual((r["at_desk"], r["vacant"], r["times_left"]), (3, 1, 1))

    def test_minutes_without_a_sample_are_no_data_not_vacant(self):
        s = samples({0: {"1": "occupied"}})
        r = day_report(s, ["1"])["1"]
        self.assertEqual((r["at_desk"], r["vacant"], r["no_data"]), (1, 0, MINUTES - 1))

    def test_samples_outside_office_hours_are_ignored(self):
        s = [{"t": OPEN_S - 300, "desks": {"1": "occupied"}}, {"t": CLOSE_S + 60, "desks": {"1": "occupied"}}]
        self.assertEqual(day_report(s, ["1"])["1"]["at_desk"], 0)

    def test_working_and_not_working_both_count_as_at_desk(self):
        s = samples({0: {"1": "working"}, 1: {"1": "not"}, 2: {"1": "working"}})
        r = day_report(s, ["1"])["1"]
        self.assertEqual((r["working"], r["not"], r["at_desk"]), (2, 1, 3))

    def test_longest_stretch_is_unbroken_time_at_desk(self):
        s = samples({0: {"1": "working"}, 1: {"1": "not"}, 2: {"1": "vacant"}, 3: {"1": "working"}, 4: {"1": "working"}, 5: {"1": "working"}})
        self.assertEqual(day_report(s, ["1"])["1"]["longest"], 3)

    def test_segments_merge_consecutive_minutes(self):
        s = samples({0: {"1": "occupied"}, 1: {"1": "occupied"}, 2: {"1": "vacant"}})
        segs = day_report(s, ["1"])["1"]["segments"]
        self.assertEqual(segs[:2], [["occupied", 0, 2], ["vacant", 2, 3]])
        self.assertEqual(segs[2], ["nodata", 3, MINUTES])


class Week(unittest.TestCase):
    def test_week_runs_monday_to_friday_around_the_date(self):
        self.assertEqual(week_dates("2026-10-14"), ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"])


class Csv(unittest.TestCase):
    def test_csv_has_one_row_per_desk_in_minutes(self):
        s = samples({0: {"1": "occupied"}, 1: {"1": "vacant"}})
        lines = to_csv(day_report(s, ["1"]), "2026-10-16").strip().splitlines()
        self.assertEqual(lines[0], "date,desk,at_desk_min,working_min,not_working_min,vacant_min,no_data_min,times_left,longest_min")
        self.assertEqual(lines[1], f"2026-10-16,1,1,0,0,1,{MINUTES - 2},1,1")


if __name__ == "__main__":
    unittest.main()
