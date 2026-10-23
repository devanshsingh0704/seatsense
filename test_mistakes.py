import unittest
from datetime import datetime

import json
import tempfile
from pathlib import Path

from mistakes import KINDS, collect, lines, parse, status

NOW = datetime(2026, 10, 23, 16, 39, 41)
DESKS = [1, 2, 3, 4, 5, 6]


class Lines(unittest.TestCase):
    def test_one_line_per_wrong_desk_from_one_report(self):
        out = lines({"3": "not_seated_shown_occupied", "1": "seated_shown_vacant"}, DESKS, NOW)
        self.assertEqual(out, ["16:39:41 desk 1 seated_shown_vacant", "16:39:41 desk 3 not_seated_shown_occupied"])

    def test_desks_left_as_correct_are_skipped(self):
        self.assertEqual(lines({"2": "ok", "4": "working_shown_not"}, DESKS, NOW), ["16:39:41 desk 4 working_shown_not"])

    def test_nothing_wrong_or_unknown_desk_or_kind_is_refused(self):
        for bad in ({}, {"2": "ok"}, {"9": "seated_shown_vacant"}, {"1": "drop table"}, {"x": "seated_shown_vacant"}):
            with self.assertRaises(ValueError):
                lines(bad, DESKS, NOW)


class Parse(unittest.TestCase):
    def test_reads_new_lines_and_free_text_lines(self):
        text = "16:39:41 desk 3 not_seated_shown_occupied\n16:35 desk 2 on phone behind monitor\n\n# note\n"
        self.assertEqual(parse(text), [
            {"t": 16 * 3600 + 39 * 60 + 41, "desk": 3, "kind": "not_seated_shown_occupied", "what": KINDS["not_seated_shown_occupied"]},
            {"t": 16 * 3600 + 35 * 60, "desk": 2, "kind": None, "what": "on phone behind monitor"}])


class Status(unittest.TestCase):
    R = {"t": 100, "desk": 3, "kind": "not_seated_shown_occupied"}

    def test_waiting_until_labelled(self):
        self.assertEqual(status(self.R, labelled=None, check=None)["status"], "waiting for labelling")

    def test_label_that_confirms_the_report_waits_for_the_retrain(self):
        self.assertEqual(status(self.R, labelled=False, check=None)["status"], "labelled, waiting for retrain")

    def test_label_that_contradicts_the_report_means_the_system_was_right(self):
        self.assertEqual(status(self.R, labelled=True, check=None)["status"], "labelled: the system was right")

    def test_after_the_retrain_fixed_or_not(self):
        self.assertEqual(status(self.R, labelled=False, check=False)["status"], "fixed")
        self.assertEqual(status(self.R, labelled=False, check=True)["status"], "not fixed yet")


class Collect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.day = Path(self.tmp.name) / "20261023-log"
        self.day.mkdir()
        (self.day / "reported.txt").write_text("16:39:41 desk 3 not_seated_shown_occupied\n16:39:41 desk 2 not_working_shown_working\n")
        (self.day / "live_d20.jsonl").write_text("\n".join(json.dumps({"t": t, "desks": {}}) for t in (59880, 59981, 60000)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_nothing_labelled_yet(self):
        out = collect(self.day, manifest=[], act_labels={})
        self.assertEqual([(r["desk"], r["t_frame"], r["status"]) for r in out],
                         [(3, 59981, "waiting for labelling"), (2, 59981, "waiting for labelling")])

    def test_seat_label_and_retrain_check(self):
        (self.day / "labels_d20.json").write_text(json.dumps({"59981": [1]}))          # desk 3 really empty
        (self.day / "reported_check.json").write_text(json.dumps({"59981": {"3": {"seat": False}}}))
        self.assertEqual(collect(self.day, [], {})[0]["status"], "fixed")

    def test_activity_label_comes_from_the_crop(self):
        man = [{"file": "c.jpg", "dir": "20261023-log", "t": 59981, "box": 2}]
        self.assertEqual(collect(self.day, man, {"c.jpg": "P"})[1]["status"], "labelled, waiting for retrain")
        self.assertEqual(collect(self.day, man, {"c.jpg": "U"})[1]["status"], "waiting for labelling")

    def test_missing_files_mean_no_reports(self):
        self.assertEqual(collect(Path(self.tmp.name) / "nope", [], {}), [])


if __name__ == "__main__":
    unittest.main()
