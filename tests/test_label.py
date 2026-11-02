import unittest

from labelling.label import notes_for, validate_labels


class ValidateLabels(unittest.TestCase):
    def test_accepts_known_seconds_and_boxes(self):
        self.assertEqual(validate_labels({"60": [1, 3], "120": []}, {1, 2, 3}, {60, 120}),
                         {"60": [1, 3], "120": []})

    def test_rejects_a_box_that_does_not_exist(self):
        with self.assertRaises(ValueError):
            validate_labels({"60": [9]}, {1, 2, 3}, {60})

    def test_rejects_a_second_with_no_labelling_frame(self):
        with self.assertRaises(ValueError):
            validate_labels({"61": [1]}, {1, 2, 3}, {60})


class NotesFor(unittest.TestCase):
    def test_each_report_marks_only_its_nearest_frame(self):
        reports = [{"t": 59981, "desk": 3, "what": "nobody seated, shown occupied"},
                   {"t": 59990, "desk": 1, "what": "someone seated, shown vacant"}]
        self.assertEqual(notes_for([59880, 59940, 60000], reports),
                         {60000: ["You reported at 16:39:41: desk 3 — nobody seated, shown occupied",
                                  "You reported at 16:39:50: desk 1 — someone seated, shown vacant"]})

    def test_no_reports_no_notes(self):
        self.assertEqual(notes_for([60, 120], []), {})


if __name__ == "__main__":
    unittest.main()
