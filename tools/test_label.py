import unittest

from label import validate_labels


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


if __name__ == "__main__":
    unittest.main()
