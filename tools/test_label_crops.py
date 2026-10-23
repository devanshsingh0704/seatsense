import unittest

from label_crops import validate_crop_labels


class ValidateCropLabels(unittest.TestCase):
    def test_accepts_sit_stand_unclear_for_known_crops(self):
        labels = {"d20_00020_0.jpg": "sit", "d3_00040_1.jpg": "stand", "d3_00060_0.jpg": "unclear"}
        self.assertEqual(validate_crop_labels(labels, set(labels)), labels)

    def test_rejects_an_unknown_answer(self):
        with self.assertRaises(ValueError):
            validate_crop_labels({"d20_00020_0.jpg": "lying"}, {"d20_00020_0.jpg"})

    def test_rejects_a_crop_that_does_not_exist(self):
        with self.assertRaises(ValueError):
            validate_crop_labels({"nope.jpg": "sit"}, {"d20_00020_0.jpg"})


if __name__ == "__main__":
    unittest.main()
