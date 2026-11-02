import unittest

from labelling.label_activity import ANSWERS, to_state, validate


class Validate(unittest.TestCase):
    def test_accepts_the_five_answers_for_known_crops(self):
        labels = {"a.jpg": "W", "b.jpg": "P", "c.jpg": "T", "d.jpg": "I", "e.jpg": "U"}
        self.assertEqual(validate(labels, set(labels)), labels)

    def test_rejects_an_unknown_answer_or_crop(self):
        with self.assertRaises(ValueError):
            validate({"a.jpg": "X"}, {"a.jpg"})
        with self.assertRaises(ValueError):
            validate({"zzz.jpg": "W"}, {"a.jpg"})


class ToState(unittest.TestCase):
    def test_phone_talking_and_idle_all_mean_not_working(self):
        self.assertEqual([to_state(a) for a in "WPTIU"], ["working", "not", "not", "not", None])

    def test_answers_cover_every_key(self):
        self.assertEqual(set(ANSWERS), set("WPTIU"))


if __name__ == "__main__":
    unittest.main()
