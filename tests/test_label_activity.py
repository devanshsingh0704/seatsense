import unittest

from labelling.label_activity import ANSWERS, shown, to_state, validate


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


class Days(unittest.TestCase):
    MANIFEST = [{"file": "a.jpg", "dir": "20261031-log"}, {"file": "b.jpg", "dir": "20261117-log"},
                {"file": "c.jpg", "dir": "20261118-log"}]

    def test_only_the_chosen_days_are_shown(self):
        self.assertEqual([m["file"] for m in shown(self.MANIFEST, ["20261117-log", "20261118-log"])], ["b.jpg", "c.jpg"])

    def test_no_days_shows_everything(self):
        self.assertEqual(shown(self.MANIFEST, []), self.MANIFEST)


if __name__ == "__main__":
    unittest.main()
