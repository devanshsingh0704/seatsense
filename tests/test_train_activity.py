import unittest

from training.train_activity import examples


class Examples(unittest.TestCase):
    MANIFEST = [{"file": "a.jpg", "dir": "20261012-day", "t": 33000, "box": 1},
                {"file": "b.jpg", "dir": "20261012-day", "t": 33060, "box": 2},
                {"file": "c.jpg", "dir": "20261017-log", "t": 40000, "box": 3},
                {"file": "d.jpg", "dir": "20261017-log", "t": 40060, "box": 4}]

    def test_working_is_0_and_every_not_working_reason_is_1(self):
        ex = examples(self.MANIFEST, {"a.jpg": "W", "b.jpg": "P", "c.jpg": "T", "d.jpg": "I"})
        self.assertEqual([e["y"] for e in ex], [0, 1, 1, 1])
        self.assertEqual([e["why"] for e in ex], ["W", "P", "T", "I"])

    def test_unclear_and_unanswered_are_left_out(self):
        ex = examples(self.MANIFEST, {"a.jpg": "U", "b.jpg": "W"})
        self.assertEqual([e["file"] for e in ex], ["b.jpg"])


if __name__ == "__main__":
    unittest.main()
