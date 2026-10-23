import unittest

from train_seat import build_examples, fold_key, fold_of


class BuildExamples(unittest.TestCase):
    def test_one_example_per_box_per_labelled_frame(self):
        labels = {"32460": [2], "32520": []}
        ex = build_examples(labels, boxes=[1, 2, 3])
        self.assertEqual(len(ex), 6)
        self.assertEqual({(e["t"], e["box"]): e["sitting"] for e in ex},
                         {(32460, 1): 0, (32460, 2): 1, (32460, 3): 0, (32520, 1): 0, (32520, 2): 0, (32520, 3): 0})


class FoldOf(unittest.TestCase):
    def test_whole_half_hours_stay_in_one_fold(self):
        # 09:00-09:29 is one block, 09:30-09:59 the next
        self.assertEqual(fold_of(9 * 3600), fold_of(9 * 3600 + 29 * 60))
        self.assertNotEqual(fold_of(9 * 3600), fold_of(9 * 3600 + 30 * 60))

    def test_blocks_rotate_over_the_folds(self):
        folds = {fold_of(9 * 3600 + k * 1800) for k in range(8)}
        self.assertEqual(folds, {0, 1, 2, 3})


class FoldKey(unittest.TestCase):
    E = {"t": 9 * 3600, "dir": "recordings/20261012-day"}

    def test_default_folds_are_half_hour_blocks(self):
        self.assertEqual(fold_key(self.E, by_day=False), fold_of(9 * 3600))

    def test_by_day_holds_out_whole_days(self):
        self.assertEqual(fold_key(self.E, by_day=True), "recordings/20261012-day")
        self.assertNotEqual(fold_key(self.E, by_day=True), fold_key({**self.E, "dir": "recordings/20261017-log"}, by_day=True))


if __name__ == "__main__":
    unittest.main()
