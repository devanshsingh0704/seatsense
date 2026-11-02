import unittest

from training.audit_activity import merge, pick


def row(file, box, y, p):
    return {"file": file, "dir": "d", "t": 0, "box": box, "y": y, "why": "P" if y else "W", "p_not": p}


class Pick(unittest.TestCase):
    ROWS = [row("missed", 1, 1, 0.05), row("caught", 1, 1, 0.95), row("borderline", 1, 1, 0.3),
            row("wrong_flag", 2, 0, 0.97), row("fine", 2, 0, 0.02), row("desk6", 6, 1, 0.01), row("desk4", 4, 0, 0.99)]

    def test_only_confident_disagreements_with_the_label(self):
        self.assertEqual(sorted(r["file"] for r in pick(self.ROWS)), ["desk4", "desk6", "missed", "wrong_flag"])

    def test_the_weak_desks_come_first(self):
        self.assertEqual([r["box"] for r in pick(self.ROWS)][:2], [4, 6])


class Merge(unittest.TestCase):
    def test_the_second_look_wins_and_is_counted(self):
        old = {"a": "W", "b": "P", "c": "W", "d": "T", "x": "W"}
        new = {"a": "W", "b": "W", "c": "U"}                  # d not looked at again; x not in the audit
        merged, n = merge(old, new)
        self.assertEqual(merged, {"a": "W", "b": "W", "c": "U", "d": "T", "x": "W"})
        self.assertEqual(n, {"same": 1, "changed": 2, "now_unclear": 1})

    def test_a_change_within_not_working_counts_as_changed(self):
        self.assertEqual(merge({"a": "P"}, {"a": "T"})[1], {"same": 0, "changed": 1, "now_unclear": 0})


if __name__ == "__main__":
    unittest.main()
