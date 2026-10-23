import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score import replay, score   # noqa: E402

BOXES = {1: (0, 0, 100, 100), 2: (200, 0, 300, 100)}


class Replay(unittest.TestCase):
    def test_states_follow_the_real_presence_rules(self):
        # someone seated in box 1 from t=0, nobody in box 2
        frames = [{"t": t, "persons": [[10, 10, 60, 90, 0.9]]} for t in range(0, 30)]
        states = replay(frames, BOXES, enter_s=15)
        self.assertEqual(states[10], {1: "vacant", 2: "vacant"})
        self.assertEqual(states[20], {1: "occupied", 2: "vacant"})


class Score(unittest.TestCase):
    def test_counts_false_occupied_and_false_vacant_per_box(self):
        states = {60: {1: "occupied", 2: "occupied"}, 120: {1: "vacant", 2: "vacant"}}
        labels = {60: [1], 120: [1, 2]}      # truth: box 1 sat in at both times, box 2 only at 120
        r = score(states, labels, BOXES)
        self.assertEqual((r["checks"], r["correct"], r["false_occupied"], r["false_vacant"]), (4, 1, 1, 2))
        self.assertEqual(r["by_box"][2], {"correct": 0, "false_occupied": 1, "false_vacant": 1})


if __name__ == "__main__":
    unittest.main()
