import unittest

from training.gate import decide, seat_answers


class Decide(unittest.TestCase):
    def test_a_candidate_as_good_as_the_live_model_is_promoted(self):
        self.assertTrue(decide(live=(30, 40), cand=(30, 40)))

    def test_a_worse_candidate_is_not(self):
        self.assertFalse(decide(live=(30, 40), cand=(29, 40)))

    def test_without_labelled_control_nothing_is_promoted(self):
        self.assertFalse(decide(live=(0, 0), cand=(0, 0)))

    def test_too_few_samples_to_trust_keeps_the_live_model(self):
        self.assertFalse(decide(live=(10, 19), cand=(19, 19), min_n=20))
        self.assertTrue(decide(live=(10, 20), cand=(19, 20), min_n=20))


class SeatAnswers(unittest.TestCase):
    def test_desks_at_or_above_half_count_as_seated(self):
        self.assertEqual(seat_answers({60: {1: 0.5, 2: 0.49, 3: 0.9}}), {"60": [1, 3]})


if __name__ == "__main__":
    unittest.main()
