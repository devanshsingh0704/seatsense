import math
import unittest

from evaluate import binary_metrics, calibrated, ece, fit_temperature, load_temperatures, logit, minutes_error, pr_table, scale


class BinaryMetrics(unittest.TestCase):
    def test_counts_and_rates_for_the_positive_class(self):
        m = binary_metrics(y=[1, 1, 1, 0, 0, 0, 0, 0], pred=[1, 1, 0, 1, 0, 0, 0, 0])
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (2, 1, 1, 4))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)
        self.assertAlmostEqual(m["accuracy"], 6 / 8)

    def test_no_positive_predictions_gives_zero_precision_not_a_crash(self):
        self.assertEqual(binary_metrics(y=[1, 0], pred=[0, 0])["precision"], 0.0)


class MinutesError(unittest.TestCase):
    def test_one_frame_per_minute_per_desk_and_day(self):
        rows = [{"dir": "d1", "box": 1, "y": 1, "p": 0.9}, {"dir": "d1", "box": 1, "y": 1, "p": 0.2},
                {"dir": "d1", "box": 1, "y": 0, "p": 0.7}, {"dir": "d1", "box": 2, "y": 0, "p": 0.1}]
        out = minutes_error(rows)
        self.assertEqual(out[("d1", 1)], {"labelled": 2, "predicted": 2, "error": 0, "missed": 1, "extra": 1})
        self.assertEqual(out[("d1", 2)], {"labelled": 0, "predicted": 0, "error": 0, "missed": 0, "extra": 0})


class Calibration(unittest.TestCase):
    def test_logit_is_clamped_so_rounded_0_and_1_stay_finite(self):
        self.assertTrue(math.isfinite(logit(0.0)) and math.isfinite(logit(1.0)))
        self.assertAlmostEqual(logit(0.5), 0.0)

    def test_temperature_above_1_for_an_overconfident_model(self):
        # says 99% but is right only 75% of the time -> must be softened
        z = [logit(0.99)] * 4 + [logit(0.01)] * 4
        y = [1, 1, 1, 0, 0, 0, 0, 1]
        self.assertGreater(fit_temperature(z, y), 1.5)

    def test_scale_divides_the_logit(self):
        self.assertAlmostEqual(scale(logit(0.9), 1.0), 0.9)
        self.assertLess(scale(logit(0.9), 2.0), 0.9)

    def test_ece_is_zero_when_confidence_matches_frequency(self):
        p = [0.25] * 4 + [0.75] * 4
        y = [1, 0, 0, 0, 1, 1, 1, 0]
        self.assertAlmostEqual(ece(p, y, bins=10)[0], 0.0)

    def test_ece_of_an_overconfident_model(self):
        self.assertAlmostEqual(ece([0.9, 0.9], [1, 0], bins=10)[0], 0.4)


class StoredTemperatures(unittest.TestCase):
    def test_missing_file_means_no_calibration(self):
        from pathlib import Path
        self.assertEqual(load_temperatures(Path("/nonexistent/calibration.json")), {"seat": 1.0, "activity": 1.0})

    def test_calibrated_keeps_none_and_the_50_percent_point(self):
        self.assertIsNone(calibrated(None, 2.0))
        self.assertAlmostEqual(calibrated(0.5, 3.0), 0.5)
        self.assertGreater(calibrated(0.01, 2.0), 0.05)


class PrTable(unittest.TestCase):
    def test_raising_the_threshold_trades_recall_for_precision(self):
        p = [0.9, 0.8, 0.6, 0.4, 0.3, 0.1]
        y = [1, 1, 0, 1, 0, 0]
        t = {r["threshold"]: r for r in pr_table(p, y, [0.5, 0.7])}
        self.assertAlmostEqual(t[0.5]["precision"], 2 / 3); self.assertAlmostEqual(t[0.5]["recall"], 2 / 3)
        self.assertAlmostEqual(t[0.7]["precision"], 1.0); self.assertAlmostEqual(t[0.7]["recall"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
