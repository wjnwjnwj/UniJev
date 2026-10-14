import unittest
from .grouping import bin_index, fit_bins, calibrate


class GroupingTests(unittest.TestCase):
    def observations(self, count=20):
        return [{"key": ("question", i), "confidence": .85, "correct": i < 15}
                for i in range(count)]

    def test_user_twenty_questions_fifteen_correct(self):
        fitted = fit_bins(self.observations())
        self.assertEqual(fitted["groups"][8]["target"], .75)
        self.assertEqual(fitted["groups"][8]["count"], 20)
        self.assertEqual(calibrate(.81, fitted), .75)
        self.assertEqual(calibrate(.89, fitted), .75)

    def test_wrong_predictions_receive_same_group_target(self):
        rows = self.observations()
        fitted = fit_bins(rows)
        self.assertEqual([calibrate(x["confidence"], fitted) for x in rows], [.75] * 20)
        # The true candidate is deliberately not an input to the lookup function.

    def test_boundaries(self):
        for value, expected in [(0., 0), (.1, 1), (.8, 8), (.9, 9), (1., 9)]:
            self.assertEqual(bin_index(value), expected)

    def test_sparse_and_empty_bins_preserve_confidence(self):
        fitted = fit_bins(self.observations(19))
        self.assertEqual(calibrate(.85, fitted), .85)
        self.assertEqual(calibrate(.45, fitted), .45)

    def test_targets_do_not_move_with_later_inputs(self):
        fitted = fit_bins(self.observations())
        for _ in range(100):
            self.assertEqual(calibrate(.85, fitted), .75)
        self.assertEqual(fitted["groups"][8]["count"], 20)

    def test_invalid_and_repeated_questions_fail(self):
        for value in [float("nan"), float("inf"), -.1, 1.1, True]:
            with self.assertRaises(ValueError):
                bin_index(value)
        rows = self.observations()
        with self.assertRaisesRegex(ValueError, "Repeated"):
            fit_bins(rows + [rows[0]])
        with self.assertRaises(ValueError):
            fit_bins([{"key": 1, "confidence": .8, "correct": .5}])


if __name__ == "__main__":
    unittest.main()
