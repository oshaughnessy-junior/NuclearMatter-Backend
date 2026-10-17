"""Focused acceptance checks for the disposable transport fixture."""
import unittest
from fixture import weighted_mean


class WeightedMeanContract(unittest.TestCase):
    def test_unequal_weights(self):
        self.assertAlmostEqual(weighted_mean([2.0, 8.0], [1.0, 3.0]), 6.5)

    def test_common_weight_scale(self):
        values = [2.0, 8.0]
        self.assertAlmostEqual(weighted_mean(values, [1.0, 3.0]),
                               weighted_mean(values, [10.0, 30.0]))

    def test_single_value(self):
        self.assertAlmostEqual(weighted_mean([7.0], [4.0]), 7.0)


if __name__ == '__main__':
    unittest.main()
