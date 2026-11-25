
"""
test_mapper_boundary.py

Tests the mathematical boundary of the
maximum deviation criterion utilized by
TestMapper.
"""

import unittest
import pandas as pd
import numpy as np

from mapper import TestMapper


class TestMapperBoundary(unittest.TestCase):
    """Tests observations inside, close, and beyond the threshold."""

    def setUp(self):
        """Generates small datasets for boundary testing."""

        self.ideal_df = pd.DataFrame({
            "x": [1.0],
            "y13": [10.0],
            "y24": [20.0],
            "y36": [30.0],
            "y40": [40.0]
        })

        self.best_functions = {
            "y1": ("y13", 0.0, 0.5),
            "y2": ("y24", 0.0, 0.5),
            "y3": ("y36", 0.0, 0.5),
            "y4": ("y40", 0.0, 0.5)
        }

    def test_observation_inside_threshold(self):
        """Validates that an observation within the threshold is acceptable."""

        test_df = pd.DataFrame({
            "x": [1.0],
            "y": [10.5]
        })

        mapper = TestMapper(
            test_df, self.ideal_df, self.best_functions
        )

        results = mapper.map_test_data()

        self.assertEqual(len(results), 1)

    def test_observation_at_threshold(self):
        """Validates acceptance immediately below the threshold."""

        threshold = 0.5 * np.sqrt(2)

        test_y = np.nextafter(
            10.0 + threshold,
            10.0
        )

        test_df = pd.DataFrame({
            "x": [1.0],
            "y": [test_y]
        })

        mapper = TestMapper(
            test_df, self.ideal_df, self.best_functions
        )

        results = mapper.map_test_data()

        self.assertEqual(len(results), 1)

    def test_observation_outside_threshold(self):
        """Validates that an observation beyond the threshold is refuted."""

        test_df = pd.DataFrame({
            "x": [1.0],
            "y": [10.8]
        })

        mapper = TestMapper(
            test_df, self.ideal_df, self.best_functions
        )

        results = mapper.map_test_data()

        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()