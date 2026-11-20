
"""
test_exceptions.py

Unit tests for validating exception handling in
the TestMapper class.
"""

import unittest
import pandas as pd

from mapper import TestMapper
from exceptions import InvalidDatasetError


class TestMapperExceptions(unittest.TestCase):
    """Tests exception managment in TestMapper."""

    def setUp(self):
        """Generates valid datasets and selected ideal functions."""

        self.ideal_df = pd.DataFrame({
            "x": [1.0, 2.0],
            "y13": [10.0, 20.0],
            "y24": [30.0, 40.0],
            "y36": [50.0, 60.0],
            "y40": [70.0, 80.0]
        })

        self.best_functions = {
            "y1": ("y13", 0.0, 0.5),
            "y2": ("y24", 0.0, 0.5),
            "y3": ("y36", 0.0, 0.5),
            "y4": ("y40", 0.0, 0.5)
        }

        self.test_df = pd.DataFrame({
            "x": [1.0],
            "y": [10.2]
        })

    def test_duplicate_ideal_x_coordinates(self):
        """Validates duplicate ideal X-coordinates raises an exception."""

        invalid_ideal_df = pd.DataFrame({
            "x": [1.0, 1.0],
            "y13": [10.0, 20.0],
            "y24": [30.0, 40.0],
            "y36": [50.0, 60.0],
            "y40": [70.0, 80.0]
        })

        with self.assertRaises(InvalidDatasetError):
            TestMapper(
                self.test_df,
                invalid_ideal_df,
                self.best_functions
            )

    def test_missing_ideal_function(self):
        """Validates that a missing selected ideal function raises ValueError."""

        invalid_functions = self.best_functions.copy()
        invalid_functions["y1"] = ("y99", 0.0, 0.5)

        with self.assertRaises(ValueError):
            TestMapper(
                self.test_df,
                self.ideal_df,
                invalid_functions
            )

    def test_missing_test_x_coordinate(self):
        """Validates an unknown test X-coordinate raises an exception."""

        invalid_test_df = pd.DataFrame({
            "x": [3.0],
            "y": [10.2]
        })

        mapper = TestMapper(
            invalid_test_df,
            self.ideal_df,
            self.best_functions
        )

        with self.assertRaises(InvalidDatasetError):
            mapper.map_test_data()


if __name__ == "__main__":
    unittest.main()
