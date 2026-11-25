
"""
test_function_selector.py

Unit test for validating ideal function selection,
result structure, and numerical computations.
"""

import unittest
import pandas as pd

from data_loader import DataLoader
from function_selector import FunctionSelector


class TestFunctionSelector(unittest.TestCase):
    """Tests the functionality of FunctionSelector."""

    def setUp(self):
        """Loading of datasets and starts the function selector."""

        loader = DataLoader()

        self.training_df = loader.load_training_data()
        self.ideal_df = loader.load_ideal_data()

        self.selector = FunctionSelector(
            self.training_df,
            self.ideal_df
        )

    def test_selector_created(self):
        """Validates that the selector is successfully initialized."""

        self.assertIsNotNone(self.selector)

    def test_best_functions_return_dictionary(self):
        """Validates that the selected functions are returned as a dictionary."""

        best_functions = self.selector.select_best_functions()

        self.assertIsInstance(best_functions, dict)

    def test_number_of_selected_functions(self):
        """Validates that exactly four ideal functions are chosen."""

        best_functions = self.selector.select_best_functions()

        self.assertEqual(len(best_functions), 4)

    def test_training_function_keys(self):
        """Validates the four training functions keys."""

        best_functions = self.selector.select_best_functions()

        expected_keys = ["y1", "y2", "y3", "y4"]

        self.assertEqual(list(best_functions.keys()), expected_keys)

    def test_selected_ideal_functions(self):
        """Validates that the correct ideal functions are chosen."""

        best_functions = self.selector.select_best_functions()

        expected_functions = {
            "y1": "y13",
            "y2": "y24",
            "y3": "y36",
            "y4": "y40"
        }

        actual_functions = {
            training_function: result[0]
            for training_function, result in best_functions.items()
        }

        self.assertEqual(actual_functions, expected_functions)

    def test_best_function_value_is_tuple(self):
        """Validates that every selected function is a tuple."""

        best_functions = self.selector.select_best_functions()

        for value in best_functions.values():
            self.assertIsInstance(value, tuple)

    def test_tuple_contains_three_elements(self):
        """Validates that every result tuple consists of three elements."""

        best_functions = self.selector.select_best_functions()

        for value in best_functions.values():
            self.assertEqual(len(value), 3)

    def test_sse_is_numeric(self):
        """Validates that SSE values are numeric."""

        best_functions = self.selector.select_best_functions()

        for value in best_functions.values():
            self.assertIsInstance(value[1], (int, float))

    def test_maximum_deviation_is_numeric(self):
        """Validates that maximum_deviations are numeric."""

        best_functions = self.selector.select_best_functions()

        for value in best_functions.values():
            self.assertIsInstance(value[2], (int, float))

    def test_sse_and_maximum_deviation_calculation(self):
        """Validates the flawlessness of SSE and maximum_deviation computations."""

        training_values = pd.Series([1.0, 2.0, 3.0])
        ideal_values = pd.Series([1.5, 1.0, 5.0])

        sse, maximum_deviation = self.selector.calculate_sse(
            training_values,
            ideal_values
        )

        self.assertAlmostEqual(sse, 5.25)
        self.assertAlmostEqual(maximum_deviation, 2.0)


if __name__ == "__main__":
    unittest.main()
