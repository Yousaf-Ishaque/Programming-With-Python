
"""
test_mapper.py

Unit tests for validating test_data mapping
output structure, and numerical values.
"""

import unittest

from data_loader import DataLoader
from function_selector import FunctionSelector
from mapper import TestMapper


class TestMapperModule(unittest.TestCase):
    """Testing the functionality of the test_mapper class."""

    def setUp(self):
        """Loading of datasets and initializing the mapper."""

        loader = DataLoader()

        self.training_df = loader.load_training_data()
        self.ideal_df = loader.load_ideal_data()
        self.test_df = loader.load_test_data()

        selector = FunctionSelector(
            self.training_df,
            self.ideal_df
        )

        self.best_functions = selector.select_best_functions()

        self.mapper = TestMapper(
            self.test_df,
            self.ideal_df,
            self.best_functions
        )

    def test_mapper_created(self):
        """Validates that the mapper is started."""
        self.assertIsNotNone(self.mapper)

    def test_mapping_returns_list(self):
        """Validates that mapped results are returned as a list."""
        mapped_points = self.mapper.map_test_data()
        self.assertIsInstance(mapped_points, list)

    def test_mapping_not_empty(self):
        """Validates that mapping generated at least one result."""
        mapped_points = self.mapper.map_test_data()
        self.assertGreater(len(mapped_points), 0)

    def test_number_of_mapped_points(self):
        """Validation of the expected number of mapped observations."""
        mapped_points = self.mapper.map_test_data()
        self.assertEqual(len(mapped_points), 34)

    def test_each_mapping_is_dictionary(self):
        """Validation that every mapping result is a dictionary."""
        mapped_points = self.mapper.map_test_data()

        for point in mapped_points:
            self.assertIsInstance(point, dict)

    def test_mapping_contains_required_keys(self):
        """Validation of the four desired mapping output fields."""
        mapped_points = self.mapper.map_test_data()

        required_keys = [
            "x",
            "y",
            "delta_y",
            "ideal_function_number"
        ]

        for point in mapped_points:
            self.assertEqual(set(point.keys()), set(required_keys))

    def test_ideal_function_number_is_string(self):
        """Validates that the selected ideal function number is string."""
        mapped_points = self.mapper.map_test_data()

        for point in mapped_points:
            self.assertIsInstance(
                point["ideal_function_number"], str
            )

    def test_delta_y_is_numeric(self):
        """Validation that the computed deviation is numeric."""
        mapped_points = self.mapper.map_test_data()

        for point in mapped_points:
            self.assertIsInstance(point["delta_y"], float)

    def test_x_and_y_are_numeric(self):
        """Validates that X and Y coordinates are numeric."""
        mapped_points = self.mapper.map_test_data()

        for point in mapped_points:
            self.assertIsInstance(point["x"], float)
            self.assertIsInstance(point["y"], float)

    def test_empty_best_functions_rejected(self):
        """Validates that an empty best-function's dictionary is rejected."""

        with self.assertRaises(ValueError):
            TestMapper(
                self.test_df,
                self.ideal_df,
                {}
            )


if __name__ == "__main__":
    unittest.main()
