
"""
test_data_loader.py

Unit tests designed for validating loading of
datasets, structure, columns, and verification
utilizing DataLoader.
"""

import unittest
import tempfile
import os

from data_loader import DataLoader, InvalidDatasetError


class TestDataLoader(unittest.TestCase):
    """Tests the functionality of the DataLoader class."""

    def setUp(self):


        """Initializes the DataLoader prior to each test."""
        self.loader = DataLoader()

    def test_training_data_loaded(self):
        """Validates that the training dataset loads successfully."""
        training_df = self.loader.load_training_data()
        self.assertIsNotNone(training_df)

    def test_training_data_shape(self):
        """Validates the dimensions/shape of the training dataset."""
        training_df = self.loader.load_training_data()
        self.assertEqual(training_df.shape, (400, 5))

    def test_training_columns(self):
        """Validates the anticipated training dataset columns."""
        training_df = self.loader.load_training_data()
        expected_columns = ["x", "y1", "y2", "y3", "y4"]
        self.assertEqual(list(training_df.columns), expected_columns)

    def test_training_not_empty(self):
        """Validates that the training dataset is not empty."""
        training_df = self.loader.load_training_data()
        self.assertFalse(training_df.empty)

    def test_ideal_data_loaded(self):
        """Validates the successful loading of ideal datasets."""
        ideal_df = self.loader.load_ideal_data()
        self.assertIsNotNone(ideal_df)

    def test_ideal_data_shape(self):
        """Validates the dimensions/shape of the ideal dataset."""
        ideal_df = self.loader.load_ideal_data()
        self.assertEqual(ideal_df.shape, (400, 51))

    def test_ideal_dataframe_not_empty(self):
        """Validates that the ideal dataset is not empty."""
        ideal_df = self.loader.load_ideal_data()
        self.assertFalse(ideal_df.empty)

    def test_ideal_columns_exist(self):
        """ Validates the first and last ideal function columns."""
        ideal_df = self.loader.load_ideal_data()
        self.assertIn("x", ideal_df.columns)
        self.assertIn("y1", ideal_df.columns)
        self.assertIn("y50", ideal_df.columns)

    def test_infinite_values_rejected(self):
        """Enables the rejection of infinite values in the datasets."""

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8"
        ) as temp_file:

            temp_file.write("x,y1,y2,y3,y4\n")
            temp_file.write("1,10,20,30,40\n")
            temp_file.write("2,inf,25,35,45\n")

            file_path = temp_file.name

        try:
            with self.assertRaises(InvalidDatasetError):
                self.loader._load_and_validate(
                    file_path,
                    ["x", "y1", "y2", "y3", "y4"]
                )

        finally:
            os.remove(file_path)


if __name__ == "__main__":
    unittest.main()