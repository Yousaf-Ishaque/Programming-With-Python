
"""
data_loader.py

Loads and verifies the training, ideal-function and test datasets
utilizing test datasets utilizing Pandas.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from exceptions import InvalidDatasetError


class DataLoader:
    """
    Loads and verifies the three CSV dataset
    as specified by the assignment
    """

    def __init__(self):


        """Establishes the path of three CSV files."""

        project_dir = Path(__file__).resolve().parent
        dataset_dir = project_dir / "datasets"

        self.train_path = dataset_dir / "train.csv"
        self.ideal_path = dataset_dir / "ideal.csv"
        self.test_path = dataset_dir / "test.csv"

    def _load_and_validate(
        self, file_path, expected_columns, allow_duplicate_x=False
    ):
        """
        Loads a CSV file and verifies its structure and contents.




        Args:

            file_path: Path to the CSV file.
            expected_columns: Expected column names.
            allow_duplicate_x: Whether duplicate X coordinates are permitted.


        Returns:
            pandas.DataFrame: Verified dataset.


        Raises:

            InvalidDatasetError: If the dataset validation is not successful.
            FileNotFoundError: If the CSV file is not available.
            PermissionError: If the CSV file is not accessible.
        """

        try:
            df = pd.read_csv(file_path)

        except pd.errors.EmptyDataError as error:
            raise InvalidDatasetError(
                f"The dataset is empty: {file_path}"
            ) from error

        except pd.errors.ParserError as error:
            raise InvalidDatasetError(
                f"Unable to parse the CSV file: {file_path}"
            ) from error

        # Check whether the dataset contains data rows
        if df.empty:
            raise InvalidDatasetError(
                f"No data rows found in {file_path}"
            )

        # Validate column names and their order
        if list(df.columns) != expected_columns:
            raise InvalidDatasetError(
                f"Incorrect columns in {file_path}. "
                f"Expected: {expected_columns}"
            )

        # Check for missing values
        if df.isnull().values.any():
            raise InvalidDatasetError(
                f"Missing values found in {file_path}"
            )

        # Validate numeric values
        try:
            df = df.apply(pd.to_numeric, errors="raise")

        except (ValueError, TypeError) as error:
            raise InvalidDatasetError(
                f"Non-numeric values found in {file_path}"
            ) from error

        # Check for infinite and other non-finite values
        if not np.isfinite(df.to_numpy()).all():
            raise InvalidDatasetError(
                f"Infinite values found in {file_path}"
            )

        # Look for duplicate X-coordinates only when not permitted
        if not allow_duplicate_x and df["x"].duplicated().any():
            raise InvalidDatasetError(
                f"Duplicate X coordinates found in {file_path}"
            )

        return df

    def load_training_data(self):
        """Loads and verifies the training dataset."""

        expected_columns = ["x", "y1", "y2", "y3", "y4"]

        return self._load_and_validate(
            self.train_path,
            expected_columns
        )

    def load_ideal_data(self):
        """ Loaad and verifies the ideal-function dataset."""

        expected_columns = ["x"] + [
            f"y{i}" for i in range(1, 51)
        ]

        return self._load_and_validate(
            self.ideal_path,
            expected_columns
        )

    def load_test_data(self):
        """
        Loads and verifies the test dataset.

        Duplicate X-coordinates are permitted because distinct
        test observations may share the identical X value
        """

        expected_columns = ["x", "y"]

        return self._load_and_validate(
            self.test_path,
            expected_columns,
            allow_duplicate_x=True
        )


if __name__ == "__main__":

    # Generates DataLoader object
    loader = DataLoader()

    # Loads the three given datasets
    train_df = loader.load_training_data()
    ideal_df = loader.load_ideal_data()
    test_df = loader.load_test_data()

    # Exhibit dataset dimensions/shape
    print("Training Dataset:", train_df.shape)
    print("Ideal Dataset:", ideal_df.shape)
    print("Test Dataset:", test_df.shape)

    print("\nAll datasets loaded and validated successfully!")
