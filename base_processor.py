
"""
base_processor.py


Module describes a base class containing shared dataset
validation functionality throughout the program
"""

import pandas as pd


class BaseDatasetProcessor:
    """
    Parent class enabling shared dataset-processing
    and validation functionality.
    """

    def _validate_required_columns(
        self, dataframe, required_columns, dataset_name
    ):
        """
        Validates whether a DataFrame contains all required columns.

        Args:
            dataframe (pandas.DataFrame): DataFrame to be validated.
            required_columns (list): Required column names.
            dataset_name (str): Name of the dataset being validated.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
            ValueError: If required columns are missing.
        """

        # Validates that the input is a pandas DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"{dataset_name} must be a Pandas DataFrame."
            )

        # Identify missing columns
        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        # Raise an exception if required columns are missing
        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing required columns: "
                f"{missing_columns}"
            )
