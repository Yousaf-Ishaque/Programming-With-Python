
"""
function_selector.py

Formulates the best-fitting ideal functions by evaluating
the Sum of Squared Error (SSE) against training functions.
"""

import numpy as np

from exceptions import InvalidDatasetError
from base_processor import BaseDatasetProcessor


class FunctionSelector(BaseDatasetProcessor):
    """
    Formulates the best ideal function for every training function
    based on the least Sum of Squared Error (SSE).


    Inherits common dataset validation functionality
    through BaseDatasetProcessor.
    """

    def __init__(self, training_df, ideal_df):
        """
        Starts the FunctionSelector.

        Parameters:

            training_df (pandas.DataFrame): Training dataset.
            ideal_df (pandas.DataFrame): Idea;-function dataset.


        Raises:

            TypeError: If any of the input function is not a DataFrame
            ValueError: If desired columns are not available.
            InvalidDatasetError: If X coordinates do not match.
        """

        self.training_df = training_df
        self.ideal_df = ideal_df

        # Validate the required training dataset columns
        self._validate_required_columns(
            self.training_df,
            ["x", "y1", "y2", "y3", "y4"],
            "Training dataset"
        )

        # Validate the required ideal dataset columns
        ideal_columns = ["x"] + [
            f"y{i}" for i in range(1, 51)
        ]

        self._validate_required_columns(
            self.ideal_df,
            ideal_columns,
            "Ideal dataset"
        )

        # Verify that both datasets have matching X coordinates
        if not np.array_equal(
            self.training_df["x"].to_numpy(),
            self.ideal_df["x"].to_numpy()
        ):
            raise InvalidDatasetError(
                "Training and ideal datasets must have "
                "identical X coordinates in the same order."
            )

    def calculate_sse(self, training_column, ideal_column):
        """
        Formulates SSE and maximum absolute deviation.

        Parameters:

            training_column (pandas.Series): Training function values.
            ideal_column (pandas.Series): Ideal function values.


        Returns:
              tuple:
                    float: Sum of Squared Errors (SSE).
                    float: Maximum absolute deviation.
        """

        training_values = training_column.to_numpy()
        ideal_values = ideal_column.to_numpy()

        # Ensure both functions contain the same number of values
        if len(training_values) != len(ideal_values):
            raise ValueError(
                "Training and ideal functions must have equal lengths."
            )

        difference = training_values - ideal_values

        squared_difference = np.square(difference)

        sse = np.sum(squared_difference)

        maximum_deviation = np.max(np.abs(difference))

        return float(sse), float(maximum_deviation)

    def select_best_functions(self):
        """
        Computes the best ideal function for every training function

        The ideal function with the least SSE is formulated.


        Returns:
            dict: Mapping of every training function to a tuple
            consisting of its selected ideal function, SSE and
            maximum absolute deviation.
        """

        best_functions = {}

        # Iterate through the four training functions
        for train_col in self.training_df.columns[1:]:

            sse_results = []

            # Compare the training function with all 50 ideal functions
            for ideal_col in self.ideal_df.columns[1:]:

                sse, maximum_deviation = self.calculate_sse(
                    self.training_df[train_col],
                    self.ideal_df[ideal_col]
                )

                sse_results.append(
                    (ideal_col, sse, maximum_deviation)
                )

            # Select the ideal function with minimum SSE
            best_match = min(
                sse_results,
                key=lambda result: result[1]
            )

            best_functions[train_col] = best_match

        return best_functions


if __name__ == "__main__":

    from data_loader import DataLoader

    # Load the datasets
    loader = DataLoader()

    train_df = loader.load_training_data()
    ideal_df = loader.load_ideal_data()

    # Create the function selector
    selector = FunctionSelector(train_df, ideal_df)

    # Select the best ideal functions
    best_functions = selector.select_best_functions()

    # Display the selection results
    print("Best Ideal Function Selection Results\n")

    for training_function, result in best_functions.items():

        ideal_function, sse, maximum_deviation = result

        print(f"Training Function: {training_function}")
        print(f"Selected Ideal Function: {ideal_function}")
        print(f"SSE: {sse:.6f}")
        print(f"Maximum Deviation: {maximum_deviation:.6f}")
        print("-" * 45)