
"""
mapper.py

Mapping of test observations to the four selected ideal functions
based on maximum deviation criterion multiplied by sqrt(2).
"""

import numpy as np

from exceptions import InvalidDatasetError
from base_processor import BaseDatasetProcessor


class TestMapper(BaseDatasetProcessor):
    """
    Maps test observations with appropriate ideal
    functions on the criterion of maximum specified
    deviation.

    Inherits common dataset validation functionality
    through BaseProcessor.
    """

    def __init__(self, test_df, ideal_df, best_functions):
        """
        Starts the TestMapper function.

        Parameters:
            test_df (pandas.DataFrame): Test dataset.
            ideal_df (pandas.DataFrame): Ideal-function dataset.
            best_functions (dict): Selected ideal functions,
            incorporating their maximum training deviations.

        Raises:
            TypeError: If any of the datasets is not a DataFrame.
            ValueError: If the desired columns are not available
            or selected function information is invalid.
            InvalidDatasetError: If the ideal dataset consists of
            duplicate X coordinates or desired functions are not
            available.
        """

        self.test_df = test_df
        self.ideal_df = ideal_df
        self.best_functions = best_functions

        # Verifies the selected ideal functions
        if not isinstance(self.best_functions, dict) or not self.best_functions:
            raise ValueError(
                "Best functions must be provided as a non-empty dictionary."
            )

        for training_function, result in self.best_functions.items():

            if not isinstance(result, (tuple, list)) or len(result) != 3:
                raise ValueError(
                    f"Invalid selection result for {training_function}. "
                    "Expected (ideal_function, SSE, maximum_deviation)."
                )

            ideal_function, sse, maximum_deviation = result

            if not isinstance(ideal_function, str):
                raise ValueError(
                    f"Invalid ideal function name for {training_function}."
                )

            if not isinstance(sse, (int, float, np.number)):
                raise ValueError(
                    f"Invalid SSE value for {training_function}."
                )

            if not isinstance(maximum_deviation, (int, float, np.number)):
                raise ValueError(
                    f"Invalid maximum deviation for {training_function}."
                )

            if (
                not np.isfinite(sse)
                or not np.isfinite(maximum_deviation)
                or sse < 0
                or maximum_deviation < 0
            ):
                raise ValueError(
                    f"Invalid numerical selection values for {training_function}."
                )

        # Verifies the test dataset
        self._validate_required_columns(
            self.test_df,
            ["x", "y"],
            "Test dataset"
        )

        # Retrieves the names of selected ideal functions
        selected_ideal_columns = [
            result[0] for result in self.best_functions.values()
        ]

        # Verifies the ideal dataset
        self._validate_required_columns(
            self.ideal_df,
            ["x"] + selected_ideal_columns,
            "Ideal dataset"
        )

        # Validates the ideal dataset has unique X coordinates
        if self.ideal_df["x"].duplicated().any():
            raise InvalidDatasetError(
                "Ideal dataset contains duplicate X coordinates."
            )

        # Generates an indexed copy for efficient X-coordinate lookup
        self.ideal_indexed = self.ideal_df.set_index("x")

        # Validates that all the selected ideal functions exist
        for ideal_function, _, _ in self.best_functions.values():

            if ideal_function not in self.ideal_indexed.columns:
                raise InvalidDatasetError(
                    f"Selected ideal function {ideal_function} "
                    "does not exist in the ideal dataset."
                )

    def map_test_data(self):
        """
        Maps every test observation to the nearest
        acceptable selected ideal functions.

        The maximum permitted deviation is computed as:

            Threshold = Maximum Training Deviation * sqrt(2)

        Returns:
            list: Mapping results containing:
                x: Test X coordinate.
                y: Test Y value.
                delta_y: Absolute deviation.
                ideal_function_number: Selected ideal function name.
        """

        mapping_results = []

        # Processing of every test observation
        for _, test_row in self.test_df.iterrows():

            x = test_row["x"]
            y = test_row["y"]

            # Validates whether the X-coordinates exist in the ideal dataset
            if x not in self.ideal_indexed.index:
                raise InvalidDatasetError(
                    f"Test X coordinate {x} does not exist "
                    "in the ideal dataset."
                )

            # Generates the best acceptable match
            best_deviation = float("inf")
            best_function = None

            # Comparative analysis against all selected ideal functions
            for _, best_match in self.best_functions.items():

                ideal_function, _, maximum_deviation = best_match

                # Extracts the respective ideal Y value
                ideal_y = self.ideal_indexed.at[x, ideal_function]

                # Computes absolute deviation
                deviation = abs(y - ideal_y)

                # Computes the maximum acceptable deviation
                threshold = np.sqrt(2) * maximum_deviation

                # Formulates the closest acceptable ideal function
                if deviation <= threshold and deviation < best_deviation:

                    best_deviation = deviation
                    best_function = ideal_function

            # Stores the mapping utilizing four desired fields
            if best_function is not None:

                mapping_results.append({
                    "x": float(x),
                    "y": float(y),
                    "delta_y": float(best_deviation),
                    "ideal_function_number": best_function
                })

        return mapping_results

    def generate_mapping_summary(self, mapping_results):
        """
        Generates a summary of test-data mapping results.

        Parameters:
            mapping_results (list): Successfully mapped observations.

        Returns:
            dict: Total, mapped, unmapped observations
            and mapping success percentage.
        """

        total = len(self.test_df)
        mapped = len(mapping_results)
        unmapped = total - mapped

        success_rate = (mapped / total * 100) if total > 0 else 0

        return {
            "total_observations": total,
            "mapped_observations": mapped,
            "unmapped_observations": unmapped,
            "mapping_success_rate": round(success_rate, 2)
        }


if __name__ == "__main__":

    from data_loader import DataLoader
    from function_selector import FunctionSelector

    # Loading of datasets
    loader = DataLoader()

    train_df = loader.load_training_data()
    ideal_df = loader.load_ideal_data()
    test_df = loader.load_test_data()

    # Selection of best ideal functions
    selector = FunctionSelector(train_df, ideal_df)
    best_functions = selector.select_best_functions()

    # Mapping of test observations
    mapper = TestMapper(test_df, ideal_df, best_functions)
    mapping_results = mapper.map_test_data()

    # Generates mapping summary
    mapping_summary = mapper.generate_mapping_summary(mapping_results)

    # Displays mapping summary
    print("Test Data Mapping Summary\n")

    print("Total test observations:",
          mapping_summary["total_observations"])
    print("Successfully mapped observations:",
          mapping_summary["mapped_observations"])
    print("Unmapped observations:",
          mapping_summary["unmapped_observations"])
    print("Mapping success rate:",
          str(mapping_summary["mapping_success_rate"]) + "%")

    # Exhibits first five mapping results
    print("\nFirst five mapping results:")

    for result in mapping_results[:5]:
        print(result)
