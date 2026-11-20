
"""
main.py

The modules controls the entire execution workflow
of the Programming With Python assignment.

It successfully loads and verifies datasets, stores data in
SQLite, formulates the four most appropriate ideal functions,
maps test observations, and creates data visualizations.
"""

from database import DatabaseManager
from data_loader import DataLoader
from function_selector import FunctionSelector
from mapper import TestMapper
import pandas as pd
from visualizer import DataVisualizer
from pandas.errors import EmptyDataError, ParserError
from exceptions import InvalidDatasetError


def main():
    """
    Implements the complete assignment workflow.

    The function generates the database, loads and validates
    datasets, formulates ideal functions, maps test observations,
    store the results and creates data visualizations.

    The database engine is closed after execution
    and it is even closed when an exception or early
    return occurs.
    """

    print("=" * 60)
    print("PROGRAM STARTED")
    print("=" * 60)

    # Step # 01: Generation of Database
    db = DatabaseManager()

    try:
        db.create_database()

        # Step # 02: Loading of Dataset
        loader = DataLoader()

        try:
            training_df = loader.load_training_data()
            ideal_df = loader.load_ideal_data()
            test_df = loader.load_test_data()

        except InvalidDatasetError as error:
            print(f"\nDataset validation error: {error}")
            return

        except (
            FileNotFoundError,
            EmptyDataError,
            ParserError,
            PermissionError
        ) as error:
            print(f"\nDataset loading error: {error}")
            return

        print("\n✓ All datasets loaded successfully.")

        # Step # 03: Demonstrating Dataset Shapes/Dimensions
        print("\nDataset Shapes")
        print("-" * 40)
        print(f"Training Dataset : {training_df.shape}")
        print(f"Ideal Dataset    : {ideal_df.shape}")
        print(f"Test Dataset     : {test_df.shape}")

        # Step # 04: Loading of Datasets into SQLite
        db.save_dataframe(training_df, "training_data")
        db.save_dataframe(ideal_df, "ideal_data")
        db.save_dataframe(test_df, "test_data")

        print("\n✓ All datasets saved into SQLite.")

        # Step # 05: Formulating the Most Appropriate Ideal Functions
        selector = FunctionSelector(training_df, ideal_df)
        best_functions = selector.select_best_functions()

        print("\nSelected Ideal Functions")
        print("=" * 60)

        for training_function, best_match in best_functions.items():
            print(f"\nTraining Function : {training_function}")
            print(f"Ideal Function    : {best_match[0]}")
            print(f"SSE               : {best_match[1]:.6f}")
            print(f"Maximum Deviation : {best_match[2]:.6f}")

        # Step # 06: Mapping of Test Data
        mapper = TestMapper(test_df, ideal_df, best_functions)
        mapping_results = mapper.map_test_data()

        mapping_df = pd.DataFrame(
            mapping_results,
            columns=[
                "x",
                "y",
                "delta_y",
                "ideal_function_number"
            ]
        )

        print("\nMapped Test Data")
        print("=" * 60)
        print(f"Total mapped test points : {len(mapping_results)}")

        print("\nFirst 10 Mapped Test Points\n")

        for result in mapping_results[:10]:
            print(result)

        db.save_dataframe(mapping_df, "mapped_test_data")

        print("\n" + "=" * 60)
        print("DATA PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)

        # Step # 07: Creating Data Visualizations
        visualizer = DataVisualizer(
            training_df,
            ideal_df,
            test_df,
            mapping_df,
            best_functions
        )

        visualizer.create_figure()
        visualizer.plot_training_functions()
        visualizer.plot_chosen_ideal_functions()

        """ Plotting of test observations prior to mapped
        points so that mapped diamonds are clearly visible 
        on top"""

        visualizer.plot_test_data()
        visualizer.plot_mapped_points()

        visualizer.setup_dropdown()
        visualizer.show_visualization()

    finally:

        # Always dispose of the database engine
        db.close()


if __name__ == "__main__":
    main()
