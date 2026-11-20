
"""
visualizer.py


This modules generated an interactive Bokeh visualization
for training functions, selected ideal functions, test
observations, and successfully mapped test points.
"""

from bokeh.plotting import figure, show
from bokeh.models import HoverTool, Select, CustomJS
from bokeh.layouts import column

from base_processor import BaseDatasetProcessor


class DataVisualizer(BaseDatasetProcessor):
    """
    Generates interactive visualizations of the datasets
    deployed in the programming assignment.



    Inherits common dataset validation functionality
    from base_processor.
    """

    def __init__(
        self,
        training_df,
        ideal_df,
        test_df,
        mapping_df,
        best_functions
    ):
        """
        Starts the visualizer.

        Parameters

        -------------

        training_df: pandas.DataFrame
            Dataset consisting of four
            training functions.


        Ideal_df: pandas.DataFrame
            Dataset consisting of 50 ideal functions.

        test_df: pandas.DataFrame
            Dataset consisting of test observations.

        mapping_df: pandas.DataFrame
             Dataset consisting of successfully mapped observations.

        best_functions: dict
             Dictionary consisting of four selected ideal functions.
        """

        self.training_df = training_df
        self.ideal_df = ideal_df
        self.test_df = test_df
        self.mapping_df = mapping_df
        self.best_functions = best_functions

        # Generates the figure and dropdown
        self.plot = None
        self.dropdown = None

        # Stores all plot renderers
        self.renderers = {}

        # Verifies datasets prior to accessing their columns
        self._validate_required_columns(
            self.training_df,
            ["x"],
            "Training dataset"
        )

        self._validate_required_columns(
            self.ideal_df,
            ["x"],
            "Ideal dataset"
        )

        self._validate_required_columns(
            self.test_df,
            ["x", "y"],
            "Test dataset"
        )

        self._validate_required_columns(
            self.mapping_df,
            ["x", "y", "delta_y", "ideal_function_number"],
            "Mapped test dataset"
        )

        # Stores the training function names
        self.training_names = list(
            self.training_df.columns.drop("x")
        )

        # Retrieves ideal function names from selection tuples
        self.ideal_names = [
            result[0]
            for result in self.best_functions.values()
        ]

        # Implements the rest of input validation
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Verifies the datasets and selected ideal functions.

        Raises:

            ValueError: If the datasets do not consist of
            desired number of functions or selected ideal
            functions are not available.
        """

        # Verifies that the training datasets comprises four functions
        if len(self.training_names) != 4:
            raise ValueError(
                "Training dataset must contain four functions."
            )

        # Verifies that exactly four ideal functions formulates
        if len(self.best_functions) != 4:
            raise ValueError(
                "Exactly four selected ideal functions are required."
            )

        # Validates the availability of all the selected ideal functions
        for function_name in self.ideal_names:

            if function_name not in self.ideal_df.columns:
                raise ValueError(
                    f"Ideal function {function_name} not found."
                )

    def create_figure(self):
        """
        Generates the main Bokeh figure and dropdown menu.
        """

        self.plot = figure(
            title="Training, Ideal and Test Data Visualization",
            x_axis_label="X Values",
            y_axis_label="Y Values",
            width=1200,
            height=750,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        hover = HoverTool(
            tooltips=[
                ("X Value", "$x"),
                ("Y Value", "$y")
            ]
        )

        self.plot.add_tools(hover)

        # Generates dropdown options
        options = [
            "All Data",
            "Training Functions",
            "Chosen Ideal Functions",
            "Test Data",
            "Mapped Test Points"
        ]

        # Incorporates comparison options dynamically
        for training_name, result in self.best_functions.items():

            ideal_name = result[0]

            options.append(
                f"Compare {training_name} ↔ {ideal_name}"
            )

        self.dropdown = Select(
            title="Visualization",
            value="All Data",
            options=options,
            width=350
        )

        return self.plot

    def plot_training_functions(self):
        """
        Plotting the four training functions.
        """

        colors = [
            "blue",
            "green",
            "orange",
            "purple"
        ]

        for index, function_name in enumerate(
            self.training_names
        ):

            renderer = self.plot.line(
                x=self.training_df["x"],
                y=self.training_df[function_name],
                line_color=colors[index],
                line_width=2,
                legend_label=f"Training Function {function_name}"
            )

            self.renderers[f"training_{function_name}"] = renderer

    def plot_chosen_ideal_functions(self):
        """
        Plotting of the four selected ideal functions.
        """

        colors = [
            "red",
            "brown",
            "magenta",
            "black"
        ]

        for index, function_name in enumerate(
            self.ideal_names
        ):

            renderer = self.plot.line(
                x=self.ideal_df["x"],
                y=self.ideal_df[function_name],
                line_color=colors[index],
                line_dash="dashed",
                line_width=2,
                legend_label=f"Ideal Function {function_name}"
            )

            self.renderers[f"ideal_{function_name}"] = renderer

    def plot_test_data(self):
        """
        Plotting of all the observations through the test dataset.
        """

        self.renderers["test"] = self.plot.scatter(
            x=self.test_df["x"],
            y=self.test_df["y"],
            marker="circle",
            size=6,
            color="darkcyan",
            alpha=0.6,
            legend_label="Test Data"
        )

    def plot_mapped_points(self):
        """
        Plotting of successfully mapped test observations utilizing
        the mapping DataFrame.
        """

        self.renderers["mapped"] = self.plot.scatter(
            x=self.mapping_df["x"],
            y=self.mapping_df["y"],
            marker="diamond",
            size=12,
            color="yellow",
            line_color="gold",
            legend_label="Mapped Test Points"
        )

    def setup_dropdown(self):
        """
        Configuring the dropdown to manage renderer visibility.
        """

        # Preparing the training and ideal renderer names
        training_renderers = [
            f"training_{name}"
            for name in self.training_names
        ]

        ideal_renderers = [
            f"ideal_{name}"
            for name in self.ideal_names
        ]

        callback = CustomJS(
            args={
                "renderers": self.renderers,
                "training_renderers": training_renderers,
                "ideal_renderers": ideal_renderers
            },

            code="""
                const option = cb_obj.value;

                // Hide every renderer initially.
                for (const key in renderers) {
                    renderers[key].visible = false;
                }

                // Display all datasets.
                if (option === "All Data") {

                    for (const key in renderers) {
                        renderers[key].visible = true;
                    }
                }

                // Display training functions.
                else if (option === "Training Functions") {

                    for (const key of training_renderers) {
                        renderers[key].visible = true;
                    }
                }

                // Display selected ideal functions.
                else if (option === "Chosen Ideal Functions") {

                    for (const key of ideal_renderers) {
                        renderers[key].visible = true;
                    }
                }

                // Display all test observations.
                else if (option === "Test Data") {

                    renderers["test"].visible = true;
                }

                // Display successfully mapped observations.
                else if (option === "Mapped Test Points") {

                    renderers["mapped"].visible = true;
                }

                // Compare one training function with its ideal function.
                else if (option.startsWith("Compare ")) {

                    const comparison = option.substring(8);
                    const names = comparison.split(" ↔ ");

                    const training_name = "training_" + names[0];
                    const ideal_name = "ideal_" + names[1];

                    renderers[training_name].visible = true;
                    renderers[ideal_name].visible = true;
                }
            """
        )

        self.dropdown.js_on_change("value", callback)

        return self.dropdown

    def show_visualization(self):
        """
        Exhibits the entire interactive visualization.
        """

        if self.plot is None:
            self.create_figure()
            self.plot_training_functions()
            self.plot_chosen_ideal_functions()
            self.plot_test_data()
            self.plot_mapped_points()
            self.setup_dropdown()

        layout = column(
            self.dropdown,
            self.plot
        )

        show(layout)


# --------------------------------------------------
# Standalone Testing
# --------------------------------------------------

if __name__ == "__main__":

    from data_loader import DataLoader
    from function_selector import FunctionSelector
    from mapper import TestMapper
    import pandas as pd

    print("=" * 55)
    print("VISUALIZER STANDALONE TEST")
    print("=" * 55)

    # 1st Step: Loading of datasets
    loader = DataLoader()

    training_df = loader.load_training_data()
    ideal_df = loader.load_ideal_data()
    test_df = loader.load_test_data()

    print("\nAll datasets loaded successfully.")

    # 2nd Step: Formulating the best-fitting ideal functions
    selector = FunctionSelector(
        training_df,
        ideal_df
    )

    best_functions = selector.select_best_functions()

    print("\nSelected Ideal Functions:")

    for training_name, result in best_functions.items():

        print(
            f"{training_name} -> {result[0]}"
        )

    # 3rd Step: Mapping of test observations
    mapper = TestMapper(
        test_df,
        ideal_df,
        best_functions
    )

    mapping_results = mapper.map_test_data()

    # Conversion of mapping results into a DataFrame
    mapping_df = pd.DataFrame(
        mapping_results,
        columns=[
            "x",
            "y",
            "delta_y",
            "ideal_function_number"
        ]
    )

    print("\nTotal test observations:", len(test_df))
    print("Successfully mapped observations:", len(mapping_df))

    # 4th Step: Starting the visualizer
    visualizer = DataVisualizer(
        training_df,
        ideal_df,
        test_df,
        mapping_df,
        best_functions
    )

    # 5th Step: Generation of Visualization
    visualizer.create_figure()

    visualizer.plot_training_functions()
    visualizer.plot_chosen_ideal_functions()
    visualizer.plot_test_data()
    visualizer.plot_mapped_points()

    visualizer.setup_dropdown()

    # 6th Step: Exhibits the visualization
    visualizer.show_visualization()

    