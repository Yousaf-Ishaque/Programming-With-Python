

"""
visualizer.py

This module creates an interactive Bokeh visualization
for training functions, selected ideal functions,
test observations, and successfully mapped test points.
"""

from bokeh.plotting import figure, show
from bokeh.models import (
    HoverTool, Select, CustomJS, Range1d, FixedTicker,
    Span, BoxAnnotation, Label, LegendItem
)

from bokeh.layouts import column

from base_processor import BaseDatasetProcessor


class DataVisualizer(BaseDatasetProcessor):
    """
    Creates interactive visualizations of the datasets
    used in the programming assignment.

    Inherits common dataset validation functionality
    from BaseDatasetProcessor.
    """

    def __init__(
        self,
        training_df,
        ideal_df,
        test_df,
        mapping_df,
        best_functions,
        mapped_points
    ):
        """
        Initialize the visualizer.

        Parameters
        ----------
        training_df : pandas.DataFrame
            Dataset containing four training functions.

        ideal_df : pandas.DataFrame
            Dataset containing 50 ideal functions.

        test_df : pandas.DataFrame
            Dataset containing test observations.

        mapping_df : pandas.DataFrame
            DataFrame containing successfully mapped observations.

        best_functions : dict
            Dictionary containing the four selected ideal functions.

        mapped_points : list
            List of successfully mapped test observations.
        """

        self.training_df = training_df
        self.ideal_df = ideal_df
        self.test_df = test_df
        self.mapping_df = mapping_df
        self.best_functions = best_functions
        self.mapped_points = mapped_points

        # Initialize figure and dropdown
        self.plot = None
        self.dropdown = None

        # Store all plot renderers
        self.renderers = {}

        # Validate datasets before accessing their columns
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
            "Mapping dataset"
        )

        # Store the training function names
        self.training_names = list(
            self.training_df.columns.drop("x")
        )

        # Extract ideal function names from selection tuples
        self.ideal_names = [
            result[0]
            for result in self.best_functions.values()
        ]

        # Perform remaining input validation
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validate the datasets and selected ideal functions.

        Raises:
            ValueError: If the datasets do not contain
                the required number of functions or
                selected ideal functions are missing.
        """

        # Check that the training dataset contains four functions
        if len(self.training_names) != 4:
            raise ValueError(
                "Training dataset must contain four functions."
            )

        # Check that exactly four ideal functions are selected
        if len(self.best_functions) != 4:
            raise ValueError(
                "Exactly four selected ideal functions are required."
            )

        # Verify that all selected ideal functions exist
        for function_name in self.ideal_names:

            if function_name not in self.ideal_df.columns:
                raise ValueError(
                    f"Ideal function {function_name} not found."
                )

    @staticmethod
    def _axis_limits(x_values, y_values):
        """
        Calculate padded axis limits from the data being displayed.
        Each comparison receives its own Y-axis range.
        """
        x_values = [float(value) for value in x_values]
        y_values = [float(value) for value in y_values]

        if not x_values or not y_values:
            return (-21, 21, -1, 1)

        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        x_span = x_max - x_min
        y_span = y_max - y_min

        x_pad = 0.05 * x_span if x_span else 1
        y_pad = 0.05 * y_span if y_span else max(abs(y_min) * 0.05, 1)

        return (
            x_min - x_pad,
            x_max + x_pad,
            y_min - y_pad,
            y_max + y_pad
        )

    def _create_range_presets(self):
        """
        Pre-calculate ranges for each dropdown choice using only
        the data that should be visible for that choice.
        """
        presets = {}

        def store(option, x_series, y_series):
            x_values = []
            y_values = []

            for series in x_series:
                x_values.extend(series.tolist())

            for series in y_series:
                y_values.extend(series.tolist())

            presets[option] = self._axis_limits(x_values, y_values)

        # Overall view: all functions and both test-point layers.
        store(
            "All Data",
            [
                self.training_df["x"], self.ideal_df["x"],
                self.test_df["x"], self.mapping_df["x"]
            ],
            [self.training_df[name] for name in self.training_names]
            + [self.ideal_df[name] for name in self.ideal_names]
            + [self.test_df["y"], self.mapping_df["y"]]
        )

        store(
            "Training Functions",
            [self.training_df["x"]],
            [self.training_df[name] for name in self.training_names]
        )

        store(
            "Chosen Ideal Functions",
            [self.ideal_df["x"]],
            [self.ideal_df[name] for name in self.ideal_names]
        )

        store("Test Data", [self.test_df["x"]], [self.test_df["y"]])
        store(
            "Mapped Test Points",
            [self.mapping_df["x"]],
            [self.mapping_df["y"]]
        )

        for training_name, result in self.best_functions.items():
            ideal_name = result[0]
            option = f"Compare {training_name} ↔ {ideal_name}"

            store(
                option,
                [self.training_df["x"], self.ideal_df["x"]],
                [
                    self.training_df[training_name],
                    self.ideal_df[ideal_name]
                ]
            )

        return presets

    def create_figure(self):
        """
        Create the main Bokeh figure and dropdown menu.
        """

        # Calculate independent axis ranges for each dropdown selection.
        self.range_presets = self._create_range_presets()
        x_start, x_end, y_start, y_end = self.range_presets["All Data"]

        self.plot = figure(
            title="Training, Ideal and Test Data Visualization",
            x_axis_label="X Values",
            y_axis_label="Absolute Deviation |Δy|",
            x_range=Range1d(start=-21, end=21),
            y_range=Range1d(start=0.0, end=1.4),
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

        # Create dropdown options
        options = [
            "All Data",
            "Training Functions",
            "Chosen Ideal Functions",
            "Test Data",
            "Mapped Test Points",
            "Test-Data Deviation Analysis"
        ]

        # Add comparison options dynamically
        for training_name, result in self.best_functions.items():

            ideal_name = result[0]

            options.append(
                f"Compare {training_name} ↔ {ideal_name}"
            )

        self.dropdown = Select(
            title="Visualization",
            value="Compare y1 ↔ y13",
            options=options,
            width=350
        )

        return self.plot

    def plot_training_functions(self):
        """
        Plot the four training functions.
        """

        colors = [
            "#1f77b4",
            "green",
            "orange",
            "#1f77b4"
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
        Plot the four selected ideal functions.
        """

        colors = [
            "#ff0000",
            "brown",
            "#7b2aba",
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
        Plot all observations from the test dataset.
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
        Plot successfully mapped test observations.
        """

        x_values = []
        y_values = []

        for point in self.mapped_points:

            x_values.append(point["x"])
            y_values.append(point["y"])

        self.renderers["mapped"] = self.plot.scatter(
            x=x_values,
            y=y_values,
            marker="diamond",
            size=12,
            color="yellow",
            line_color="gold",
            legend_label="Mapped Test Points"
        )

    def plot_deviation_analysis(self):
        """Plot the Figure 6-style test-data deviation analysis."""

        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        ideal_index = self.ideal_df.set_index("x")

        # Common displayed threshold: largest selected training deviation * sqrt(2).
        self.deviation_threshold = max(
            float(result[2]) for result in self.best_functions.values()
        ) * (2 ** 0.5)
        # Display the threshold truncated to three decimals (0.706),
        # while retaining the full-precision value for mapping and plotting.
        self.deviation_threshold_display = int(
            self.deviation_threshold * 1000
        ) / 1000

        grouped = {
            result[0]: {"x": [], "deviation": []}
            for result in self.best_functions.values()
        }

        mapped_x = set()
        for _, row in self.mapping_df.iterrows():
            x_value = float(row["x"])
            mapped_x.add(x_value)
            function_name = row["ideal_function_number"]
            grouped[function_name]["x"].append(x_value)
            grouped[function_name]["deviation"].append(float(row["delta_y"]))

        for index, function_name in enumerate(grouped):
            points = grouped[function_name]
            renderer = self.plot.scatter(
                x=points["x"], y=points["deviation"],
                marker="diamond", size=9, color=colors[index],
                legend_label=f"Mapped to {function_name}"
            )
            renderer.visible = False
            self.renderers[f"deviation_mapped_{function_name}"] = renderer

        # The reference explicitly labels rejected points as an excerpt.
        # Show only rejected observations that fit within the 0–1.4 display.
        excerpt_x, excerpt_y = [], []
        for _, test_row in self.test_df.iterrows():
            x_value = float(test_row["x"])
            if x_value in mapped_x:
                continue

            y_value = float(test_row["y"])
            deviations = [
                abs(y_value - float(ideal_index.at[test_row["x"], result[0]]))
                for result in self.best_functions.values()
            ]
            closest_deviation = min(deviations)

            if self.deviation_threshold < closest_deviation <= 1.4:
                excerpt_x.append(x_value)
                excerpt_y.append(closest_deviation)

        unmapped = self.plot.scatter(
            x=excerpt_x, y=excerpt_y, marker="x", size=8,
            color="#9aaec4", alpha=0.9,
            legend_label="Unmapped Test Points (excerpt)"
        )
        unmapped.visible = False
        self.renderers["deviation_unmapped"] = unmapped

        threshold_line = self.plot.line(
            x=[float(self.test_df["x"].min()), float(self.test_df["x"].max())],
            y=[self.deviation_threshold, self.deviation_threshold],
            line_color="#e34a33", line_dash="dashed", line_width=2,
            legend_label=f"Acceptance Threshold (δ_max√2 = {self.deviation_threshold_display:.3f})"
        )
        threshold_line.visible = False
        self.renderers["deviation_threshold"] = threshold_line

        self.deviation_acceptance_region = BoxAnnotation(
            bottom=0, top=self.deviation_threshold,
            fill_color="green", fill_alpha=0.06, visible=False
        )
        self.deviation_rejection_region = BoxAnnotation(
            bottom=self.deviation_threshold, top=1.4,
            fill_color="red", fill_alpha=0.04, visible=False
        )
        self.plot.add_layout(self.deviation_acceptance_region)
        self.plot.add_layout(self.deviation_rejection_region)

        mapped_count = len(self.mapping_df)
        unmapped_count = len(self.test_df) - mapped_count

        self.deviation_acceptance_label = Label(
            x=-19.2, y=0.045, x_units="data", y_units="data",
            text=f"Acceptance Region (|Δy| ≤ {self.deviation_threshold_display:.3f} — "
                 f"{mapped_count} Mapped Observations)",
            text_color="seagreen", text_font_size="8pt",
            text_font_style="bold", visible=False
        )
        self.deviation_rejection_label = Label(
            x=-19.2, y=0.735, x_units="data", y_units="data",
            text=f"Rejection Region (|Δy| > {self.deviation_threshold_display:.3f} — "
                 f"{unmapped_count} Unmapped Observations)",
            text_color="firebrick", text_font_size="8pt",
            text_font_style="bold", visible=False
        )
        self.plot.add_layout(self.deviation_acceptance_label)
        self.plot.add_layout(self.deviation_rejection_label)

        self.deviation_renderer_keys = [
            key for key in self.renderers if key.startswith("deviation_")
        ]

    def setup_dropdown(self):
        """Configure view renderers, axes, and view-specific legends."""

        training_renderers = [f"training_{name}" for name in self.training_names]
        ideal_renderers = [f"ideal_{name}" for name in self.ideal_names]

        annotations = [
            self.deviation_acceptance_region,
            self.deviation_rejection_region,
            self.deviation_acceptance_label,
            self.deviation_rejection_label
        ]

        legend = self.plot.legend[0]
        default_legend_items = list(legend.items)

        # Two-entry comparison legends, preserving the established colors and positions.
        comparison_legend_items = {}
        for training_name, result in self.best_functions.items():
            ideal_name = result[0]
            option = f"Compare {training_name} ↔ {ideal_name}"
            comparison_legend_items[option] = [
                LegendItem(
                    label="Training Data",
                    renderers=[self.renderers[f"training_{training_name}"]]
                ),
                LegendItem(
                    label="Selected Ideal Function",
                    renderers=[self.renderers[f"ideal_{ideal_name}"]]
                )
            ]

        # Test Data view: show only the test observations in its legend.
        test_data_only_legend_items = [
            LegendItem(label="Test Data", renderers=[self.renderers["test"]])
        ]

        # Mapped Test Points view: show only the mapped-point renderer.
        mapped_points_only_legend_items = [
            LegendItem(
                label="Mapped Test Points",
                renderers=[self.renderers["mapped"]]
            )
        ]

        # Chosen Ideal Functions view: show only the four selected ideal curves.
        ideal_only_legend_items = [
            LegendItem(
                label=f"Ideal Function {name}",
                renderers=[self.renderers[f"ideal_{name}"]]
            )
            for name in self.ideal_names
        ]

        # Training Functions view: show only the four training curves that are drawn.
        training_only_legend_items = [
            LegendItem(
                label=f"Training Function {name}",
                renderers=[self.renderers[f"training_{name}"]]
            )
            for name in self.training_names
        ]

        # All Data legend: retain training, selected-ideal, test, and mapped-data
        # entries, but exclude the six deviation-analysis-specific entries.
        all_data_legend_items = []
        for name in self.training_names:
            all_data_legend_items.append(
                LegendItem(
                    label=f"Training Function {name}",
                    renderers=[self.renderers[f"training_{name}"]]
                )
            )
        for name in self.ideal_names:
            all_data_legend_items.append(
                LegendItem(
                    label=f"Ideal Function {name}",
                    renderers=[self.renderers[f"ideal_{name}"]]
                )
            )
        all_data_legend_items.extend([
            LegendItem(label="Test Data", renderers=[self.renderers["test"]]),
            LegendItem(label="Mapped Test Points", renderers=[self.renderers["mapped"]])
        ])

        # Six-entry deviation legend, arranged as three columns with two items each.
        deviation_legend_items = [
            LegendItem(
                label="Mapped to y13",
                renderers=[self.renderers["deviation_mapped_y13"]]
            ),
            LegendItem(
                label="Mapped to y24",
                renderers=[self.renderers["deviation_mapped_y24"]]
            ),
            LegendItem(
                label="Mapped to y36",
                renderers=[self.renderers["deviation_mapped_y36"]]
            ),
            LegendItem(
                label="Mapped to y40",
                renderers=[self.renderers["deviation_mapped_y40"]]
            ),
            LegendItem(
                label="Unmapped Test Points (|δy| > δ_max√2, excerpt)",
                renderers=[self.renderers["deviation_unmapped"]]
            ),
            LegendItem(
                label=(
                    "Acceptance Threshold "
                    f"(δ_max√2 = {self.deviation_threshold_display:.3f})"
                ),
                renderers=[self.renderers["deviation_threshold"]]
            )
        ]

        callback = CustomJS(
            args={
                "renderers": self.renderers,
                "training_renderers": training_renderers,
                "ideal_renderers": ideal_renderers,
                "plot": self.plot,
                "legend": legend,
                "default_legend_items": default_legend_items,
                "all_data_legend_items": all_data_legend_items,
                "training_only_legend_items": training_only_legend_items,
                "ideal_only_legend_items": ideal_only_legend_items,
                "test_data_only_legend_items": test_data_only_legend_items,
                "mapped_points_only_legend_items": mapped_points_only_legend_items,
                "comparison_legend_items": comparison_legend_items,
                "deviation_legend_items": deviation_legend_items,
                "x_range": self.plot.x_range,
                "y_range": self.plot.y_range,
                "x_axis": self.plot.xaxis[0],
                "y_axis": self.plot.yaxis[0],
                "range_presets": self.range_presets,
                "deviation_renderer_keys": self.deviation_renderer_keys,
                "deviation_annotations": annotations,
                "deviation_ticker": FixedTicker(
                    ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
                ),
                "normal_ticker": self.plot.yaxis[0].ticker,
                "y1_y13_x_ticker": FixedTicker(
                    ticks=[-20, -15, -10, -5, 0, 5, 10, 15, 20]
                ),
                "y1_y13_y_ticker": FixedTicker(
                    ticks=[-40, -20, 0, 20]
                ),
                "y2_y24_y_ticker": FixedTicker(
                    ticks=[-15000, -10000, -5000, 0, 5000, 10000, 15000]
                )
            },
            code="""
                const option = cb_obj.value;

                // Put the selected y36 ideal curve above the orange y3 training
                // curve only in this comparison. Restore its normal drawing layer
                // in every other view so the remaining charts are unaffected.
                if (renderers["ideal_y36"]) {
                    const isY3Comparison = option === "Compare y3 ↔ y36";
                    renderers["ideal_y36"].level = isY3Comparison ? "overlay" : "glyph";
                    // Slightly thicken only the purple y36 dashed curve in its
                    // comparison so it remains visibly above the orange y3 curve.
                    renderers["ideal_y36"].glyph.line_width = isY3Comparison ? 3 : 2;
                    // Use longer dashes with tighter gaps in the y3/y36 comparison
                    // so the purple ideal curve is distinct above the orange curve.
                    renderers["ideal_y36"].glyph.line_dash = isY3Comparison ? [8, 2] : [6];
                    renderers["ideal_y36"].glyph.change.emit();
                }

                // Restore the appropriate legend for each visualization.
                if (option === "Test-Data Deviation Analysis") {
                    legend.items = deviation_legend_items;
                    legend.location = "top_right";
                    legend.orientation = "vertical";
                    legend.ncols = 3;
                    legend.label_text_font_size = "12px";
                    legend.label_text_font_style = "bold";
                    legend.padding = 5;
                    legend.spacing = 4;
                    legend.margin = 4;
                    legend.glyph_width = 18;
                    legend.glyph_height = 16;
                    legend.background_fill_color = "white";
                    legend.background_fill_alpha = 1.0;
                    legend.border_line_color = "#b7b7b7";
                    legend.border_line_width = 1;
                } else if (option.startsWith("Compare ")) {
                    legend.items = comparison_legend_items[option];
                    legend.orientation = "vertical";
                    legend.ncols = 1;
                    legend.label_text_font_size = "8pt";
                    legend.label_text_font_style = "normal";
                    if (option === "Compare y3 ↔ y36") {
                        legend.location = "bottom_left";
                    } else if (option === "Compare y4 ↔ y40") {
                        legend.location = "top_right";
                    } else {
                        legend.location = "top_left";
                    }
                } else {
                    if (option === "Training Functions") {
                        legend.items = training_only_legend_items;
                    } else if (option === "Chosen Ideal Functions") {
                        legend.items = ideal_only_legend_items;
                    } else if (option === "Test Data") {
                        legend.items = test_data_only_legend_items;
                    } else if (option === "Mapped Test Points") {
                        legend.items = mapped_points_only_legend_items;
                    } else if (option === "All Data") {
                        legend.items = all_data_legend_items;
                    } else {
                        legend.items = default_legend_items;
                    }
                    legend.location = "top_left";
                    legend.orientation = "vertical";
                    legend.ncols = 1;
                    legend.label_text_font_size = "8pt";
                    legend.label_text_font_style = "normal";
                }

                // Hide all data renderers and deviation-only annotations first.
                for (const key in renderers) {
                    renderers[key].visible = false;
                }
                for (const annotation of deviation_annotations) {
                    annotation.visible = false;
                }

                if (option === "All Data") {
                    for (const key in renderers) {
                        renderers[key].visible = !key.startsWith("deviation_");
                    }
                } else if (option === "Training Functions") {
                    for (const key of training_renderers) renderers[key].visible = true;
                } else if (option === "Chosen Ideal Functions") {
                    for (const key of ideal_renderers) renderers[key].visible = true;
                } else if (option === "Test Data") {
                    renderers["test"].visible = true;
                } else if (option === "Mapped Test Points") {
                    renderers["mapped"].visible = true;
                } else if (option === "Test-Data Deviation Analysis") {
                    for (const key of deviation_renderer_keys) renderers[key].visible = true;
                    for (const annotation of deviation_annotations) annotation.visible = true;
                } else if (option.startsWith("Compare ")) {
                    const names = option.substring(8).split(" ↔ ");
                    const trainingKey = "training_" + names[0];
                    const idealKey = "ideal_" + names[1];
                    if (renderers[trainingKey]) renderers[trainingKey].visible = true;
                    if (renderers[idealKey]) renderers[idealKey].visible = true;
                }

                // Preserve the established axis behavior for every view.
                if (option === "Test-Data Deviation Analysis") {
                    plot.title.text = "Test-Data Deviation Analysis";
                    x_axis.axis_label = "Test Observation X-Coordinate";
                    y_axis.axis_label = "Absolute Deviation |Δy|";
                    x_range.start = -21;
                    x_range.end = 21;
                    y_range.start = 0;
                    y_range.end = 1.4;
                    y_axis.ticker = deviation_ticker;
                } else {
                    const limits = range_presets[option];
                    if (limits) {
                        x_range.start = limits[0];
                        x_range.end = limits[1];
                        y_range.start = limits[2];
                        y_range.end = limits[3];
                    }

                    plot.title.text = option === "All Data"
                        ? "Training, Ideal and Test Data Visualization" : option;
                    x_axis.axis_label = "X Values";
                    y_axis.axis_label = "Y Values (Auto-Scaled)";

                    if (option === "Compare y1 ↔ y13") {
                        x_range.start = -22;
                        x_range.end = 22;
                        y_range.start = -50;
                        y_range.end = 40;
                        x_axis.ticker = y1_y13_x_ticker;
                        y_axis.ticker = y1_y13_y_ticker;
                    } else if (option === "Compare y2 ↔ y24") {
                        x_range.start = -22;
                        x_range.end = 22;
                        y_range.start = -17000;
                        y_range.end = 17000;
                        x_axis.ticker = y1_y13_x_ticker;
                        y_axis.ticker = y2_y24_y_ticker;
                    } else if (option === "Mapped Test Points") {
                        // Add horizontal margin so the -20 tick is not on the left edge.
                        // Keep the Y range, mapped markers, and legend untouched.
                        x_range.start = -22;
                        x_range.end = 22;
                        x_axis.ticker = y1_y13_x_ticker;
                        y_axis.ticker = normal_ticker;
                    } else {
                        y_axis.ticker = normal_ticker;
                    }
                }

                y_range.change.emit();
                x_range.change.emit();
                y_axis.change.emit();
                x_axis.change.emit();
                legend.change.emit();
                plot.change.emit();
            """
        )

        self.dropdown.js_on_change("value", callback)
        return self.dropdown

    def _set_initial_y1_y13_view(self):
        """Initialize the generated HTML directly in the y1 vs y13 comparison view."""

        # Hide every renderer, then show only the requested pair.
        for renderer in self.renderers.values():
            renderer.visible = False

        self.renderers["training_y1"].visible = True
        self.renderers["ideal_y13"].visible = True

        # Hide all deviation-only annotations.
        for annotation in [
            self.deviation_acceptance_region,
            self.deviation_rejection_region,
            self.deviation_acceptance_label,
            self.deviation_rejection_label
        ]:
            annotation.visible = False

        # Set the exact reference axes in Python before the HTML is generated.
        self.plot.title.text = "Training y1 vs. Ideal y13"
        self.plot.xaxis[0].axis_label = "X Values"
        self.plot.yaxis[0].axis_label = "Y Values (Auto-Scaled)"
        self.plot.x_range.start = -22
        self.plot.x_range.end = 22
        self.plot.y_range.start = -50
        self.plot.y_range.end = 40
        self.plot.xaxis[0].ticker = FixedTicker(
            ticks=[-20, -15, -10, -5, 0, 5, 10, 15, 20]
        )
        self.plot.yaxis[0].ticker = FixedTicker(
            ticks=[-40, -20, 0, 20]
        )

        # Replace the full-plot legend with exactly the two reference entries.
        if self.plot.legend:
            self.plot.legend[0].items = [
                LegendItem(
                    label="Training Data",
                    renderers=[self.renderers["training_y1"]]
                ),
                LegendItem(
                    label="Selected Ideal Function",
                    renderers=[self.renderers["ideal_y13"]]
                )
            ]
            self.plot.legend[0].location = "top_left"
            self.plot.legend[0].background_fill_color = "white"
            self.plot.legend[0].background_fill_alpha = 1.0
            self.plot.legend[0].border_line_color = "gray"
            self.plot.legend[0].border_line_width = 1
            self.plot.legend[0].label_text_font_size = "8pt"

    def show_visualization(self):
        """
        Display the complete interactive visualization.
        """

        if self.plot is None:
            self.create_figure()
            self.plot_training_functions()
            self.plot_chosen_ideal_functions()
            self.plot_test_data()
            self.plot_mapped_points()
            self.plot_deviation_analysis()
            self.setup_dropdown()
            self._set_initial_y1_y13_view()

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

    # Step 1: Load datasets
    loader = DataLoader()

    training_df = loader.load_training_data()
    ideal_df = loader.load_ideal_data()
    test_df = loader.load_test_data()

    print("\nAll datasets loaded successfully.")

    # Step 2: Select the best-fitting ideal functions
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

    # Step 3: Map test observations
    mapper = TestMapper(
        test_df,
        ideal_df,
        best_functions
    )

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

    print("\nTotal test observations:", len(test_df))
    print("Successfully mapped observations:", len(mapping_df))

    # Step 4: Initialize the visualizer
    visualizer = DataVisualizer(
        training_df,
        ideal_df,
        test_df,
        mapping_df,
        best_functions,
        mapping_results
    )

    # Step 5: Display the interactive visualization.
    # show_visualization() creates the figure, renderers, and dropdown.
    visualizer.show_visualization()