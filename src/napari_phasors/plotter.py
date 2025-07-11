import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.ticker as ticker
import numpy as np
from biaplotter.plotter import CanvasWidget
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from napari.layers import Image
from napari.utils import colormaps, notifications
from phasorpy.phasor import phasor_from_lifetime
from qtpy import uic
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from .calibration_tab import CalibrationWidget
from .components_tab import ComponentsWidget
from .cursors_tab import CursorWidget
from .filter_tab import FilterWidget
from .fret_tab import FretWidget
from .lifetime_tab import LifetimeWidget

if TYPE_CHECKING:
    import napari


class PlotterWidget(QWidget):
    """A widget for plotting phasor features.

    This widget contains a fixed canvas widget at the top for plotting phasor features
    and a tabbed interface below with different analysis tools. The widget creates a
    phasors selected layer based on the manual selection in the canvas widget.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer object.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer object.
    canvas_widget : biaplotter.plotter.CanvasWidget
        The canvas widget for plotting phasor features (fixed at the top).
    tab_widget : QTabWidget
        The tab widget containing different analysis tools.
    settings_tab : QWidget
        The Settings tab containing the main plotting controls.
    cursors_tab : QWidget
        The Cursors tab for cursor-based analysis.
    components_tab : QWidget
        The Components tab for component analysis.
    lifetime_tab : QWidget
        The Lifetime tab for lifetime analysis.
    fret_tab : QWidget
        The FRET tab for FRET analysis.
    plotter_inputs_widget : QWidget
        The main plotter inputs widget (in Settings tab). The widget contains:
        - image_layer_with_phasor_features_combobox : QComboBox
            The combobox for selecting the image layer with phasor features.
        - phasor_selection_id_combobox : QComboBox
            The combobox for selecting the phasor selection id.
        - harmonic_spinbox : QSpinBox
            The spinbox for selecting the harmonic.
        - threshold_slider : QSlider
            The slider for selecting the threshold.
        - median_filter_spinbox : QSpinBox
            The spinbox for selecting the median filter kernel size (in pixels).
        - semi_circle_checkbox : QCheckBox
            The checkbox for displaying the universal semi-circle (if True) or the full polar plot (if False).
    plotter_inputs_widget : QWidget
        The extra plotter inputs widget (in Settings tab). It is collapsible. The widget contains:
        - plot_type_combobox : QComboBox
            The combobox for selecting the plot type.
        - colormap_combobox : QComboBox
            The combobox for selecting the histogram colormap.
        - number_of_bins_spinbox : QSpinBox
            The spinbox for selecting the number of bins in the histogram.
        - log_scale_checkbox : QCheckBox
            The checkbox for selecting the log scale in the histogram.
    plot_button : QPushButton
        The plot button (in Settings tab).
    _labels_layer_with_phasor_features : Labels
        The labels layer with phasor features.
    _phasors_selected_layer : Labels
        The phasors selected layer.
    _colormap : matplotlib.colors.Colormap
        The colormap for the canvas widget.
    _histogram_colormap : matplotlib.colors.Colormap
        The histogram colormap for the canvas widget.

    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self._labels_layer_with_phasor_features = None

        # Create a splitter to separate canvas from controls
        splitter = QSplitter(Qt.Vertical)
        self.layout().addWidget(splitter)

        # Create top widget for canvas
        canvas_container = QWidget()
        canvas_container.setLayout(QVBoxLayout())

        # Load canvas widget (fixed at the top)
        self.canvas_widget = CanvasWidget(napari_viewer)
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self.canvas_widget.setMinimumSize(600, 400)
        self.canvas_widget.class_spinbox.setValue(1)
        self.set_axes_labels()
        canvas_container.layout().addWidget(self.canvas_widget)

        # Create bottom widget for controls
        controls_container = QWidget()
        controls_container.setLayout(QVBoxLayout())

        # Add select image combobox
        controls_container.layout().addWidget(QLabel("Image Layer:"))
        self.image_layer_with_phasor_features_combobox = QComboBox()
        controls_container.layout().addWidget(
            self.image_layer_with_phasor_features_combobox
        )

        # Add harmonic spinbox below image layer combobox
        controls_container.layout().addWidget(QLabel("Harmonic:"))
        self.harmonic_spinbox = QSpinBox()
        self.harmonic_spinbox.setMinimum(1)
        self.harmonic_spinbox.setValue(1)
        controls_container.layout().addWidget(self.harmonic_spinbox)

        # Create tab widget
        self.tab_widget = QTabWidget()
        controls_container.layout().addWidget(self.tab_widget)

        # Add widgets to splitter
        splitter.addWidget(canvas_container)
        splitter.addWidget(controls_container)

        # Configure splitter to prevent overlap
        splitter.setStretchFactor(0, 1)  # Canvas gets priority
        splitter.setStretchFactor(1, 0)  # Controls maintain minimum size
        splitter.setCollapsible(0, False)  # Canvas cannot be collapsed
        splitter.setCollapsible(1, True)  # Controls can be collapsed if needed

        # Set minimum heights
        canvas_container.setMinimumHeight(400)
        controls_container.setMinimumHeight(300)

        # Set initial splitter sizes (canvas gets more space)
        splitter.setSizes([400, 600])

        # Create Settings tab
        self.settings_tab = QWidget()
        self.settings_tab.setLayout(QVBoxLayout())
        self.tab_widget.addTab(self.settings_tab, "Plot Settings")

        # Load plotter inputs widget from ui file (moved to Settings tab)
        self.plotter_inputs_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/plotter_inputs_widget.ui",
            self.plotter_inputs_widget,
        )
        self.settings_tab.layout().addWidget(self.plotter_inputs_widget)

        # Set minimum size (increased height to accommodate both sections)
        self.setMinimumSize(600, 800)

        # Connect napari signals when new layer is inseted or removed
        self.viewer.layers.events.inserted.connect(self.reset_layer_choices)
        self.viewer.layers.events.removed.connect(self.reset_layer_choices)

        # Connect callbacks
        self.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.on_labels_layer_with_phasor_features_changed
        )
        self.plotter_inputs_widget.semi_circle_checkbox.stateChanged.connect(
            self.on_toggle_semi_circle
        )
        self.harmonic_spinbox.valueChanged.connect(self.plot)
        self.plotter_inputs_widget.plot_type_combobox.currentIndexChanged.connect(
            self.plot
        )
        self.plotter_inputs_widget.colormap_combobox.currentIndexChanged.connect(
            self.plot
        )
        self.plotter_inputs_widget.number_of_bins_spinbox.valueChanged.connect(
            self.plot
        )
        self.plotter_inputs_widget.log_scale_checkbox.stateChanged.connect(
            self.plot
        )
        self.plotter_inputs_widget.white_background_checkbox.stateChanged.connect(
            self.on_white_background_changed
        )

        # Populate plot type combobox
        self.plotter_inputs_widget.plot_type_combobox.addItems(
            ['SCATTER', 'HISTOGRAM2D']
        )
        # Populate colormap combobox
        self.plotter_inputs_widget.colormap_combobox.addItems(
            list(colormaps.ALL_COLORMAPS.keys())
        )
        self.histogram_colormap = (
            "turbo"  # Set default colormap (same as in biaplotter)
        )

        # Initialize attributes
        self.polar_plot_artist_list = []
        self.semi_circle_plot_artist_list = []
        self.toggle_semi_circle = True
        self.colorbar = None
        self._colormap = self.canvas_widget.artists[
            'HISTOGRAM2D'
        ].overlay_colormap
        self._histogram_colormap = self.canvas_widget.artists[
            'HISTOGRAM2D'
        ].histogram_colormap
        # Start with the histogram2d plot type
        self.plot_type = 'HISTOGRAM2D'

        # Create other tabs
        self._create_calibration_tab()
        self._create_filter_tab()
        self._create_cursors_tab()
        self._create_components_tab()
        self._create_lifetime_tab()
        self._create_fret_tab()

        # Connect canvas signals
        self.canvas_widget.artists[
            'SCATTER'
        ].color_indices_changed_signal.connect(
            self.cursors_tab.manual_selection_changed
        )
        self.canvas_widget.artists[
            'HISTOGRAM2D'
        ].color_indices_changed_signal.connect(
            self.cursors_tab.manual_selection_changed
        )

        # Set intial axes limits
        self._redefine_axes_limits()
        # Set initial background color
        self._update_plot_bg_color()
        # Populate labels layer combobox
        self.reset_layer_choices()

        self.canvas_widget.figure.canvas.mpl_connect(
            'button_press_event', self._on_canvas_click
        )
    
    def _on_canvas_click(self, event):
        """Handle click events on the canvas widget."""
        if event.inaxes != self.canvas_widget.axes:
            return None, None
        # Check if the click is on the canvas widget axes
        if event.button == 1:  # Left click
            # Get the coordinates of the click
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                return x, y
        return None, None


    def on_white_background_changed(self, state):
        """Callback function when the white background checkbox is toggled."""
        self.set_axes_labels()
        # Update both polar and semicircle plots to refresh their colors
        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)
        else:
            self._update_polar_plot(self.canvas_widget.axes, visible=True)
        # Force canvas redraw to update colors
        self.canvas_widget.figure.canvas.draw_idle()
        self.plot()

    @property
    def white_background(self):
        """Gets the white background value from the white background checkbox.

        Returns
        -------
        bool
            The white background value.
        """
        return self.plotter_inputs_widget.white_background_checkbox.isChecked()

    @white_background.setter
    def white_background(self, value: bool):
        """Sets the white background value from the white background checkbox."""
        self.plotter_inputs_widget.white_background_checkbox.setChecked(value)
        self.set_axes_labels()
        self.plot()

    def _create_calibration_tab(self):
        """Create the Calibration tab."""
        self.calibration_tab = CalibrationWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.calibration_tab, "Calibration")

        self.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.calibration_tab._on_image_layer_changed
        )

    def _create_filter_tab(self):
        """Create the Filtering and Thresholding tab."""
        self.filter_tab = FilterWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.filter_tab, "Filter/Threshold")

    def _create_cursors_tab(self):
        """Create the Cursor selection tab."""
        self.cursors_tab = CursorWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.cursors_tab, "Selection")

    def _create_components_tab(self):
        """Create the Components tab."""
        self.components_tab = ComponentsWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.components_tab, "Components")

    def _create_lifetime_tab(self):
        """Create the Lifetime tab."""
        self.lifetime_tab = LifetimeWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.lifetime_tab, "Lifetime")

        self.harmonic_spinbox.valueChanged.connect(
            self.lifetime_tab._on_harmonic_changed
        )
        self.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.lifetime_tab._on_image_layer_changed
        )

    def _create_fret_tab(self):
        """Create the FRET tab."""
        self.fret_tab = FretWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.fret_tab, "FRET")

        self.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.lifetime_tab._on_image_layer_changed
        )

    @property
    def harmonic(self):
        """Gets or sets the harmonic value from the harmonic spinbox.

        Returns
        -------
        int
            The harmonic value.
        """
        return self.harmonic_spinbox.value()

    @harmonic.setter
    def harmonic(self, value: int):
        """Sets the harmonic value from the harmonic spinbox."""
        if value < 1:
            notifications.WarningNotification(
                f"Harmonic value should be greater than 0. Setting to 1."
            )
            value = 1
        self.harmonic_spinbox.setValue(value)

    @property
    def toggle_semi_circle(self):
        """Gets the display semi circle value from the semi circle checkbox.

        Returns
        -------
        bool
            The display semi circle value.
        """
        return self.plotter_inputs_widget.semi_circle_checkbox.isChecked()

    @property
    def toggle_semi_circle(self):
        """Gets the display semi circle value from the semi circle checkbox.

        Returns
        -------
        bool
            The display semi circle value.
        """
        return self.plotter_inputs_widget.semi_circle_checkbox.isChecked()

    @toggle_semi_circle.setter
    def toggle_semi_circle(self, value: bool):
        """Sets the display semi circle value from the semi circle checkbox."""
        self.plotter_inputs_widget.semi_circle_checkbox.setChecked(value)
        if value:
            self._update_polar_plot(self.canvas_widget.axes, visible=False)
            self._update_semi_circle_plot(self.canvas_widget.axes)
        else:
            self._update_semi_circle_plot(
                self.canvas_widget.axes, visible=False
            )
            self._update_polar_plot(self.canvas_widget.axes)
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self._redefine_axes_limits()
        # Force canvas redraw
        self.canvas_widget.figure.canvas.draw_idle()

    def on_toggle_semi_circle(self, state):
        """Callback function when the semi circle checkbox is toggled.

        This function updates the `toggle_semi_circle` attribute with the checked status of the checkbox.
        And it displays either the universal semi-circle or the full polar plot in the canvas widget.
        """
        self.toggle_semi_circle = state

    def _update_polar_plot(self, ax, visible=True, alpha=0.5, zorder=3):
        """
        Generate the polar plot in the canvas widget.

        Build the inner and outer circle and the 45 degrees lines in the plot.
        """
        # Determine color based on background
        line_color = 'black' if self.white_background else 'white'

        if len(self.polar_plot_artist_list) > 0:
            for artist in self.polar_plot_artist_list:
                artist.set_visible(visible)
                artist.set_alpha(alpha)
                artist.set_color(line_color)
        else:
            self.polar_plot_artist_list.append(
                ax.add_line(
                    Line2D(
                        [-1, 1],
                        [0, 0],
                        linestyle='-',
                        linewidth=1,
                        color=line_color,
                    )
                )
            )
            self.polar_plot_artist_list.append(
                ax.add_line(
                    Line2D(
                        [0, 0],
                        [-1, 1],
                        linestyle='-',
                        linewidth=1,
                        color=line_color,
                    )
                )
            )
            circle = Circle((0, 0), 1, fill=False, color=line_color)
            self.polar_plot_artist_list.append(ax.add_patch(circle))
            for r in (1 / 3, 2 / 3):
                circle = Circle((0, 0), r, fill=False, color=line_color)
                self.polar_plot_artist_list.append(ax.add_patch(circle))
            for a in (3, 6):
                x = math.cos(math.pi / a)
                y = math.sin(math.pi / a)
                self.polar_plot_artist_list.append(
                    ax.add_line(
                        Line2D(
                            [-x, x],
                            [-y, y],
                            linestyle=':',
                            linewidth=0.5,
                            color=line_color,
                        )
                    )
                )
                self.polar_plot_artist_list.append(
                    ax.add_line(
                        Line2D(
                            [-x, x],
                            [y, -y],
                            linestyle=':',
                            linewidth=0.5,
                            color=line_color,
                        )
                    )
                )
        return ax

    def _update_semi_circle_plot(self, ax, visible=True, alpha=0.5, zorder=3):
        '''
        Generate FLIM universal semi-circle plot with lifetime ticks if frequency is available
        '''
        # Determine color based on background
        line_color = 'black' if self.white_background else 'white'

        # Always clear existing artists and regenerate to ensure ticks are updated
        if len(self.semi_circle_plot_artist_list) > 0:
            for artist in self.semi_circle_plot_artist_list:
                artist.remove()
            self.semi_circle_plot_artist_list.clear()

        # Only create new artists if visible is True
        if visible:
            # Create semicircle
            angles = np.linspace(0, np.pi, 180)
            x = (np.cos(angles) + 1) / 2
            y = np.sin(angles) / 2
            self.semi_circle_plot_artist_list.append(
                ax.plot(
                    x,
                    y,
                    color=line_color,
                    alpha=alpha,
                    visible=visible,
                    zorder=zorder,
                )[0]
            )

            # Add lifetime ticks if frequency is available
            self._add_lifetime_ticks_to_semicircle(ax, visible, alpha, zorder)

        return ax

    def _add_lifetime_ticks_to_semicircle(
        self, ax, visible=True, alpha=0.5, zorder=3
    ):
        """Add lifetime ticks to the semicircle plot based on frequency in layer metadata."""
        # Get frequency from layer metadata
        frequency = self._get_frequency_from_layer()
        if frequency is None:
            return

        # Determine color based on background
        tick_color = 'black' if self.white_background else 'darkgray'

        # Generate lifetime values using powers of 2, similar to the reference function
        # Start with 0 and then add powers of 2 that are visible on the semicircle
        lifetimes = [0.0]

        # Add powers of 2 that result in S coordinates >= 0.18 (visible on semicircle)
        for t in range(-8, 32):  # Wide range to cover different frequencies
            lifetime_val = 2**t
            try:
                g_pos, s_pos = phasor_from_lifetime(frequency, lifetime_val)
                # Only include if S coordinate is >= 0.18 (visible on semicircle)
                if s_pos >= 0.18:
                    lifetimes.append(lifetime_val)
            except:
                # Skip if phasor_from_lifetime fails for this value
                continue

        # Calculate phasor coordinates for each lifetime
        for i, lifetime in enumerate(lifetimes):
            if lifetime == 0:
                # For lifetime 0, position is at (1, 0)
                g_pos, s_pos = 1.0, 0.0
            else:
                # Use phasor_from_lifetime to get G and S coordinates
                g_pos, s_pos = phasor_from_lifetime(frequency, lifetime)

            # Calculate radial direction from center of semicircle (0.5, 0)
            center_x, center_y = 0.5, 0.0
            # Vector from center to point on semicircle
            dx = g_pos - center_x
            dy = s_pos - center_y
            # Normalize the vector
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
            else:
                # For lifetime 0, point radially outward
                dx_norm = 1.0
                dy_norm = 0.0

            # Tick length
            tick_length = 0.03

            # Start tick at the semicircle point, extend outward
            tick_start_x = g_pos
            tick_start_y = s_pos
            tick_end_x = g_pos + tick_length * dx_norm
            tick_end_y = s_pos + tick_length * dy_norm

            # Add tick mark outside the semicircle
            tick_line = ax.plot(
                [tick_start_x, tick_end_x],
                [tick_start_y, tick_end_y],
                color=tick_color,
                linewidth=1.5,
                alpha=alpha,
                visible=visible,
                zorder=zorder + 1,
            )[0]
            self.semi_circle_plot_artist_list.append(tick_line)

            # Add lifetime label
            if lifetime == 0:
                label_text = "0"
            elif lifetime < 1:
                label_text = f"{lifetime:g}"
            else:
                label_text = f"{lifetime:g}"

            # Add units to the third label (index 2) if it exists
            if i == 2 and len(lifetimes) > 2:
                label_text += " ns"

            # Position label further outside the semicircle
            label_offset = 0.08
            label_x = g_pos + label_offset * dx_norm
            label_y = s_pos + label_offset * dy_norm

            # Use same color as ticks
            text_color = tick_color

            label = ax.text(
                label_x,
                label_y,
                label_text,
                fontsize=8,
                ha='center',
                va='center',
                color=text_color,
                alpha=alpha,
                visible=visible,
                zorder=zorder + 1,
            )
            self.semi_circle_plot_artist_list.append(label)

    def _get_frequency_from_layer(self):
        """Get frequency from the current layer's metadata."""
        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if layer_name == "":
            return None
        layer = self.viewer.layers[layer_name]
        if "settings" in layer.metadata:
            settings = layer.metadata["settings"]
            if "frequency" in settings:
                return settings["frequency"]

        return None

    def _redefine_axes_limits(self, ensure_full_circle_displayed=True):
        """
        Redefine axes limits based on the data plotted in the canvas widget.

        Parameters
        ----------
        ensure_full_circle_displayed : bool, optional
            Whether to ensure the full circle is displayed in the canvas widget, by default True.
        """
        # Redefine axes limits
        if self.toggle_semi_circle:
            # Get semi circle plot limits
            circle_plot_limits = [0, 1, 0, 0.6]  # xmin, xmax, ymin, ymax
        else:
            # Get polar plot limits
            circle_plot_limits = [-1, 1, -1, 1]  # xmin, xmax, ymin, ymax
        # Check if histogram is plotted
        if self.canvas_widget.artists['HISTOGRAM2D'].histogram is not None:
            # Get histogram data limits
            x_edges = self.canvas_widget.artists['HISTOGRAM2D'].histogram[1]
            y_edges = self.canvas_widget.artists['HISTOGRAM2D'].histogram[2]
            plotted_data_limits = [
                x_edges[0],
                x_edges[-1],
                y_edges[0],
                y_edges[-1],
            ]
        else:
            plotted_data_limits = circle_plot_limits
        # Check if full circle should be displayed
        if not ensure_full_circle_displayed:
            # If not, only the data limits are used
            circle_plot_limits = plotted_data_limits

        x_range = np.amax(
            [plotted_data_limits[1], circle_plot_limits[1]]
        ) - np.amin([plotted_data_limits[0], circle_plot_limits[0]])
        y_range = np.amax(
            [plotted_data_limits[3], circle_plot_limits[3]]
        ) - np.amin([plotted_data_limits[2], circle_plot_limits[2]])
        # 10% of the range as a frame
        xlim_0 = (
            np.amin([plotted_data_limits[0], circle_plot_limits[0]])
            - 0.1 * x_range
        )
        xlim_1 = (
            np.amax([plotted_data_limits[1], circle_plot_limits[1]])
            + 0.1 * x_range
        )
        ylim_0 = (
            np.amin([plotted_data_limits[2], circle_plot_limits[2]])
            - 0.1 * y_range
        )
        ylim_1 = (
            np.amax([plotted_data_limits[3], circle_plot_limits[3]])
            + 0.1 * y_range
        )

        self.canvas_widget.axes.set_ylim([ylim_0, ylim_1])
        self.canvas_widget.axes.set_xlim([xlim_0, xlim_1])
        self.canvas_widget.figure.canvas.draw_idle()

    def _update_plot_bg_color(self, color=None):
        """Change the background color of the canvas widget.

        Parameters
        ----------
        color : str, optional
            The color to set the background, by default None. If None, the background
            will be set based on the white_background checkbox state.
        """
        if color is None:
            if self.white_background:
                color = "white"
            else:
                color = "none"  # Transparent background

        if color == "none":
            # Set transparent background only for the axes (plot area)
            self.canvas_widget.axes.set_facecolor('none')
            # Keep the figure background transparent/default
            self.canvas_widget.figure.patch.set_facecolor('none')
        else:
            # Set solid background color only for the axes (plot area)
            self.canvas_widget.axes.set_facecolor(color)
            # Keep the figure background transparent so only the plot area is colored
            self.canvas_widget.figure.patch.set_facecolor('none')

        self.canvas_widget.figure.canvas.draw_idle()

    @property
    def plot_type(self):
        """Gets or sets the plot type from the plot type combobox.

        Returns
        -------
        str
            The plot type.
        """
        return self.plotter_inputs_widget.plot_type_combobox.currentText()

    @plot_type.setter
    def plot_type(self, type):
        """Sets the plot type from the plot type combobox."""
        self.plotter_inputs_widget.plot_type_combobox.setCurrentText(type)

    @property
    def histogram_colormap(self):
        """Gets or sets the histogram colormap from the colormap combobox.

        Returns
        -------
        str
            The colormap name.
        """
        return self.plotter_inputs_widget.colormap_combobox.currentText()

    @histogram_colormap.setter
    def histogram_colormap(self, colormap: str):
        """Sets the histogram colormap from the colormap combobox."""
        if colormap not in colormaps.ALL_COLORMAPS.keys():
            notifications.WarningNotification(
                f"{colormap} is not a valid colormap. Setting to default colormap."
            )
            colormap = self._histogram_colormap.name
        self.plotter_inputs_widget.colormap_combobox.setCurrentText(colormap)

    @property
    def histogram_bins(self):
        """Gets the histogram bins from the histogram bins spinbox.

        Returns
        -------
        int
            The histogram bins value.
        """
        return self.plotter_inputs_widget.number_of_bins_spinbox.value()

    @histogram_bins.setter
    def histogram_bins(self, value: int):
        """Sets the histogram bins from the histogram bins spinbox."""
        if value < 2:
            notifications.WarningNotification(
                f"Number of bins should be greater than 1. Setting to 10."
            )
            value = 10
        self.plotter_inputs_widget.number_of_bins_spinbox.setValue(value)

    @property
    def histogram_log_scale(self):
        """Gets the histogram log scale from the histogram log scale checkbox.

        Returns
        -------
        bool
            The histogram log scale value.
        """
        return self.plotter_inputs_widget.log_scale_checkbox.isChecked()

    @histogram_log_scale.setter
    def histogram_log_scale(self, value: bool):
        """Sets the histogram log scale from the histogram log scale checkbox."""
        self.plotter_inputs_widget.log_scale_checkbox.setChecked(value)

    def reset_layer_choices(self):
        """Reset the image layer with phasor features combobox choices.

        This function is called when a new layer is added or removed.
        It also updates `_labels_layer_with_phasor_features` attribute with the Labels layer in the metadata of the selected image layer.
        """
        self.image_layer_with_phasor_features_combobox.clear()
        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and "phasor_features_labels_layer" in layer.metadata.keys()
        ]
        self.image_layer_with_phasor_features_combobox.addItems(layer_names)
        # Update layer names in the phasor selection id combobox when layer name changes
        for layer_name in layer_names:
            layer = self.viewer.layers[layer_name]
            layer.events.name.connect(self.reset_layer_choices)
        self.on_labels_layer_with_phasor_features_changed()

    def on_labels_layer_with_phasor_features_changed(self):
        if getattr(
            self, "_in_on_labels_layer_with_phasor_features_changed", False
        ):
            return
        self._in_on_labels_layer_with_phasor_features_changed = True
        try:
            labels_layer_name = (
                self.image_layer_with_phasor_features_combobox.currentText()
            )
            if labels_layer_name == "":
                self._labels_layer_with_phasor_features = None
                return
            layer_metadata = self.viewer.layers[labels_layer_name].metadata
            self._labels_layer_with_phasor_features = layer_metadata[
                "phasor_features_labels_layer"
            ]
            self.harmonic_spinbox.setMaximum(
                self._labels_layer_with_phasor_features.features[
                    "harmonic"
                ].max()
            )
            self.cursors_tab.add_selection_id_to_features(
                "MANUAL SELECTION #1"
            )

            # No need to manually clear - _update_semi_circle_plot will handle it
            self.plot()
        finally:
            self._in_on_labels_layer_with_phasor_features_changed = False

    def get_features(self):
        """Get the G and S features for the selected harmonic and selection id.

        Returns
        -------
        x_data : np.ndarray
            The G feature data.
        y_data : np.ndarray
            The S feature data.
        selection_data : np.ndarray
            The selection data.
        """
        if self._labels_layer_with_phasor_features is None:
            return None
        # Check if layer contains features
        if self._labels_layer_with_phasor_features.features is None:
            return None

        table = self._labels_layer_with_phasor_features.features
        x_data = table['G'][table['harmonic'] == self.harmonic].values
        y_data = table['S'][table['harmonic'] == self.harmonic].values
        mask = np.isnan(x_data) & np.isnan(y_data)
        x_data = x_data[~mask]
        y_data = y_data[~mask]
        if (
            self.cursors_tab.selection_id is None
            or self.cursors_tab.selection_id == ""
        ):
            return x_data, y_data, np.zeros_like(x_data)
        else:
            selection_data = table[self.cursors_tab.selection_id][
                table['harmonic'] == self.harmonic
            ].values
            selection_data = selection_data[~mask]

        return x_data, y_data, selection_data

    def set_axes_labels(self):
        """Set the axes labels in the canvas widget."""
        # Always use white text color for napari compatibility
        text_color = "white"

        self.canvas_widget.artists['SCATTER'].ax.set_xlabel(
            "G", color=text_color
        )
        self.canvas_widget.artists['SCATTER'].ax.set_ylabel(
            "S", color=text_color
        )
        self.canvas_widget.artists['HISTOGRAM2D'].ax.set_xlabel(
            "G", color=text_color
        )
        self.canvas_widget.artists['HISTOGRAM2D'].ax.set_ylabel(
            "S", color=text_color
        )

        # Update tick colors
        self.canvas_widget.artists['SCATTER'].ax.tick_params(colors=text_color)
        self.canvas_widget.artists['HISTOGRAM2D'].ax.tick_params(
            colors=text_color
        )

        # Update spine colors
        for spine in self.canvas_widget.artists['SCATTER'].ax.spines.values():
            spine.set_color(text_color)
        for spine in self.canvas_widget.artists[
            'HISTOGRAM2D'
        ].ax.spines.values():
            spine.set_color(text_color)

    def plot(self, x_data=None, y_data=None, selection_id_data=None):
        """Plot the selected phasor features.

        This function plots the selected phasor features in the canvas widget.
        It also creates the phasors selected layer.
        """
        # TODO: split this into an update and a plot to avoid doing unnecessary operations during update
        if self._labels_layer_with_phasor_features is None:
            return

        if x_data is None or y_data is None or selection_id_data is None:
            features = self.get_features()
            if features is None:
                return
            x_data, y_data, selection_id_data = features
        # Set active artist
        self.canvas_widget.active_artist = self.plot_type

        # Set data in the active artist
        self.canvas_widget.active_artist.data = np.column_stack(
            (x_data, y_data)
        )
        self.canvas_widget.artists['HISTOGRAM2D'].cmin = 1
        # Set selection data in the active artist
        self.canvas_widget.active_artist.color_indices = selection_id_data
        # Set colormap in the active artist
        selected_histogram_colormap = colormaps.ALL_COLORMAPS[
            self.histogram_colormap
        ]
        # Temporary convertion to LinearSegmentedColormap to match matplotlib format, while biaplotter is not updated
        selected_histogram_colormap = LinearSegmentedColormap.from_list(
            selected_histogram_colormap.name,
            selected_histogram_colormap.colors,
        )
        self.canvas_widget.artists['HISTOGRAM2D'].histogram_colormap = (
            selected_histogram_colormap
        )
        # Set number of bins in the active artist
        self.canvas_widget.artists['HISTOGRAM2D'].bins = self.histogram_bins
        # Temporarily set active artist "again" to have it displayed on top #TODO: Fix this
        self.canvas_widget.active_artist = self.plot_type
        # Set log scale in the active artist
        if self.canvas_widget.artists['HISTOGRAM2D'].histogram is not None:
            if self.histogram_log_scale:
                self.canvas_widget.artists[
                    'HISTOGRAM2D'
                ].histogram_color_normalization_method = "log"
            else:
                self.canvas_widget.artists[
                    'HISTOGRAM2D'
                ].histogram_color_normalization_method = "linear"

        # if active artist is histogram, add a colorbar
        if self.plot_type == 'HISTOGRAM2D':
            if self.colorbar is not None:
                self.colorbar.remove()
            # Create cax for colorbar on the right side of the histogram
            self.cax = self.canvas_widget.artists['HISTOGRAM2D'].ax.inset_axes(
                [1.05, 0, 0.05, 1]
            )
            # Create colorbar
            self.colorbar = Colorbar(
                ax=self.cax,
                cmap=selected_histogram_colormap,
                norm=self.canvas_widget.artists[
                    'HISTOGRAM2D'
                ]._get_normalization(
                    self.canvas_widget.artists['HISTOGRAM2D'].histogram[0],
                    is_overlay=False,
                ),
            )

            # Set colorbar style - always white for napari compatibility
            self.set_colorbar_style(color="white")
        else:
            if self.colorbar is not None:
                # remove colorbar
                self.colorbar.remove()
                self.colorbar = None

        # Update semicircle plot if it's visible (to refresh ticks when frequency changes)
        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)

        # Update axes limits
        self._redefine_axes_limits()
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        # Update plot background color (only the plot area, not the entire figure)
        self._update_plot_bg_color()

    def set_colorbar_style(self, color="white"):
        """Set the colorbar style in the canvas widget."""
        # Set colorbar tick color
        self.colorbar.ax.yaxis.set_tick_params(color=color)
        # Set colorbar edgecolor
        self.colorbar.outline.set_edgecolor(color)
        # Add label to colorbar
        if isinstance(self.colorbar.norm, LogNorm):
            self.colorbar.ax.set_ylabel("Log10(Count)", color=color)
        else:
            self.colorbar.ax.set_ylabel("Count", color=color)
        # Get the current ticks
        ticks = self.colorbar.ax.get_yticks()
        # Set the ticks using a FixedLocator
        self.colorbar.ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
        # Set colorbar ticklabels colors individually (this may fail for some math expressions)
        tick_labels = self.colorbar.ax.get_yticklabels()
        for tick_label in tick_labels:
            tick_label.set_color(color)


if __name__ == "__main__":
    import napari

    time_constants = [0.1, 1, 2, 3, 4, 5, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    viewer = napari.Viewer()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter, area="right")
    napari.run()
