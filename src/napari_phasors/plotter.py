import math
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np
from biaplotter.plotter import CanvasWidget
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from napari.layers import Image
from napari.utils import colormaps, notifications
from phasorpy.lifetime import phasor_from_lifetime
from qtpy import uic
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QFrame,
)

from ._utils import update_frequency_in_metadata
from .calibration_tab import CalibrationWidget
from .components_tab import ComponentsWidget
from .filter_tab import FilterWidget
from .fret_tab import FretWidget
from .lifetime_tab import LifetimeWidget
from .selection_tab import SelectionWidget


class PhasorLayerSelector(QWidget):
    """A widget for selecting multiple phasor layers with checkboxes."""
    
    layers_changed = Signal()
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(2)
        
        # Add header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Phasor Layers:"))
        
        # Add select all/none buttons
        select_all_btn = QPushButton("All")
        select_all_btn.setMaximumWidth(40)
        select_all_btn.setMaximumHeight(20)
        select_all_btn.clicked.connect(self._select_all)
        
        select_none_btn = QPushButton("None")
        select_none_btn.setMaximumWidth(40)
        select_none_btn.setMaximumHeight(20)
        select_none_btn.clicked.connect(self._select_none)
        
        header_layout.addWidget(select_all_btn)
        header_layout.addWidget(select_none_btn)
        header_layout.addStretch()
        
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        self.layout().addWidget(header_widget)
        
        # Create scrollable area for layer checkboxes
        self.layer_checkboxes = {}
        self.layers_frame = QFrame()
        self.layers_frame.setLayout(QVBoxLayout())
        self.layers_frame.layout().setContentsMargins(5, 0, 0, 0)
        self.layers_frame.setMaximumHeight(150)  # Limit height
        
        self.layout().addWidget(self.layers_frame)
        
        # Connect to viewer events
        self.viewer.layers.events.inserted.connect(self._update_layer_list)
        self.viewer.layers.events.removed.connect(self._update_layer_list)
        
        self._update_layer_list()
    
    def _update_layer_list(self):
        """Update the list of available phasor layers."""
        # Clear existing checkboxes
        for checkbox in self.layer_checkboxes.values():
            checkbox.deleteLater()
        self.layer_checkboxes.clear()
        
        # Get available phasor layers
        phasor_layers = [
            layer for layer in self.viewer.layers
            if isinstance(layer, Image)
            and "phasor_features_labels_layer" in layer.metadata.keys()
        ]
        
        # Create checkboxes for each layer
        for layer in phasor_layers:
            checkbox = QCheckBox(layer.name)
            checkbox.setMaximumHeight(20)
            checkbox.stateChanged.connect(self._on_selection_changed)
            
            # Connect to layer name changes
            layer.events.name.connect(lambda event, cb=checkbox: cb.setText(event.source.name))
            
            self.layer_checkboxes[layer.name] = checkbox
            self.layers_frame.layout().addWidget(checkbox)
        
        # Add stretch at the end
        self.layers_frame.layout().addStretch()
        
        self._on_selection_changed()
    
    def _select_all(self):
        """Select all available layers."""
        for checkbox in self.layer_checkboxes.values():
            checkbox.setChecked(True)
    
    def _select_none(self):
        """Deselect all layers."""
        for checkbox in self.layer_checkboxes.values():
            checkbox.setChecked(False)
    
    def _on_selection_changed(self):
        """Emit signal when selection changes."""
        self.layers_changed.emit()
    
    def get_selected_layers(self):
        """Get list of selected layer objects."""
        selected_layers = []
        for layer_name, checkbox in self.layer_checkboxes.items():
            if checkbox.isChecked() and layer_name in self.viewer.layers:
                layer = self.viewer.layers[layer_name]
                if "phasor_features_labels_layer" in layer.metadata:
                    phasor_layer = layer.metadata["phasor_features_labels_layer"]
                    selected_layers.append(phasor_layer)
        return selected_layers
    
    def get_selected_layer_names(self):
        """Get list of selected layer names."""
        return [
            layer_name for layer_name, checkbox in self.layer_checkboxes.items()
            if checkbox.isChecked()
        ]


class PlotterWidget(QWidget):
    """A widget for plotting phasor features.

    This widget contains a fixed canvas widget at the top for plotting phasor features
    and a tabbed interface below with different analysis tools.

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
    layer_selector : PhasorLayerSelector
        The widget for selecting multiple phasor layers with checkboxes.
    harmonic_spinbox : QSpinBox
        The spinbox for selecting the harmonic.
    tab_widget : QTabWidget
        The tab widget containing different analysis tools.
    settings_tab : QWidget
        The Settings tab containing the main plotting controls.
    selection_tab : QWidget
        The selection tab for cursor-based analysis.
    components_tab : QWidget
        The Components tab for component analysis.
    lifetime_tab : QWidget
        The Lifetime tab for lifetime analysis.
    fret_tab : QWidget
        The FRET tab for FRET analysis.
    plotter_inputs_widget : QWidget
        The main plotter inputs widget (in Settings tab). The widget contains:
        - semi_circle_checkbox : QCheckBox
            The checkbox for toggling the display of the semi-circle plot.
        - white_background_checkbox : QCheckBox
            The checkbox for toggling the white background in the plot.
        - plot_type_combobox : QComboBox
            The combobox for selecting the plot type.
        - colormap_combobox : QComboBox
            The combobox for selecting the histogram colormap.
        - number_of_bins_spinbox : QSpinBox
            The spinbox for selecting the number of bins in the histogram.
        - log_scale_checkbox : QCheckBox
            The checkbox for selecting the log scale in the histogram.

    """

    def __init__(self, napari_viewer):
        """Initialize the PlotterWidget."""
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self._labels_layers_with_phasor_features = []

        # Create a splitter to separate canvas from controls
        splitter = QSplitter(Qt.Vertical)
        self.layout().addWidget(splitter)

        # Create top widget for canvas
        canvas_container = QWidget()
        canvas_container.setLayout(QVBoxLayout())

        # Load canvas widget (fixed at the top)
        self.canvas_widget = CanvasWidget(
            napari_viewer, highlight_enabled=False
        )
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self.canvas_widget.setMinimumSize(300, 300)
        self.set_axes_labels()
        canvas_container.layout().addWidget(self.canvas_widget)

        # Create bottom widget for controls
        controls_container = QWidget()
        controls_container.setLayout(QVBoxLayout())

        # Create improved layer selector
        self.layer_selector = PhasorLayerSelector(self.viewer)
        self.layer_selector.layers_changed.connect(self.on_layers_selection_changed)
        controls_container.layout().addWidget(self.layer_selector)

        # Add harmonic spinbox below image layer comboboxes
        harmonic_layout = QHBoxLayout()
        harmonic_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        harmonic_layout.setSpacing(5)  # Reduce spacing between widgets
        harmonic_layout.addWidget(QLabel("Harmonic:"))
        self.harmonic_spinbox = QSpinBox()
        self.harmonic_spinbox.setMinimum(1)
        self.harmonic_spinbox.setValue(1)
        self.harmonic_spinbox.setMaximumHeight(25)  # Set smaller height
        harmonic_layout.addWidget(
            self.harmonic_spinbox, 1
        )  # Add stretch factor of 1

        harmonic_widget = QWidget()
        harmonic_widget.setLayout(harmonic_layout)
        controls_container.layout().addWidget(harmonic_widget)

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
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background: #414851;         /* default state */
            }
            QSplitter::handle:hover {
                background: #414851;         /* on hover */
            }
            QSplitter::handle:pressed {
                background: #7b8d8e;         /* while dragging */
            }
        """
        )

        canvas_container.setMinimumHeight(300)
        controls_container.setMinimumHeight(300)
        splitter.setSizes([800, 400])

        # Add a flag to prevent recursive calls
        self._updating_plot = False

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
        self.setMinimumSize(300, 400)

        # Create other tabs
        self._create_calibration_tab()
        self._create_filter_tab()
        self._create_selection_tab()
        self._create_components_tab()
        self._create_lifetime_tab()
        self._create_fret_tab()

        # Connect napari signals when new layer is inserted or removed
        self.viewer.layers.events.inserted.connect(self.reset_layer_choices)
        self.viewer.layers.events.removed.connect(self.reset_layer_choices)

        # Connect callbacks for harmonic and plot settings
        self.plotter_inputs_widget.semi_circle_checkbox.stateChanged.connect(
            self.on_toggle_semi_circle
        )
        self.harmonic_spinbox.valueChanged.connect(self.refresh_current_plot)
        self.plotter_inputs_widget.plot_type_combobox.currentIndexChanged.connect(
            self._on_plot_type_changed
        )
        self.plotter_inputs_widget.colormap_combobox.currentIndexChanged.connect(
            self._on_colormap_changed
        )
        self.plotter_inputs_widget.number_of_bins_spinbox.valueChanged.connect(
            self._on_bins_changed
        )
        self.plotter_inputs_widget.log_scale_checkbox.stateChanged.connect(
            self._on_log_scale_changed
        )
        self.plotter_inputs_widget.white_background_checkbox.stateChanged.connect(
            self.on_white_background_changed
        )

        # Populate plot type combobox
        self.plotter_inputs_widget.plot_type_combobox.addItems(
            ['HISTOGRAM2D', 'SCATTER']
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
        self._current_plot_type = 'HISTOGRAM2D'

        # Connect only the initial active artist
        self._connect_active_artist_signals()
        self._connect_selector_signals()
        self.canvas_widget.show_color_overlay_signal.connect(
            self._enforce_axes_aspect
        )

        # Set the initial plot
        self._redefine_axes_limits()
        self._update_plot_bg_color()

        # Populate labels layer choices
        self.reset_layer_choices()

        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def on_layers_selection_changed(self):
        """Handle changes to layer selection."""
        if getattr(self, "_in_on_layers_selection_changed", False):
            return
        self._in_on_layers_selection_changed = True
        
        try:
            # Get selected layers
            self._labels_layers_with_phasor_features = self.layer_selector.get_selected_layers()
            
            # Update max harmonic across all selected layers
            max_harmonic = 1
            for phasor_layer in self._labels_layers_with_phasor_features:
                if phasor_layer.features is not None:
                    layer_max_harmonic = phasor_layer.features["harmonic"].max()
                    max_harmonic = max(max_harmonic, layer_max_harmonic)
            
            self.harmonic_spinbox.setMaximum(max_harmonic)
            
            # Update other tabs
            if hasattr(self, 'filter_tab'):
                self.filter_tab.on_labels_layer_with_phasor_features_changed()

            # Update calibration button state when layer changes
            if hasattr(self, 'calibration_tab'):
                self.calibration_tab._update_button_state()

            # Sync frequency inputs from first selected layer
            self._sync_frequency_inputs_from_metadata()

            self.plot()

        finally:
            self._in_on_layers_selection_changed = False

    def _on_tab_changed(self, index):
        """Handle tab change events to show/hide tab-specific lines."""
        # Get the current tab widget
        current_tab = self.tab_widget.widget(index)

        # Hide all tab-specific artists first
        self._hide_all_tab_artists()

        # Show artists for the current tab
        self._show_tab_artists(current_tab)

        # Refresh canvas
        self.canvas_widget.figure.canvas.draw_idle()

    def _hide_all_tab_artists(self):
        """Hide all tab-specific artists."""
        # Hide components tab artists
        if hasattr(self, 'components_tab'):
            self._set_components_visibility(False)

        # Hide other tabs' artists (add similar methods for other tabs)
        if hasattr(self, 'fret_tab'):
            self._set_fret_visibility(False)

    def _show_tab_artists(self, current_tab):
        """Show artists for the specified tab."""
        if current_tab == getattr(self, 'components_tab', None):
            self._set_components_visibility(True)
        elif current_tab == getattr(self, 'fret_tab', None):
            self._set_fret_visibility(True)

    def _set_components_visibility(self, visible):
        """Set visibility of components tab artists."""
        if hasattr(self, 'components_tab'):
            self.components_tab.set_artists_visible(visible)

    def _set_fret_visibility(self, visible):
        """Set visibility of FRET tab artists."""
        if hasattr(self, 'fret_tab'):
            self.fret_tab.set_artists_visible(visible)

    def _on_plot_type_changed(self):
        """Callback for plot type change."""
        new_plot_type = (
            self.plotter_inputs_widget.plot_type_combobox.currentText()
        )

        # Store the old plot type before it gets updated
        old_plot_type = getattr(self, '_current_plot_type', None)

        if new_plot_type != old_plot_type:
            self._current_plot_type = new_plot_type
            self._connect_active_artist_signals()
            self.switch_plot_type(new_plot_type)

    def _on_colormap_changed(self):
        """Callback for colormap change."""
        if self.plot_type == 'HISTOGRAM2D':
            self.refresh_current_plot()

    def _on_bins_changed(self):
        """Callback for bins change."""
        self.refresh_current_plot()

    def _on_log_scale_changed(self):
        """Callback for log scale change."""
        if self.plot_type == 'HISTOGRAM2D':
            self.refresh_current_plot()

    def on_white_background_changed(self):
        """Callback function when the white background checkbox is toggled."""
        self.set_axes_labels()
        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)
        else:
            self._update_polar_plot(self.canvas_widget.axes, visible=True)
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
        self.calibration_tab.calibration_widget.frequency_input.editingFinished.connect(
            lambda: self._broadcast_frequency_value_across_tabs(
                self.calibration_tab.calibration_widget.frequency_input.text()
            )
        )

    def _create_filter_tab(self):
        """Create the Filtering and Thresholding tab."""
        self.filter_tab = FilterWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.filter_tab, "Filter/Threshold")

    def _create_selection_tab(self):
        """Create the Cursor selection tab."""
        self.selection_tab = SelectionWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.selection_tab, "Selection")

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
        self.lifetime_tab.frequency_input.editingFinished.connect(
            lambda: self._broadcast_frequency_value_across_tabs(
                self.lifetime_tab.frequency_input.text()
            )
        )

    def _create_fret_tab(self):
        """Create the FRET tab."""
        self.fret_tab = FretWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.fret_tab, "FRET")
        self.fret_tab.frequency_input.editingFinished.connect(
            lambda: self._broadcast_frequency_value_across_tabs(
                self.fret_tab.frequency_input.text()
            )
        )
        self.harmonic_spinbox.valueChanged.connect(
            self.fret_tab._on_harmonic_changed
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
                "Harmonic value should be greater than 0. Setting to 1."
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
        self._enforce_axes_aspect()

    def on_toggle_semi_circle(self, state):
        """Callback function when the semi circle checkbox is toggled.

        This function updates the `toggle_semi_circle` attribute with the
        checked status of the checkbox. And it displays either the universal
        semi-circle or the full polar plot in the canvas widget.

        """
        self.toggle_semi_circle = state

    def _update_polar_plot(self, ax, visible=True, alpha=0.5):
        """Generate the polar plot in the canvas widget."""
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
        """Generate FLIM universal semi-circle plot."""
        line_color = 'black' if self.white_background else 'white'

        if len(self.semi_circle_plot_artist_list) > 0:
            for artist in self.semi_circle_plot_artist_list:
                artist.remove()
            self.semi_circle_plot_artist_list.clear()

        if visible:
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
        """Add lifetime ticks to the semicircle plot based on frequency."""
        frequency = self._get_frequency_from_layer()
        if frequency is None:
            return

        effective_frequency = frequency * self.harmonic

        tick_color = 'black' if self.white_background else 'darkgray'

        # Generate lifetime values using powers of 2
        lifetimes = [0.0]

        # Add powers of 2 that result in S coordinates >= 0.18 (visible on semicircle)
        for t in range(-8, 32):
            lifetime_val = 2**t
            try:
                g_pos, s_pos = phasor_from_lifetime(
                    effective_frequency, lifetime_val
                )
                if s_pos >= 0.18:
                    lifetimes.append(lifetime_val)
            except:
                continue

        for i, lifetime in enumerate(lifetimes):
            if lifetime == 0:
                g_pos, s_pos = 1.0, 0.0
            else:
                g_pos, s_pos = phasor_from_lifetime(
                    effective_frequency, lifetime
                )

            center_x, center_y = 0.5, 0.0
            dx = g_pos - center_x
            dy = s_pos - center_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
            else:
                dx_norm = 1.0
                dy_norm = 0.0

            tick_length = 0.03
            tick_start_x = g_pos
            tick_start_y = s_pos
            tick_end_x = g_pos + tick_length * dx_norm
            tick_end_y = s_pos + tick_length * dy_norm

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

            if lifetime == 0:
                label_text = "0"
            else:
                label_text = f"{lifetime:g}"

            label_offset = 0.08
            label_x = g_pos + label_offset * dx_norm
            label_y = s_pos + label_offset * dy_norm

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
        """Get frequency from the first selected layer's metadata."""
        selected_layers = self.layer_selector.get_selected_layer_names()
        if not selected_layers:
            return None
            
        layer_name = selected_layers[0]
        if layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            if "settings" in layer.metadata:
                settings = layer.metadata["settings"]
                if "frequency" in settings:
                    return settings["frequency"]

        return None

    def _sync_frequency_inputs_from_metadata(self):
        """Sync the frequency widget input fields with metadata."""
        frequency = self._get_frequency_from_layer()
        if frequency is None:
            return
        self._broadcast_frequency_value_across_tabs(str(frequency))

    def _broadcast_frequency_value_across_tabs(self, value):
        """
        Broadcast the frequency value to all relevant input fields and update semicircle.
        """
        try:
            freq_val = float(value)
            # Update frequency in all selected layers
            selected_layer_names = self.layer_selector.get_selected_layer_names()
            for layer_name in selected_layer_names:
                if layer_name in self.viewer.layers:
                    layer = self.viewer.layers[layer_name]
                    update_frequency_in_metadata(layer, freq_val)
        except Exception:
            pass

        # Update frequency input widgets in tabs
        if hasattr(self, 'calibration_tab'):
            self.calibration_tab.calibration_widget.frequency_input.setText(value)
        if hasattr(self, 'lifetime_tab'):
            self.lifetime_tab.frequency_input.setText(value)
        if hasattr(self, 'fret_tab'):
            self.fret_tab.frequency_input.setText(value)
        if hasattr(self, 'components_tab'):
            self.components_tab._update_lifetime_inputs_visibility()

        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)
            self.canvas_widget.figure.canvas.draw_idle()

    # Add property for backward compatibility
    @property
    def image_layer_with_phasor_features_combobox(self):
        """Get a mock combobox-like object for backward compatibility."""
        class MockComboBox:
            def __init__(self, layer_selector):
                self.layer_selector = layer_selector
                # Create mock signals that forward to the layer selector's signal
                self.currentTextChanged = layer_selector.layers_changed
                self.currentIndexChanged = layer_selector.layers_changed
            
            def currentText(self):
                selected_names = self.layer_selector.get_selected_layer_names()
                return selected_names[0] if selected_names else ""
            
            def setCurrentText(self, text):
                # For compatibility, this could select/deselect layers
                pass
                
            def blockSignals(self, block):
                # Mock method for compatibility
                pass
                
            def clear(self):
                # Mock method for compatibility
                pass
                
            def addItems(self, items):
                # Mock method for compatibility
                pass
                
            def findText(self, text):
                # Mock method for compatibility
                selected_names = self.layer_selector.get_selected_layer_names()
                return 0 if text in selected_names else -1
                
            def setCurrentIndex(self, index):
                # Mock method for compatibility
                pass
        
        return MockComboBox(self.layer_selector)

    def _redefine_axes_limits(self, ensure_full_circle_displayed=True):
        """
        Redefine axes limits based on the data plotted in the canvas widget.

        Parameters
        ----------
        ensure_full_circle_displayed : bool, optional
            Whether to ensure the full circle is displayed in the canvas widget.
            By default True.
        """
        if self.toggle_semi_circle:
            circle_plot_limits = [0, 1, 0, 0.6]  # xmin, xmax, ymin, ymax
        else:
            circle_plot_limits = [-1, 1, -1, 1]  # xmin, xmax, ymin, ymax
        if self.canvas_widget.artists['HISTOGRAM2D'].histogram is not None:
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
        if not ensure_full_circle_displayed:
            circle_plot_limits = plotted_data_limits

        x_range = np.amax(
            [plotted_data_limits[1], circle_plot_limits[1]]
        ) - np.amin([plotted_data_limits[0], circle_plot_limits[0]])
        y_range = np.amax(
            [plotted_data_limits[3], circle_plot_limits[3]]
        ) - np.amin([plotted_data_limits[2], circle_plot_limits[2]])
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
            The color to set the background, by default None.
            If None, the background will be set based on the white_background
            checkbox state.
        """
        if color is None:
            if self.white_background:
                color = "white"
            else:
                color = "none"  # Transparent background

        if color == "none":
            self.canvas_widget.axes.set_facecolor('none')
            self.canvas_widget.figure.patch.set_facecolor('none')
        else:
            self.canvas_widget.axes.set_facecolor(color)
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
                "Number of bins should be greater than 1. Setting to 10."
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

    def _enforce_axes_aspect(self):
        """Ensure the axes aspect is set to 'box' after artist redraws."""
        self._redefine_axes_limits()
        self.canvas_widget.active_artist.ax.set_aspect(1, adjustable='box')
        self.canvas_widget.figure.canvas.draw_idle()

    def _connect_selector_signals(self):
        """Connect selection applied signal from all selectors to enforce axes aspect."""
        for selector in self.canvas_widget.selectors.values():
            selector.selection_applied_signal.connect(
                self._enforce_axes_aspect
            )

    def _connect_active_artist_signals(self):
        """Connect signals for the currently active artist only."""
        # Disconnect all existing connections first
        self._disconnect_all_artist_signals()

        # Connect only the active artist
        if self.plot_type == 'HISTOGRAM2D':
            self.canvas_widget.artists[
                'HISTOGRAM2D'
            ].color_indices_changed_signal.connect(
                self.selection_tab.manual_selection_changed
            )
        elif self.plot_type == 'SCATTER':
            self.canvas_widget.artists[
                'SCATTER'
            ].color_indices_changed_signal.connect(
                self.selection_tab.manual_selection_changed
            )

    def _disconnect_all_artist_signals(self):
        """Disconnect all artist signals to prevent conflicts."""
        try:
            self.canvas_widget.artists[
                'SCATTER'
            ].color_indices_changed_signal.disconnect(
                self.selection_tab.manual_selection_changed
            )
        except (TypeError, AttributeError):
            # Signal wasn't connected, ignore
            pass

        try:
            self.canvas_widget.artists[
                'HISTOGRAM2D'
            ].color_indices_changed_signal.disconnect(
                self.selection_tab.manual_selection_changed
            )
        except (TypeError, AttributeError):
            # Signal wasn't connected, ignore
            pass

    def reset_layer_choices(self):
        """Reset layer choices in the layer selector."""
        # The layer selector widget handles its own updates automatically
        # This method is kept for backward compatibility
        pass

    def on_labels_layer_with_phasor_features_changed(self):
        """Handle changes to labels layers (kept for backward compatibility)."""
        # This functionality is now handled by on_layers_selection_changed
        self.on_layers_selection_changed()

    def get_features(self):
        """Get the combined G and S features from all selected layers for the selected harmonic.

        Returns
        -------
        x_data : np.ndarray
            The combined G feature data from all layers.
        y_data : np.ndarray
            The combined S feature data from all layers.
        selection_data : np.ndarray
            The combined selection data from all layers.
        """
        if not self._labels_layers_with_phasor_features:
            return None

        all_x_data = []
        all_y_data = []
        all_selection_data = []

        for phasor_layer in self._labels_layers_with_phasor_features:
            if phasor_layer.features is None:
                continue

            table = phasor_layer.features
            x_data = table['G'][table['harmonic'] == self.harmonic].values
            y_data = table['S'][table['harmonic'] == self.harmonic].values
            
            # Remove NaN values
            mask = np.isnan(x_data) | np.isnan(y_data)
            x_data = x_data[~mask]
            y_data = y_data[~mask]
            
            if len(x_data) > 0 and len(y_data) > 0:
                all_x_data.append(x_data)
                all_y_data.append(y_data)
                
                # Handle selection data
                if (
                    self.selection_tab.selection_id is not None
                    and self.selection_tab.selection_id != ""
                    and self.selection_tab.selection_id in table.columns
                ):
                    selection_data = table[self.selection_tab.selection_id][
                        table['harmonic'] == self.harmonic
                    ].values
                    selection_data = selection_data[~mask]
                    all_selection_data.append(selection_data)
                else:
                    # Add zeros for no selection
                    all_selection_data.append(np.zeros_like(x_data))

        if not all_x_data:
            return None

        # Combine all data
        combined_x = np.concatenate(all_x_data)
        combined_y = np.concatenate(all_y_data)
        
        if all_selection_data and any(len(sel) > 0 for sel in all_selection_data):
            combined_selection = np.concatenate(all_selection_data)
        else:
            combined_selection = None

        return combined_x, combined_y, combined_selection

    def set_axes_labels(self):
        """Set the axes labels in the canvas widget."""
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

        self.canvas_widget.artists['SCATTER'].ax.tick_params(colors=text_color)
        self.canvas_widget.artists['HISTOGRAM2D'].ax.tick_params(
            colors=text_color
        )

        for spine in self.canvas_widget.artists['SCATTER'].ax.spines.values():
            spine.set_color(text_color)
        for spine in self.canvas_widget.artists[
            'HISTOGRAM2D'
        ].ax.spines.values():
            spine.set_color(text_color)

    def _update_scatter_plot(self, x_data, y_data, selection_id_data=None):
        """Update the scatter plot with new data."""
        if len(x_data) == 0 or len(y_data) == 0:
            return

        plot_data = np.column_stack((x_data, y_data))
        self.canvas_widget.artists['SCATTER'].data = plot_data

        if selection_id_data is not None:
            self.canvas_widget.artists['SCATTER'].color_indices = (
                selection_id_data
            )
        else:
            self.canvas_widget.artists['SCATTER'].color_indices = 0

    def _update_histogram_plot(self, x_data, y_data, selection_id_data=None):
        """Update the histogram plot with new data."""
        plot_data = np.column_stack((x_data, y_data))

        # Configure histogram artist properties
        self.canvas_widget.artists['HISTOGRAM2D'].data = plot_data
        self.canvas_widget.artists['HISTOGRAM2D'].cmin = 1
        self.canvas_widget.artists['HISTOGRAM2D'].bins = self.histogram_bins

        # Set colormap
        selected_histogram_colormap = colormaps.ALL_COLORMAPS[
            self.histogram_colormap
        ]
        selected_histogram_colormap = LinearSegmentedColormap.from_list(
            self.histogram_colormap,
            selected_histogram_colormap.colors,
        )
        self.canvas_widget.artists['HISTOGRAM2D'].histogram_colormap = (
            selected_histogram_colormap
        )

        # Set normalization method
        if self.canvas_widget.artists['HISTOGRAM2D'].histogram is not None:
            if self.histogram_log_scale:
                self.canvas_widget.artists[
                    'HISTOGRAM2D'
                ].histogram_color_normalization_method = "log"
            else:
                self.canvas_widget.artists[
                    'HISTOGRAM2D'
                ].histogram_color_normalization_method = "linear"

        # Apply selection data if available
        if selection_id_data is not None:
            self.canvas_widget.artists['HISTOGRAM2D'].color_indices = (
                selection_id_data
            )
        else:
            self.canvas_widget.artists['HISTOGRAM2D'].color_indices = 0

        # Update colorbar for histogram
        self._update_colorbar(selected_histogram_colormap)

    def _update_colorbar(self, colormap):
        """Update or create colorbar for histogram plot."""
        self._remove_colorbar()
        # Create new colorbar for histogram
        if self.plot_type == 'HISTOGRAM2D':
            self.cax = self.canvas_widget.artists['HISTOGRAM2D'].ax.inset_axes(
                [1.05, 0, 0.05, 1]
            )
            self.colorbar = Colorbar(
                ax=self.cax,
                cmap=colormap,
                norm=self.canvas_widget.artists[
                    'HISTOGRAM2D'
                ]._get_normalization(
                    self.canvas_widget.artists['HISTOGRAM2D'].histogram[0],
                    is_overlay=False,
                ),
            )
            self.set_colorbar_style(color="white")

    def _remove_colorbar(self):
        """Remove colorbar if it exists."""
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

    def _update_plot_elements(self):
        """Update common plot elements like semicircle, axes, etc."""
        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)

        self._enforce_axes_aspect()
        self._update_plot_bg_color()

    def plot(self, x_data=None, y_data=None, selection_id_data=None):
        """Plot the selected phasor features efficiently."""
        if self._labels_layer_with_phasor_features is None:
            return

        # Check if we're already updating to prevent infinite recursion
        if getattr(self, '_updating_plot', False):
            return

        # Set flag to prevent recursion from canvas events
        self._updating_plot = True

        if x_data is None or y_data is None:
            features = self.get_features()
            if features is None:
                return
            x_data, y_data, selection_id_data = features

        if len(x_data) == 0 or len(y_data) == 0:
            return

        self._set_active_artist_and_plot(
            self.plot_type, x_data, y_data, selection_id_data
        )

        self._updating_plot = False

    def refresh_current_plot(self):
        """Refresh the current plot with existing data."""
        self.plot()

    def switch_plot_type(self, new_plot_type):
        """Switch between plot types efficiently."""
        self._connect_active_artist_signals()

        features = self.get_features()
        if features is not None:
            x_data, y_data, selection_id_data = features
            self._set_active_artist_and_plot(
                new_plot_type, x_data, y_data, selection_id_data
            )

    def _set_active_artist_and_plot(
        self, plot_type, x_data, y_data, selection_id_data=None
    ):
        """Set the active artist and update only the relevant plot."""
        if len(x_data) == 0 or len(y_data) == 0:
            return
        if plot_type != self.plot_type:
            self.plotter_inputs_widget.plot_type_combobox.blockSignals(True)
            self.plotter_inputs_widget.plot_type_combobox.setCurrentText(
                plot_type
            )
            self.plotter_inputs_widget.plot_type_combobox.blockSignals(False)
            self._connect_active_artist_signals()

        if plot_type == 'HISTOGRAM2D':
            self._update_histogram_plot(x_data, y_data, selection_id_data)
        elif plot_type == 'SCATTER':
            self._remove_colorbar()
            self._update_scatter_plot(x_data, y_data, selection_id_data)

        current_active = getattr(self.canvas_widget, 'active_artist', None)

        if (
            current_active != plot_type
            and plot_type in self.canvas_widget.artists
        ):
            self.canvas_widget.active_artist = plot_type
        elif plot_type not in self.canvas_widget.artists:
            return

        self._update_plot_elements()

    def set_colorbar_style(self, color="white"):
        """Set the colorbar style in the canvas widget."""
        self.colorbar.ax.yaxis.set_tick_params(color=color)
        self.colorbar.outline.set_edgecolor(color)
        if isinstance(self.colorbar.norm, LogNorm):
            self.colorbar.ax.set_ylabel("Log10(Count)", color=color)
        else:
            self.colorbar.ax.set_ylabel("Count", color=color)
        ticks = self.colorbar.ax.get_yticks()
        self.colorbar.ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
        tick_labels = self.colorbar.ax.get_yticklabels()
        for tick_label in tick_labels:
            tick_label.set_color(color)
