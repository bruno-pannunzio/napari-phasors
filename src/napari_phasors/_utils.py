"""
This module contains utility functions used by other modules.

"""

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.colors import LinearSegmentedColormap
from napari.layers import Image
from phasorpy.filter import (
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_threshold,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

if TYPE_CHECKING:
    import napari


def validate_harmonics_for_wavelet(harmonics):
    """Validate that harmonics have their double or half correspondent.

    Parameters
    ----------
    harmonics : array-like
        Array of harmonic values

    Returns
    -------
    bool
        True if harmonics are valid for wavelet filtering, False otherwise
    """
    harmonics = np.atleast_1d(harmonics)

    for harmonic in harmonics:
        # Check if double or half exists
        has_double = (harmonic * 2) in harmonics
        has_half = (harmonic / 2) in harmonics

        if not (has_double or has_half):
            return False

    return True


def _extract_phasor_arrays_from_layer(
    layer: Image, harmonics: np.ndarray = None
):
    """Extract phasor arrays from layer metadata.

    Parameters
    ----------
    layer : napari.layers.Image
        Napari image layer with phasor features.
    harmonics : np.ndarray, optional
        Harmonic values. If None, will be extracted from layer.

    Returns
    -------
    tuple
        (mean, real, imag, harmonics) arrays
    """
    mean = layer.metadata['original_mean'].copy()

    if harmonics is None:
        harmonics = layer.metadata.get('harmonics')

    harmonics = np.atleast_1d(harmonics)

    real = layer.metadata['G_original'].copy()
    imag = layer.metadata['S_original'].copy()

    # Apply mask if present in metadata
    if 'mask' in layer.metadata:
        mask = layer.metadata['mask']
        # Apply mask: set values to NaN where mask <= 0
        mask_invalid = mask <= 0
        mean = np.where(mask_invalid, np.nan, mean)
        for h in range(len(harmonics)):
            real[h] = np.where(mask_invalid, np.nan, real[h])
            imag[h] = np.where(mask_invalid, np.nan, imag[h])

    return mean, real, imag, harmonics


def _apply_filter_and_threshold_to_phasor_arrays(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    harmonics: np.ndarray,
    *,
    threshold: float = None,
    threshold_upper: float = None,
    filter_method: str = None,
    size: int = None,
    repeat: int = None,
    sigma: float = None,
    levels: int = None,
):
    """Apply filter and threshold to phasor arrays.

    Parameters
    ----------
    mean : np.ndarray
        Mean intensity array.
    real : np.ndarray
        Real part of phasor (G).
    imag : np.ndarray
        Imaginary part of phasor (S).
    harmonics : np.ndarray
        Harmonic values.
    threshold : float, optional
        Lower threshold value for the mean value to be applied to G and S.
        If None, no lower threshold is applied.
    threshold_upper : float, optional
        Upper threshold value for the mean value to be applied to G and S.
        If None, no upper threshold is applied.
    filter_method : str, optional
        Filter method. Options are 'median' or 'wavelet'.
        If None, no filter is applied.
    size : int, optional
        Size of the median filter. Only used if filter_method is 'median'.
    repeat : int, optional
        Number of times to apply the median filter. Only used if filter_method is 'median'.
    sigma : float, optional
        Sigma parameter for wavelet filter. Only used if filter_method is 'wavelet'.
    levels : int, optional
        Number of levels for wavelet filter. Only used if filter_method is 'wavelet'.

    Returns
    -------
    tuple
        (mean, real, imag) filtered and thresholded arrays
    """
    if filter_method == "median" and repeat is not None and repeat > 0:
        mean, real, imag = phasor_filter_median(
            mean,
            real,
            imag,
            repeat=repeat,
            size=size if size is not None else 3,
        )
    elif filter_method == "wavelet" and validate_harmonics_for_wavelet(
        harmonics
    ):
        mean, real, imag = phasor_filter_pawflim(
            mean,
            real,
            imag,
            sigma=sigma if sigma is not None else 1.0,
            levels=levels if levels is not None else 3,
            harmonic=harmonics,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, real, imag = phasor_threshold(
            mean, real, imag, mean_min=threshold, mean_max=threshold_upper
        )

    return mean, real, imag


def apply_filter_and_threshold(
    layer: Image,
    /,
    *,
    threshold: float = None,
    threshold_upper: float = None,
    threshold_method: str = None,
    filter_method: str = None,
    size: int = None,
    repeat: int = None,
    sigma: float = None,
    levels: int = None,
    harmonics: np.ndarray = None,
):
    """Apply filter to an image layer.

    Parameters
    ----------
    layer : napari.layers.Image
        Napari image layer with phasor features.
    threshold : float, optional
        Lower threshold value for the mean value to be applied to G and S.
        If None, no lower threshold is applied.
    threshold_upper : float, optional
        Upper threshold value for the mean value to be applied to G and S.
        If None, no upper threshold is applied.
    threshold_method : str, optional
        Threshold method used. If None, no threshold method is saved.
    filter_method : str, optional
        Filter method. Options are 'median' or 'wavelet'.
        If None, no filter is applied.
    size : int, optional
        Size of the median filter. Only used if filter_method is 'median'.
    repeat : int, optional
        Number of times to apply the median filter. Only used if filter_method is 'median'.
    sigma : float, optional
        Sigma parameter for wavelet filter. Only used if filter_method is 'wavelet'.
    levels : int, optional
        Number of levels for wavelet filter. Only used if filter_method is 'wavelet'.
    harmonics : np.ndarray, optional
        Harmonic values for wavelet filter. If None, will be extracted from layer.

    """
    mean, real, imag, harmonics = _extract_phasor_arrays_from_layer(
        layer, harmonics
    )

    mean, real, imag = _apply_filter_and_threshold_to_phasor_arrays(
        mean,
        real,
        imag,
        harmonics,
        threshold=threshold,
        threshold_upper=threshold_upper,
        filter_method=filter_method,
        size=size,
        repeat=repeat,
        sigma=sigma,
        levels=levels,
    )

    layer.metadata['G'] = real
    layer.metadata['S'] = imag
    layer.data = mean

    if "settings" not in layer.metadata:
        layer.metadata["settings"] = {}

    # Only save filter settings if a filter was actually applied
    if filter_method is not None:
        layer.metadata["settings"]["filter"] = {}
        layer.metadata["settings"]["filter"]["method"] = filter_method

        if filter_method == "median":
            if size is not None:
                layer.metadata["settings"]["filter"]["size"] = size
            if repeat is not None:
                layer.metadata["settings"]["filter"]["repeat"] = repeat
        elif filter_method == "wavelet":
            if sigma is not None:
                layer.metadata["settings"]["filter"]["sigma"] = sigma
            if levels is not None:
                layer.metadata["settings"]["filter"]["levels"] = levels

    layer.metadata["settings"]["threshold"] = threshold
    layer.metadata["settings"]["threshold_upper"] = threshold_upper
    layer.metadata["settings"]["threshold_method"] = threshold_method
    layer.refresh()

    return


def colormap_to_dict(colormap, num_colors=10, exclude_first=True):
    """
    Converts a matplotlib colormap into a dictionary of RGBA colors.

    Parameters
    ----------
    colormap : matplotlib.colors.Colormap
        The colormap to convert.
    num_colors : int, optional
        The number of colors in the colormap, by default 10.
    exclude_first : bool, optional
        Whether to exclude the first color in the colormap, by default True.

    Returns
    -------
    color_dict: dict
        A dictionary with keys as positive integers and values as RGBA colors.
    """
    color_dict = {}
    start = 0
    if exclude_first:
        start = 1
    for i in range(start, num_colors + start):
        pos = i / (num_colors - 1)
        color = colormap(pos)
        color_dict[i + 1 - start] = color
    color_dict[None] = (0, 0, 0, 0)
    return color_dict


def update_frequency_in_metadata(
    image_layer: "napari.layers.Image",
    frequency: float,
):
    """Update the frequency in the layer metadata."""
    if "settings" not in image_layer.metadata.keys():
        image_layer.metadata["settings"] = {}
    image_layer.metadata["settings"]["frequency"] = frequency


class HistogramSettingsDialog(QDialog):
    """Dialog for histogram visualization settings.

    Provides controls for:
    - Display mode: Merged / Separate layers / Grouped.
    - Toggling SD shading (for Merged and Grouped modes).
    - Group assignment per layer (for Grouped mode).

    Parameters
    ----------
    display_mode : str
        Initial display mode.
    show_sd : bool
        Initial state of the *Show standard deviation* checkbox.
    layer_labels : list of str, optional
        Layer names for group assignment.
    group_assignments : dict, optional
        ``{label: group_int}`` initial group assignments.
    parent : QWidget, optional
        Parent widget.
    """

    DISPLAY_MODES = ("Merged", "Separate layers", "Grouped")

    def __init__(
        self,
        display_mode: str = "Merged",
        show_sd: bool = False,
        layer_labels: list = None,
        group_assignments: dict = None,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Histogram Settings")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        # --- Display mode ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Display mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(self.DISPLAY_MODES))
        self.mode_combo.setCurrentText(display_mode)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # --- Show SD ---
        self.sd_checkbox = QCheckBox("Show standard deviation")
        self.sd_checkbox.setChecked(show_sd)
        layout.addWidget(self.sd_checkbox)

        # --- Group assignments (only for Grouped mode) ---
        self._group_section = QWidget()
        group_layout = QVBoxLayout(self._group_section)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.addWidget(QLabel("Assign layers to groups:"))

        self._group_combos = {}
        if layer_labels:
            for label in layer_labels:
                row = QHBoxLayout()
                name_label = QLabel(label)
                name_label.setMaximumWidth(200)
                row.addWidget(name_label)
                combo = QComboBox()
                combo.addItems([str(i) for i in range(1, 7)])
                if group_assignments and label in group_assignments:
                    combo.setCurrentText(str(group_assignments[label]))
                row.addWidget(combo)
                group_layout.addLayout(row)
                self._group_combos[label] = combo

        layout.addWidget(self._group_section)

        # Show / hide sections based on mode
        self._update_ui_for_mode(display_mode)
        self.mode_combo.currentTextChanged.connect(self._update_ui_for_mode)

        # --- OK / Cancel ---
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _update_ui_for_mode(self, mode: str) -> None:
        """Show/hide controls depending on the selected mode."""
        is_grouped = mode == "Grouped"
        is_separate = mode == "Separate layers"
        self._group_section.setVisible(is_grouped)
        # SD only meaningful for Merged / Grouped
        self.sd_checkbox.setEnabled(not is_separate)

    def get_group_assignments(self) -> dict:
        """Return ``{label: group_int}`` from the dialog."""
        return {
            label: int(combo.currentText())
            for label, combo in self._group_combos.items()
        }


class HistogramWidget(QWidget):
    """Reusable 1D histogram widget with colormap-synced colored bars.

    This widget wraps a Matplotlib figure that renders a 1D histogram
    with bars colored according to a given colormap and contrast limits.
    It is designed to be embedded in any tab that needs to display a
    histogram of scalar data (e.g. lifetime, concentration, FRET efficiency).

    When *range_slider_enabled* is ``True`` the widget also displays a
    range slider together with min / max line-edits that allow the user
    to clip the displayed / stored data.  The ``rangeChanged`` signal is
    emitted whenever the effective range changes (min, max as floats).

    Parameters
    ----------
    xlabel : str, optional
        Label for the x-axis, by default ``"Value"``.
    ylabel : str, optional
        Label for the y-axis, by default ``"Pixel count"``.
    bins : int, optional
        Number of histogram bins, by default 300.
    default_colormap_name : str, optional
        Name of the Matplotlib colormap to use as fallback when no explicit
        colormap colors are provided, by default ``"plasma"``.
    canvas_height : int, optional
        Fixed pixel height of the canvas, by default 150.
    range_slider_enabled : bool, optional
        If ``True``, show a range slider with min / max edits above the
        histogram plot, by default ``False``.
    range_label_prefix : str, optional
        Prefix for the range label, e.g. ``"Lifetime range (ns)"``.
        Only used when *range_slider_enabled* is ``True``.
    range_factor : int, optional
        Multiplicative factor to convert float range values to integer
        slider positions, by default 1000.
    parent : QWidget, optional
        Parent widget.
    """

    # Emitted as (min_float, max_float) whenever the range changes.
    rangeChanged = Signal(float, float)

    def __init__(
        self,
        xlabel: str = "Value",
        ylabel: str = "Pixel count",
        bins: int = 150,
        default_colormap_name: str = "plasma",
        canvas_height: int = 150,
        range_slider_enabled: bool = False,
        range_label_prefix: str = "Range",
        range_factor: int = 1000,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.bins = bins
        self.default_colormap_name = default_colormap_name

        # Histogram state
        self.counts = None
        self.bin_edges = None
        self.bin_centers = None

        # Multi-layer state
        self._datasets = {}  # {label: valid_1d_array}
        self._counts_per_dataset = {}  # {label: counts on common bins}

        # Colormap state (set externally)
        self.colormap_colors = None  # Nx4 array of RGBA colors
        self.contrast_limits = None  # [vmin, vmax]

        # Raw pooled data (for central tendency computation)
        self._raw_valid_data = None

        # Display settings
        self._display_mode = "Merged"  # "Merged", "Separate layers", "Grouped"
        self._show_sd = False
        self._group_assignments = {}  # {label: group_int}

        # Range slider state
        self._range_slider_enabled = range_slider_enabled
        self._range_label_prefix = range_label_prefix
        self.range_factor = range_factor
        self._slider_being_dragged = False

        # Build UI
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Optional range slider section ---
        if self._range_slider_enabled:
            self.range_label = QLabel(
                f"{self._range_label_prefix}: 0.0 - 100.0"
            )
            layout.addWidget(self.range_label)

            edit_layout = QHBoxLayout()
            self.range_min_edit = QLineEdit("0.0")
            self.range_max_edit = QLineEdit("100.0")
            self.range_min_edit.setValidator(QDoubleValidator())
            self.range_max_edit.setValidator(QDoubleValidator())
            edit_layout.addWidget(QLabel("Min:"))
            edit_layout.addWidget(self.range_min_edit)
            edit_layout.addWidget(QLabel("Max:"))
            edit_layout.addWidget(self.range_max_edit)
            layout.addLayout(edit_layout)

            self.range_slider = QRangeSlider(Qt.Orientation.Horizontal)
            self.range_slider.setRange(0, 100)
            self.range_slider.setValue((0, 100))
            self.range_slider.setBarMovesAllHandles(False)

            self.range_slider.valueChanged.connect(
                self._on_range_label_update
            )
            self.range_slider.sliderPressed.connect(self._on_slider_pressed)
            self.range_slider.sliderReleased.connect(self._on_slider_released)
            layout.addWidget(self.range_slider)

            self.range_min_edit.editingFinished.connect(
                self._on_range_min_edit
            )
            self.range_max_edit.editingFinished.connect(
                self._on_range_max_edit
            )

        # --- Matplotlib canvas ---
        self.fig, self.ax = plt.subplots(
            figsize=(8, 4), constrained_layout=True
        )
        self._style_axes()

        canvas = FigureCanvas(self.fig)
        canvas.setFixedHeight(canvas_height)
        canvas.setSizePolicy(
            canvas.sizePolicy().Expanding, canvas.sizePolicy().Fixed
        )
        layout.addWidget(canvas)

        # --- Bottom controls: settings button + central tendency combo ---
        bottom_layout = QHBoxLayout()
        self._settings_button = QPushButton("Histogram Settings…")
        self._settings_button.setMaximumWidth(180)
        self._settings_button.clicked.connect(self._open_settings_dialog)
        bottom_layout.addWidget(self._settings_button)

        bottom_layout.addWidget(QLabel("Show line:"))
        self._central_tendency_combo = QComboBox()
        self._central_tendency_combo.addItems(
            ["None", "Mean", "Median", "Center of mass"]
        )
        self._central_tendency_combo.currentTextChanged.connect(
            self._on_central_tendency_changed
        )
        bottom_layout.addWidget(self._central_tendency_combo)
        bottom_layout.addStretch()
        layout.addLayout(bottom_layout)

        # Start hidden
        self.hide()

    # ------------------------------------------------------------------
    # Range-slider helpers (only active when range_slider_enabled=True)
    # ------------------------------------------------------------------

    def set_range(
        self, min_val: float, max_val: float, *, slider_max: float = None
    ) -> None:
        """Programmatically set the range slider position.

        Parameters
        ----------
        min_val : float
            Minimum value.
        max_val : float
            Maximum value.
        slider_max : float, optional
            If given, also update the slider's maximum to
            ``int(slider_max * range_factor)``.
        """
        if not self._range_slider_enabled:
            return
        if slider_max is not None:
            self.range_slider.setRange(
                0, int(slider_max * self.range_factor)
            )
        min_s = int(min_val * self.range_factor)
        max_s = int(max_val * self.range_factor)
        self.range_slider.setValue((min_s, max_s))
        self.range_min_edit.setText(f"{min_val:.2f}")
        self.range_max_edit.setText(f"{max_val:.2f}")
        self.range_label.setText(
            f"{self._range_label_prefix}: {min_val:.2f} - {max_val:.2f}"
        )

    def get_range(self) -> tuple:
        """Return ``(min_float, max_float)`` from the slider."""
        if not self._range_slider_enabled:
            return (0.0, 0.0)
        lo, hi = self.range_slider.value()
        return lo / self.range_factor, hi / self.range_factor

    # internal range-slider callbacks

    def _on_range_label_update(self, value):
        """Update label + edits while dragging (no heavy work)."""
        lo, hi = value
        lo_f = lo / self.range_factor
        hi_f = hi / self.range_factor
        self.range_label.setText(
            f"{self._range_label_prefix}: {lo_f:.2f} - {hi_f:.2f}"
        )
        self.range_min_edit.setText(f"{lo_f:.2f}")
        self.range_max_edit.setText(f"{hi_f:.2f}")

    def _on_slider_pressed(self):
        self._slider_being_dragged = True

    def _on_slider_released(self):
        self._slider_being_dragged = False
        lo, hi = self.range_slider.value()
        self.rangeChanged.emit(
            lo / self.range_factor, hi / self.range_factor
        )

    def _on_range_min_edit(self):
        try:
            lo = float(self.range_min_edit.text())
            hi = float(self.range_max_edit.text())
        except ValueError:
            return
        if lo >= hi:
            hi = lo + 0.01
        lo_s = int(lo * self.range_factor)
        hi_s = int(hi * self.range_factor)
        self.range_slider.setValue((lo_s, hi_s))
        self.rangeChanged.emit(lo, hi)

    def _on_range_max_edit(self):
        try:
            lo = float(self.range_min_edit.text())
            hi = float(self.range_max_edit.text())
        except ValueError:
            return
        if hi <= lo:
            lo = hi - 0.01 if hi > 0.01 else 0.0
        lo_s = int(lo * self.range_factor)
        hi_s = int(hi * self.range_factor)
        self.range_slider.setValue((lo_s, hi_s))
        self.rangeChanged.emit(lo, hi)

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------

    def _open_settings_dialog(self):
        """Open the histogram settings dialog."""
        layer_labels = list(self._datasets.keys()) if self._datasets else None
        dlg = HistogramSettingsDialog(
            display_mode=self._display_mode,
            show_sd=self._show_sd,
            layer_labels=layer_labels,
            group_assignments=self._group_assignments,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._display_mode = dlg.mode_combo.currentText()
            self._show_sd = dlg.sd_checkbox.isChecked()
            if dlg._group_combos:
                self._group_assignments = dlg.get_group_assignments()
            if self.counts is not None:
                self._render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_data(self, data: np.ndarray) -> None:
        """Compute histogram from *data* and render.

        Values that are NaN, non-positive, or non-finite are excluded
        before computing the histogram.  This is the single-dataset
        entry point; multi-layer features are disabled.

        Parameters
        ----------
        data : np.ndarray
            Scalar data array (any shape – will be flattened internally).
        """
        flat = np.asarray(data).ravel()
        valid = flat[~np.isnan(flat) & (flat > 0) & np.isfinite(flat)]

        if len(valid) == 0:
            self.ax.clear()
            self.ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=self.ax.transAxes,
                ha="center",
            )
            self.fig.canvas.draw_idle()
            self.show()
            return

        # Clear multi-layer state (single-dataset mode)
        self._datasets = {}
        self._counts_per_dataset = {}
        self._raw_valid_data = valid

        self.counts, self.bin_edges = np.histogram(valid, bins=self.bins)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self._render()
        self.show()

    def update_multi_data(self, datasets: dict) -> None:
        """Compute histograms from multiple datasets and render.

        Each dataset (one per layer) is stored individually so that
        *Separate layers*, *Grouped*, and *Merged + SD* display modes
        can operate on per-layer counts.

        Parameters
        ----------
        datasets : dict
            ``{label: np.ndarray}`` mapping layer names to their scalar
            data arrays.  Arrays will be flattened and filtered.
        """
        self._datasets = {}
        for label, data in datasets.items():
            flat = np.asarray(data).ravel()
            valid = flat[~np.isnan(flat) & (flat > 0) & np.isfinite(flat)]
            if len(valid) > 0:
                self._datasets[label] = valid

        if not self._datasets:
            self.ax.clear()
            self.ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=self.ax.transAxes,
                ha="center",
            )
            self.fig.canvas.draw_idle()
            self.show()
            return

        # Compute common bins from pooled data
        all_valid = np.concatenate(list(self._datasets.values()))
        self._raw_valid_data = all_valid
        self.counts, self.bin_edges = np.histogram(
            all_valid, bins=self.bins
        )
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        # Per-dataset histograms on the same bins
        self._counts_per_dataset = {}
        for label, valid in self._datasets.items():
            counts, _ = np.histogram(valid, bins=self.bin_edges)
            self._counts_per_dataset[label] = counts

        self._render()
        self.show()

    def update_colormap(
        self,
        colormap_colors: np.ndarray = None,
        contrast_limits: list = None,
    ) -> None:
        """Update the colormap / contrast limits and re-render.

        Parameters
        ----------
        colormap_colors : np.ndarray, optional
            Nx4 RGBA array that defines the colormap.
        contrast_limits : list, optional
            ``[vmin, vmax]`` for the normalisation.
        """
        self.colormap_colors = colormap_colors
        self.contrast_limits = contrast_limits
        if self.counts is not None:
            self._render()

    def clear(self) -> None:
        """Clear the histogram and hide the widget."""
        self.counts = None
        self.bin_edges = None
        self.bin_centers = None
        self._datasets = {}
        self._counts_per_dataset = {}
        self._raw_valid_data = None
        self.ax.clear()
        self.fig.canvas.draw_idle()
        self.hide()

    @property
    def display_mode(self) -> str:
        """Current display mode."""
        return self._display_mode

    @display_mode.setter
    def display_mode(self, value: str):
        self._display_mode = value
        if self.counts is not None:
            self._render()

    @property
    def show_sd(self) -> bool:
        """Whether SD shading is enabled."""
        return self._show_sd

    @show_sd.setter
    def show_sd(self, value: bool):
        self._show_sd = value
        if self.counts is not None:
            self._render()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _style_axes(self) -> None:
        """Apply consistent styling to the axes and figure."""
        self.ax.patch.set_alpha(0)
        self.fig.patch.set_alpha(0)
        for spine in self.ax.spines.values():
            spine.set_color("grey")
            spine.set_linewidth(1)
        self.ax.set_ylabel(self.ylabel, fontsize=6, color="grey")
        self.ax.set_xlabel(self.xlabel, fontsize=6, color="grey")
        for which in ("major", "minor"):
            self.ax.tick_params(
                axis="x", which=which, labelsize=7, colors="grey"
            )
            self.ax.tick_params(
                axis="y", which=which, labelsize=7, colors="grey"
            )

    def _get_cmap_and_norm(self):
        """Return (cmap, norm) from current colormap state."""
        if self.colormap_colors is None or self.contrast_limits is None:
            cmap = plt.cm.get_cmap(self.default_colormap_name)
            norm = plt.Normalize(
                vmin=(
                    np.min(self.bin_centers)
                    if len(self.bin_centers) > 0
                    else 0
                ),
                vmax=(
                    np.max(self.bin_centers)
                    if len(self.bin_centers) > 0
                    else 1
                ),
            )
        else:
            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", self.colormap_colors
            )
            norm = plt.Normalize(
                vmin=self.contrast_limits[0],
                vmax=self.contrast_limits[1],
            )
        return cmap, norm

    # ------------------------------------------------------------------
    # Smoothing helper
    # ------------------------------------------------------------------

    def _smooth_curve(self, y, sigma=2, upsample=5):
        """Return (x_fine, y_fine) with Gaussian-smoothed, upsampled data.

        Parameters
        ----------
        y : np.ndarray
            Y-values corresponding to ``self.bin_centers``.
        sigma : float
            Gaussian smoothing sigma (in bins).
        upsample : int
            Upsampling factor for the output grid.

        Returns
        -------
        x_fine : np.ndarray
        y_fine : np.ndarray
        """
        y_smooth = gaussian_filter1d(y.astype(float), sigma=sigma)
        x_fine = np.linspace(
            self.bin_centers[0],
            self.bin_centers[-1],
            len(self.bin_centers) * upsample,
        )
        y_fine = np.interp(x_fine, self.bin_centers, y_smooth)
        return x_fine, y_fine

    # ------------------------------------------------------------------
    # Central-tendency vertical lines
    # ------------------------------------------------------------------

    def _on_central_tendency_changed(self, _text: str) -> None:
        """Re-render when the central tendency selection changes."""
        if self.counts is not None:
            self._render()

    @staticmethod
    def _compute_central_tendency(
        data: np.ndarray,
        method: str,
        bin_centers: np.ndarray = None,
        bin_edges: np.ndarray = None,
    ):
        """Return a scalar central-tendency value.

        Parameters
        ----------
        data : np.ndarray
            1-D array of valid values.
        method : str
            ``"Mean"``, ``"Median"``, or ``"Center of mass"``.
        bin_centers, bin_edges : np.ndarray, optional
            Needed only for ``"Center of mass"``.
        """
        if data is None or len(data) == 0:
            return None
        if method == "Mean":
            return float(np.mean(data))
        if method == "Median":
            return float(np.median(data))
        if method == "Center of mass":
            if bin_centers is None or bin_edges is None:
                return float(np.mean(data))
            counts, _ = np.histogram(data, bins=bin_edges)
            total = counts.sum()
            if total == 0:
                return None
            return float(np.sum(bin_centers * counts) / total)
        return None

    def _draw_central_tendency_lines(self) -> None:
        """Draw vertical lines at the selected central-tendency statistic."""
        choice = self._central_tendency_combo.currentText()
        if choice == "None":
            return

        n_datasets = len(self._counts_per_dataset)

        if n_datasets > 1 and self._display_mode == "Separate layers":
            outline_colors = plt.cm.tab10.colors
            for idx, (label, valid) in enumerate(self._datasets.items()):
                color = outline_colors[idx % len(outline_colors)]
                val = self._compute_central_tendency(
                    valid, choice, self.bin_centers, self.bin_edges
                )
                if val is not None:
                    self.ax.axvline(
                        val, color=color, ls="--", lw=1, alpha=0.85
                    )
        elif n_datasets > 1 and self._display_mode == "Grouped":
            outline_colors = plt.cm.tab10.colors
            groups: dict[int, list] = {}
            for label, valid in self._datasets.items():
                g = self._group_assignments.get(label, 1)
                groups.setdefault(g, []).append(valid)
            for gidx, (gid, data_list) in enumerate(sorted(groups.items())):
                color = outline_colors[gidx % len(outline_colors)]
                pooled = np.concatenate(data_list)
                val = self._compute_central_tendency(
                    pooled, choice, self.bin_centers, self.bin_edges
                )
                if val is not None:
                    self.ax.axvline(
                        val, color=color, ls="--", lw=1, alpha=0.85
                    )
        else:
            # Merged or single-dataset
            if self._raw_valid_data is not None:
                val = self._compute_central_tendency(
                    self._raw_valid_data,
                    choice,
                    self.bin_centers,
                    self.bin_edges,
                )
                if val is not None:
                    self.ax.axvline(
                        val, color="white", ls="--", lw=1, alpha=0.85
                    )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Re-draw the histogram using the active display mode."""
        self.ax.clear()

        n_datasets = len(self._counts_per_dataset)

        if n_datasets > 1:
            if self._display_mode == "Separate layers":
                self._render_separate()
            elif self._display_mode == "Grouped":
                self._render_grouped()
            else:
                self._render_merged()
        else:
            # Single dataset or update_data() path – colormap bars
            self._render_bars()

        self._draw_central_tendency_lines()
        self._style_axes()
        self.fig.canvas.draw_idle()

    def _render_bars(self) -> None:
        """Render the standard colormap-colored bar histogram."""
        cmap, norm = self._get_cmap_and_norm()
        # invisible line keeps axes auto-scaled correctly
        self.ax.plot(
            self.bin_centers, self.counts, color="none", alpha=0
        )
        for count, bin_start, bin_end in zip(
            self.counts, self.bin_edges[:-1], self.bin_edges[1:]
        ):
            bin_center = (bin_start + bin_end) / 2
            color = cmap(norm(bin_center))
            self.ax.fill_between(
                [bin_start, bin_end], 0, count, color=color, alpha=0.7
            )

    def _render_merged(self) -> None:
        """Render merged histogram with optional SD shading.

        Uses Gaussian smoothing + upsampling for seamless color
        transitions in both the line and the shaded SD band.
        """
        cmap, norm = self._get_cmap_and_norm()
        n = len(self._counts_per_dataset)

        if self._show_sd and n > 1:
            all_counts = np.array(
                list(self._counts_per_dataset.values()), dtype=float
            )
            mean_counts = np.mean(all_counts, axis=0)
            std_counts = np.std(all_counts, axis=0, ddof=1)
            lower = np.maximum(mean_counts - std_counts, 0)
            upper = mean_counts + std_counts

            # Smooth + upsample for a seamless look
            x_fine, mean_fine = self._smooth_curve(mean_counts)
            _, lower_fine = self._smooth_curve(lower)
            _, upper_fine = self._smooth_curve(upper)

            # Many fine coloured segments → imperceptible transitions
            for i in range(len(x_fine) - 1):
                x0, x1 = x_fine[i], x_fine[i + 1]
                color = cmap(norm(x0))
                self.ax.fill_between(
                    [x0, x1],
                    [lower_fine[i], lower_fine[i + 1]],
                    [upper_fine[i], upper_fine[i + 1]],
                    color=color,
                    alpha=0.35,
                    linewidth=0,
                )
                self.ax.plot(
                    [x0, x1],
                    [mean_fine[i], mean_fine[i + 1]],
                    color=color,
                    linewidth=1.5,
                )
        else:
            self._render_bars()

    def _render_separate(self) -> None:
        """Render each dataset as a smooth outline."""
        outline_colors = plt.cm.tab10.colors
        for idx, (label, counts) in enumerate(
            self._counts_per_dataset.items()
        ):
            color = outline_colors[idx % len(outline_colors)]
            x_fine, y_fine = self._smooth_curve(counts)
            self.ax.plot(
                x_fine, y_fine,
                color=color, linewidth=1.5, label=label,
            )
        if self._counts_per_dataset:
            self.ax.legend(fontsize=5, loc="upper right")

    def _render_grouped(self) -> None:
        """Render grouped histograms with smooth curves and optional SD."""
        outline_colors = plt.cm.tab10.colors

        groups: dict[int, list[tuple[str, np.ndarray]]] = {}
        for label, counts in self._counts_per_dataset.items():
            g = self._group_assignments.get(label, 1)
            groups.setdefault(g, []).append((label, counts))

        for gidx, (group_id, members) in enumerate(sorted(groups.items())):
            color = outline_colors[gidx % len(outline_colors)]
            all_counts = np.array([c for _, c in members], dtype=float)
            mean_counts = np.mean(all_counts, axis=0)

            x_fine, mean_fine = self._smooth_curve(mean_counts)
            group_label = f"Group {group_id}"
            self.ax.plot(
                x_fine, mean_fine,
                color=color, linewidth=1.5, label=group_label,
            )

            if self._show_sd and len(members) > 1:
                std_counts = np.std(all_counts, axis=0, ddof=1)
                lower = np.maximum(mean_counts - std_counts, 0)
                upper = mean_counts + std_counts
                _, lower_fine = self._smooth_curve(lower)
                _, upper_fine = self._smooth_curve(upper)
                self.ax.fill_between(
                    x_fine, lower_fine, upper_fine,
                    color=color, alpha=0.25, linewidth=0,
                )

        if groups:
            self.ax.legend(fontsize=5, loc="upper right")
