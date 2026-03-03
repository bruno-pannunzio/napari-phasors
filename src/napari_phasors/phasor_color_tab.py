"""Tab widget to color the phasor histogram by phase or modulation."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from napari.utils import colormaps
from phasorpy.phasor import phasor_to_polar
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.stats import binned_statistic_2d


class PhasorColorWidget(QWidget):
    """Widget to color the phasor histogram overlay by phase or modulation.

    Colors the histogram via a direct matplotlib imshow overlay (bypassing
    the biaplotter color_indices / selection mechanism entirely). Also creates
    a new napari Image layer for each selected image layer coloured by the
    same quantity.
    """

    def __init__(self, viewer, parent=None):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

        self._color_mode = "None"
        self._phase_colormap_name = "hsv"
        self._modulation_colormap_name = "viridis"

        # Matplotlib artists managed directly
        self._overlay_imshow = None
        # Napari layers created by this widget (for event linking)
        self._linked_layers: list = []

        layout = QVBoxLayout(self)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Color by:"))
        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(["None", "Phase", "Modulation"])
        mode_layout.addWidget(self.mode_combobox, 1)
        layout.addLayout(mode_layout)

        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combobox = QComboBox()
        self.colormap_combobox.addItems(list(colormaps.ALL_COLORMAPS.keys()))
        self.colormap_combobox.setCurrentText("hsv")
        cmap_layout.addWidget(self.colormap_combobox, 1)
        layout.addLayout(cmap_layout)

        btn_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply Coloring")
        btn_layout.addWidget(self.apply_button)
        self.clear_button = QPushButton("Clear")
        btn_layout.addWidget(self.clear_button)
        layout.addLayout(btn_layout)
        layout.addStretch()

        self.mode_combobox.currentTextChanged.connect(self._on_mode_changed)
        self.apply_button.clicked.connect(self.apply_coloring)
        self.clear_button.clicked.connect(self.clear_coloring)


    def apply_coloring(self):
        """Compute phase/modulation, overlay on histogram and create image layers."""
        pw = self.parent_widget
        if pw is None or not pw.has_phasor_data():
            return
        mode = self.mode_combobox.currentText()
        if mode == "None":
            self.clear_coloring()
            return
        self._color_mode = mode
        self._apply_histogram_coloring(pw, mode)
        self._create_image_layers(pw, mode)

    def clear_coloring(self):
        """Remove overlay coloring, restore histogram density, and redraw."""
        self._color_mode = "None"
        self._remove_overlay()
        pw = self.parent_widget
        if pw is not None:
            self._set_histogram_density_visible(pw, True)
            pw.canvas_widget.figure.canvas.draw_idle()

    def reapply_if_active(self):
        """Re-apply overlay coloring after a plot refresh (if active)."""
        if self._color_mode != "None":
            self._apply_histogram_coloring(self.parent_widget, self._color_mode)


    def _apply_histogram_coloring(self, pw, mode):
        hist_artist = pw.canvas_widget.artists.get("HISTOGRAM2D")
        if hist_artist is None:
            return
        histogram = hist_artist.histogram
        if histogram is None:
            return
        H, x_edges, y_edges = histogram

        features = pw.get_merged_features()
        if features is None:
            return
        g_flat, s_flat = features

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            phase, modulation = phasor_to_polar(g_flat, s_flat)

        values = phase if mode == "Phase" else modulation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stat, _, _, _ = binned_statistic_2d(
                g_flat,
                s_flat,
                values,
                statistic="median",
                bins=[x_edges, y_edges],
            )

        # Mask empty bins
        stat[H == 0] = np.nan
        # binned_statistic_2d uses (x, y) → transpose to (row=y, col=x) for imshow
        stat_display = stat.T

        cmap = self._napari_cmap_to_mpl(self.colormap_combobox.currentText())
        # Use overlay vmin/vmax from the first linked layer's contrast limits if available
        vmin, vmax = self._get_clim_from_linked_layers(mode)
        if vmin is None or vmax is None:
            vmin = float(np.nanmin(stat_display))
            vmax = float(np.nanmax(stat_display))

        ax = pw.canvas_widget.axes
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        # Remove old overlay
        if self._overlay_imshow is not None:
            try:
                self._overlay_imshow.remove()
            except (ValueError, AttributeError):
                pass
            self._overlay_imshow = None

        # Hide the histogram density so only the colored stat is visible
        self._set_histogram_density_visible(pw, False)

        # Draw colored stat as opaque imshow; do NOT pass aspect= so matplotlib
        # does not override the axes aspect, then explicitly restore it afterward
        self._overlay_imshow = ax.imshow(
            stat_display,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=3,
            alpha=1.0,
            aspect="auto",
        )
        # Restore locked 1:1 aspect ratio that imshow overrides
        ax.set_aspect(1, adjustable="box")

        pw.canvas_widget.figure.canvas.draw_idle()

    @staticmethod
    def _set_histogram_density_visible(pw, visible: bool):
        """Show or hide the biaplotter histogram density image."""
        hist_artist = pw.canvas_widget.artists.get("HISTOGRAM2D")
        if hist_artist is None:
            return
        img = hist_artist._mpl_artists.get("histogram_image")
        if img is not None:
            img.set_visible(visible)

    def _remove_overlay(self):
        if self._overlay_imshow is not None:
            try:
                self._overlay_imshow.remove()
            except (ValueError, AttributeError):
                pass
            self._overlay_imshow = None

    def _get_clim_from_linked_layers(self, mode):
        """Return (vmin, vmax) from the first linked layer matching the mode, or (None, None)."""
        label = "Phase" if mode == "Phase" else "Modulation"
        for layer in self._linked_layers:
            if label in layer.name:
                try:
                    vmin, vmax = layer.contrast_limits
                    return float(vmin), float(vmax)
                except Exception:
                    pass
        return None, None

    def _create_image_layers(self, pw, mode):
        # Disconnect events from old linked layers
        self._disconnect_linked_layers()
        cmap_name = self.colormap_combobox.currentText()
        for layer in pw.get_selected_layers():
            g_array = layer.metadata.get("G")
            s_array = layer.metadata.get("S")
            harmonics_array = layer.metadata.get("harmonics")
            if g_array is None or s_array is None:
                continue

            if harmonics_array is not None:
                harmonics_array = np.atleast_1d(harmonics_array)
                try:
                    h_idx = int(
                        np.where(harmonics_array == pw.harmonic)[0][0]
                    )
                except (IndexError, ValueError):
                    continue
            else:
                h_idx = 0

            g = g_array[h_idx] if g_array.ndim == 3 else g_array
            s = s_array[h_idx] if s_array.ndim == 3 else s_array

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                phase, modulation = phasor_to_polar(g, s)

            data = phase if mode == "Phase" else modulation
            label = "Phase" if mode == "Phase" else "Modulation"
            layer_name = f"{layer.name} — {label} (H{pw.harmonic})"

            # Remove existing layer with same name
            try:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
            except (KeyError, ValueError):
                pass

            new_layer = self.viewer.add_image(
                data,
                name=layer_name,
                scale=layer.scale,
                colormap=cmap_name,
            )

            valid = np.isfinite(data)
            if valid.any():
                vmin = float(np.nanmin(data[valid]))
                vmax = float(np.nanmax(data[valid]))
                if vmin < vmax:
                    new_layer.contrast_limits = (vmin, vmax)

            # Link layer events → update phasor overlay
            new_layer.events.colormap.connect(self._on_linked_layer_colormap_changed)
            new_layer.events.contrast_limits.connect(self._on_linked_layer_clim_changed)
            self._linked_layers.append(new_layer)

    def _on_linked_layer_colormap_changed(self, event):
        """When a linked layer's colormap changes, sync the combobox and reapply."""
        cmap_name = event.source.colormap.name
        # Block combobox signal to avoid cascading
        self.colormap_combobox.blockSignals(True)
        idx = self.colormap_combobox.findText(cmap_name)
        if idx >= 0:
            self.colormap_combobox.setCurrentIndex(idx)
        self.colormap_combobox.blockSignals(False)
        if self._color_mode != "None" and self.parent_widget is not None:
            self._apply_histogram_coloring(self.parent_widget, self._color_mode)

    def _on_linked_layer_clim_changed(self, event):
        """When a linked layer's contrast limits change, update the overlay clim."""
        if self._overlay_imshow is None or self._color_mode == "None":
            return
        vmin, vmax = event.source.contrast_limits
        self._overlay_imshow.set_clim(float(vmin), float(vmax))
        pw = self.parent_widget
        if pw is not None:
            pw.canvas_widget.figure.canvas.draw_idle()

    def _disconnect_linked_layers(self):
        """Disconnect event listeners from previously linked layers."""
        for layer in self._linked_layers:
            try:
                layer.events.colormap.disconnect(self._on_linked_layer_colormap_changed)
            except Exception:
                pass
            try:
                layer.events.contrast_limits.disconnect(self._on_linked_layer_clim_changed)
            except Exception:
                pass
        self._linked_layers = []

    def _on_image_layer_changed(self):
        """Handle image layer change — reset color mode, clear overlay, restore density."""
        self._disconnect_linked_layers()
        self._color_mode = "None"
        self._remove_overlay()
        pw = self.parent_widget
        if pw is not None:
            self._set_histogram_density_visible(pw, True)

    def _on_mode_changed(self, text):
        if text == "Phase":
            self.colormap_combobox.setCurrentText(self._phase_colormap_name)
        elif text == "Modulation":
            self.colormap_combobox.setCurrentText(
                self._modulation_colormap_name
            )

    @staticmethod
    def _napari_cmap_to_mpl(name: str):
        if name in colormaps.ALL_COLORMAPS:
            napari_cmap = colormaps.ALL_COLORMAPS[name]
            return LinearSegmentedColormap.from_list(name, napari_cmap.colors)
        try:
            return plt.get_cmap(name)
        except ValueError:
            return plt.get_cmap("viridis")

    def _restore_settings_from_metadata(self):
        pass
