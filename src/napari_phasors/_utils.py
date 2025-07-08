"""
This module contains utility functions used by other modules.

"""

import warnings

import numpy as np
from napari.layers import Image
from phasorpy.phasor import phasor_filter_median, phasor_threshold
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPen
from qtpy.QtWidgets import QWidget


class RangeSlider(QWidget):
    """A custom range slider with two handles on a single track."""

    rangeChanged = Signal(int, int)

    def __init__(self, min_val=0, max_val=100, initial_min=0, initial_max=100):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.min_handle = initial_min
        self.max_handle = initial_max
        self.handle_radius = 18
        self.track_height = 7
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)
        self.dragging = None  # Track which handle is being dragged

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate dimensions
        width = self.width() - 2 * self.handle_radius
        height = self.height()
        track_y = (height - self.track_height) // 2

        # Draw track background
        track_rect = QRectF(
            self.handle_radius, track_y, width, self.track_height
        )
        painter.fillRect(track_rect, QColor(64, 64, 64))

        # Calculate handle positions
        range_size = self.max_val - self.min_val
        if range_size > 0:
            min_pos = (self.min_handle - self.min_val) / range_size * width
            max_pos = (self.max_handle - self.min_val) / range_size * width
        else:
            min_pos = max_pos = 0

        # Draw active track (between handles)
        if max_pos > min_pos:
            active_rect = QRectF(
                self.handle_radius + min_pos,
                track_y,
                max_pos - min_pos,
                self.track_height,
            )
            painter.fillRect(active_rect, QColor(120, 120, 120))

        # Draw handles
        painter.setBrush(QBrush(QColor(155, 155, 155)))  # Full grey color
        painter.setPen(QPen(QColor(155, 155, 155)))

        # Min handle (left)
        painter.drawEllipse(
            int(min_pos + self.handle_radius - self.handle_radius // 2),
            int(height // 2 - self.handle_radius // 2),
            self.handle_radius,
            self.handle_radius,
        )

        # Max handle (right)
        painter.drawEllipse(
            int(max_pos + self.handle_radius - self.handle_radius // 2),
            int(height // 2 - self.handle_radius // 2),
            self.handle_radius,
            self.handle_radius,
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._handle_mouse_press(event)

    def mouseMoveEvent(self, event):
        if self.dragging is not None:
            self._handle_mouse_move(event)

    def mouseReleaseEvent(self, event):
        self.dragging = None

    def _handle_mouse_press(self, event):
        width = self.width() - 2 * self.handle_radius
        range_size = self.max_val - self.min_val

        if range_size == 0:
            return

        min_pos = (
            self.min_handle - self.min_val
        ) / range_size * width + self.handle_radius
        max_pos = (
            self.max_handle - self.min_val
        ) / range_size * width + self.handle_radius

        # Check which handle is closer
        dist_to_min = abs(event.x() - min_pos)
        dist_to_max = abs(event.x() - max_pos)

        if dist_to_min <= self.handle_radius and dist_to_min <= dist_to_max:
            self.dragging = 'min'
        elif dist_to_max <= self.handle_radius:
            self.dragging = 'max'

    def _handle_mouse_move(self, event):
        width = self.width() - 2 * self.handle_radius
        range_size = self.max_val - self.min_val

        if range_size == 0:
            return

        # Calculate new value based on mouse position
        pos_fraction = max(0, min(1, (event.x() - self.handle_radius) / width))
        new_value = int(self.min_val + pos_fraction * range_size)

        if self.dragging == 'min':
            self.min_handle = min(new_value, self.max_handle - 1)
        elif self.dragging == 'max':
            self.max_handle = max(new_value, self.min_handle + 1)

        self.update()
        self.rangeChanged.emit(self.min_handle, self.max_handle)

    def set_range(self, min_val, max_val):
        """Set the range of the slider."""
        self.min_val = min_val
        self.max_val = max_val
        self.update()

    def get_values(self):
        """Get current min and max values."""
        return self.min_handle, self.max_handle

    def set_values(self, min_val, max_val):
        """Set current min and max values."""
        self.min_handle = min_val
        self.max_handle = max_val
        self.update()


def apply_filter_and_threshold(
    layer: Image,
    /,
    *,
    threshold: float = 0,
    size: int = 3,
    repeat: int = 1,
):
    """Apply filter to an image layer.

    Parameters
    ----------
    layer : napari.layers.Image
        Napari image layer with phasor features.
    threshold : float
        Threshold value for the mean value to be applied to G and S.
    method : str
        Filter method. Options are 'median'.
    size : int
        Size of the filter.
    repeat : int
        Number of times to apply the filter.

    """
    mean = layer.metadata['original_mean'].copy()
    phasor_features = layer.metadata['phasor_features_labels_layer'].features
    harmonics = np.unique(phasor_features['harmonic'])
    real, imag = (
        phasor_features['G_original'].copy(),
        phasor_features['S_original'].copy(),
    )
    real = np.reshape(real, (len(harmonics),) + mean.shape)
    imag = np.reshape(imag, (len(harmonics),) + mean.shape)
    if repeat > 0:
        mean, real, imag = phasor_filter_median(
            mean,
            real,
            imag,
            repeat=repeat,
            size=size,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, real, imag = phasor_threshold(mean, real, imag, threshold)
        (
            layer.metadata['phasor_features_labels_layer'].features['G'],
            layer.metadata['phasor_features_labels_layer'].features['S'],
        ) = (real.flatten(), imag.flatten())
    layer.data = mean
    # Update the settings dictionary of the layer
    if "settings" not in layer.metadata:
        layer.metadata["settings"] = {}
    layer.metadata["settings"]["filter"] = {
        "size": size,
        "repeat": repeat,
    }
    layer.metadata["settings"]["threshold"] = threshold
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
    parent_widget: QWidget,
    frequency: float,
):
    """Update the frequency in the layer metadata."""
    layer = parent_widget.viewer.layers[
        parent_widget.image_layer_with_phasor_features_combobox.currentText()
    ]
    if "settings" not in layer.metadata.keys():
        layer.metadata["settings"] = {}
    layer.metadata["settings"]["frequency"] = frequency
    parent_widget.calibration_tab.calibration_widget.frequency_input.setText(
        str(frequency)
    )
    parent_widget.lifetime_tab.frequency_input.setText(str(frequency))
    parent_widget.fret_tab.frequency_input.setText(str(frequency))
