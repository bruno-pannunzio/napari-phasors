from pathlib import Path

import numpy as np
from napari.layers import Labels
from napari.utils import DirectLabelColormap, notifications
from qtpy import uic
from qtpy.QtWidgets import QVBoxLayout, QWidget
from skimage.util import map_array

from ._utils import colormap_to_dict

#: The columns in the phasor features table that should not be used as selection id.
DATA_COLUMNS = ["label", "G_original", "S_original", "G", "S", "harmonic"]


class CursorWidget(QWidget):
    """Widget to perform cursor selection."""

    def __init__(self, viewer, parent=None):
        super().__init__()
        self.parent_widget = parent
        self.viewer = viewer

        self.cursors_input_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/cursors_tab.ui",
            self.cursors_input_widget,
        )
        self.cursors_input_widget.phasor_selection_id_combobox.addItem(
            "MANUAL SELECTION #1"
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.cursors_input_widget)

        self.selection_id = "MANUAL SELECTION #1"
        self._phasors_selected_layer = None

        self.cursors_input_widget.phasor_selection_id_combobox.currentIndexChanged.connect(
            self.on_selection_id_changed
        )

    @property
    def selection_id(self):
        """Gets or sets the selection id from the phasor selection id combobox.

        Value should not be one of these: ['label', 'Average Image', 'G', 'S', 'harmonic'].

        Returns
        -------
        str
            The selection id. Returns `None` if no selection id is available.
        """
        if self.cursors_input_widget.phasor_selection_id_combobox.count() == 0:
            return None
        else:
            return (
                self.cursors_input_widget.phasor_selection_id_combobox.currentText()
            )

    @selection_id.setter
    def selection_id(self, new_selection_id: str):
        """Sets the selection id from the phasor selection id combobox."""
        if new_selection_id in DATA_COLUMNS:
            notifications.WarningNotification(
                f"{new_selection_id} is not a valid selection column. It must not be one of {DATA_COLUMNS}."
            )
            return
        else:
            if new_selection_id not in [
                self.cursors_input_widget.phasor_selection_id_combobox.itemText(
                    i
                )
                for i in range(
                    self.cursors_input_widget.phasor_selection_id_combobox.count()
                )
            ]:
                self.cursors_input_widget.phasor_selection_id_combobox.addItem(
                    new_selection_id
                )
            self.cursors_input_widget.phasor_selection_id_combobox.setCurrentText(
                new_selection_id
            )
            self.add_selection_id_to_features(new_selection_id)

    def add_selection_id_to_features(self, new_selection_id: str):
        """Add a new selection id to the features table in the labels layer with phasor features.

        Parameters
        ----------
        new_selection_id : str
            The new selection id to add to the features table.
        """
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if new_selection_id in DATA_COLUMNS:
            notifications.WarningNotification(
                f"{new_selection_id} is not a valid selection column. It must not be one of {DATA_COLUMNS}."
            )
            return
        # If column_name is not in features, add it with zeros
        if (
            new_selection_id
            not in self.parent_widget._labels_layer_with_phasor_features.features.columns
        ):
            self.parent_widget._labels_layer_with_phasor_features.features[
                new_selection_id
            ] = np.zeros_like(
                self.parent_widget._labels_layer_with_phasor_features.features[
                    "label"
                ].values
            )

    def on_selection_id_changed(self):
        """Callback function when the phasor selection id combobox is changed.

        This function updates the `selection_id` attribute with the selected text from the combobox.
        """
        self.selection_id = (
            self.cursors_input_widget.phasor_selection_id_combobox.currentText()
        )

    def manual_selection_changed(self, manual_selection):
        """Update the manual selection in the labels layer with phasor features.

        This method serves as a Slot for the `color_indices_changed_signal` emitted by the canvas widget.
        It should receive the `color_indices` array from the active artist in the canvas widget.
        It also updates/creates the phasors selected layer by calling the `create_phasors_selected_layer` method.

        Parameters
        ----------
        manual_selection : np.ndarray
            The manual selection array.
        """
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if self.selection_id is None or self.selection_id == "":
            return
        column = self.selection_id
        # Update the manual selection in the labels layer with phasor features for each harmonic
        self.parent_widget._labels_layer_with_phasor_features.features[
            column
        ] = 0
        # Filter rows where 'G' is not NaN
        valid_rows = (
            ~self.parent_widget._labels_layer_with_phasor_features.features[
                "G"
            ].isna()
            & ~self.parent_widget._labels_layer_with_phasor_features.features[
                "S"
            ].isna()
        )
        num_valid_rows = valid_rows.sum()
        # Tile the manual_selection array to match the number of valid rows
        tiled_manual_selection = np.tile(
            manual_selection, (num_valid_rows // len(manual_selection)) + 1
        )[:num_valid_rows]
        self.parent_widget._labels_layer_with_phasor_features.features.loc[
            valid_rows, column
        ] = tiled_manual_selection
        self.create_phasors_selected_layer()
        # TODO: FIX BUG CHANGING SELECTION ID DOES NOT UPDATE PHASOR LAYER

    def create_phasors_selected_layer(self):
        """Create or update the phasors selected layer."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        input_array = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.data
        )
        input_array_values = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.features[
                "label"
            ].values
        )
        # If no selection id is provided, set all pixels to 0
        if self.selection_id is None or self.selection_id == "":
            phasors_layer_data = np.zeros_like(
                self.parent_widget._labels_layer_with_phasor_features.features[
                    "label"
                ].values
            )
        else:
            phasors_layer_data = np.asarray(
                self.parent_widget._labels_layer_with_phasor_features.features[
                    self.selection_id
                ].values
            )

        mapped_data = map_array(
            input_array, input_array_values, phasors_layer_data
        )
        color_dict = colormap_to_dict(
            self.parent_widget._colormap,
            self.parent_widget._colormap.N,
            exclude_first=True,
        )
        # Build output phasors Labels layer
        phasors_selected_layer = Labels(
            mapped_data,
            name="Phasors Selected",
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap=DirectLabelColormap(
                color_dict=color_dict, name="cat10_mod"
            ),
        )
        if self._phasors_selected_layer is None:
            self._phasors_selected_layer = self.viewer.add_layer(
                phasors_selected_layer
            )
        else:
            self._phasors_selected_layer.data = mapped_data
            self._phasors_selected_layer.scale = (
                self.parent_widget._labels_layer_with_phasor_features.scale
            )
