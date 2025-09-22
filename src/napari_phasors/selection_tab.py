from pathlib import Path

import numpy as np
from napari.layers import Labels
from napari.utils import DirectLabelColormap, notifications
from qtpy import uic
from qtpy.QtWidgets import (
    QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLineEdit, QLabel, 
    QColorDialog, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QComboBox
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QIcon
from skimage.util import map_array
from phasorpy.cursor import mask_from_elliptic_cursor
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

from ._utils import colormap_to_dict

#: The columns in the phasor features table that should not be used as selection id.
DATA_COLUMNS = ["label", "G_original", "S_original", "G", "S", "harmonic"]


class ColorButton(QPushButton):
    """A button that displays and allows selection of a color."""
    
    def __init__(self, color=QColor(255, 0, 0)):
        super().__init__()
        self.color = color
        self.setMaximumSize(30, 25)
        self.setStyleSheet(f"background-color: {color.name()}")
        self.clicked.connect(self._select_color)
        
    def _select_color(self):
        """Open color dialog to select color."""
        new_color = QColorDialog.getColor(self.color, self)
        if new_color.isValid():
            self.color = new_color
            self.setStyleSheet(f"background-color: {new_color.name()}")
            # Emit a signal that the parent can listen to
            if hasattr(self.parent(), '_on_cursor_table_changed'):
                self.parent()._on_cursor_table_changed()


class DeleteButton(QPushButton):
    """A button for deleting cursor rows."""
    
    def __init__(self, row_index, parent_widget):
        super().__init__()
        self.row_index = row_index
        self.parent_widget = parent_widget
        self.setMaximumSize(25, 25)
        self.setToolTip("Delete cursor")
        
        # Try to use a trash icon, fallback to "×" if not available
        try:
            # Try to create a simple "×" text button
            self.setText("×")
            self.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    color: red;
                    border: 1px solid gray;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #ffcccc;
                }
            """)
        except:
            self.setText("Del")
            
        self.clicked.connect(self._delete_cursor)
        
    def _delete_cursor(self):
        """Delete the cursor associated with this button."""
        self.parent_widget._remove_cursor_by_index(self.row_index)


class SelectionWidget(QWidget):
    """
    Widget for interactive phasor selection using the cursor in napari.

    Provides:
      - A dropdown to manage and select manual or custom selection IDs
      - Elliptical cursor controls for interactive selection using a table interface

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    parent : QWidget, optional
        The parent widget (typically the main PlotterWidget).

    Notes
    -----
    This widget is designed to be used as a tab within the main PlotterWidget.

    """

    def __init__(self, viewer, parent=None):
        """Initialize the SelectionWidget."""
        super().__init__()
        self.parent_widget = parent
        self.viewer = viewer

        # Load the UI from the .ui file
        self.selection_input_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/selection_tab.ui",
            self.selection_input_widget,
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self.selection_input_widget)

        # Add elliptical cursor controls
        self._create_cursor_controls()

        # Add default items to the selection id combobox
        self.selection_input_widget.phasor_selection_id_combobox.addItem(
            "None"
        )
        self.selection_input_widget.phasor_selection_id_combobox.addItem(
            "MANUAL SELECTION #1"
        )

        # Initialize the current selection id to match the default
        self._current_selection_id = "None"
        self.selection_id = "None"
        self._phasors_selected_layer = None

        # Initialize cursor tracking
        self._cursors = []  # List to store cursor patches
        self._selected_cursor_index = None
        self._dragging = False
        self._updating_from_drag = False  # Flag to prevent recursive updates during drag

        # Timer for delayed selection updates
        self._selection_update_timer = QTimer()
        self._selection_update_timer.setSingleShot(True)
        self._selection_update_timer.timeout.connect(self._apply_cursor_selection)

        # Connect to multiple signals to handle both selection and text editing
        self.selection_input_widget.phasor_selection_id_combobox.currentIndexChanged.connect(
            self.on_selection_id_changed
        )
        self.selection_input_widget.phasor_selection_id_combobox.activated.connect(
            self.on_selection_id_changed
        )
        if hasattr(
            self.selection_input_widget.phasor_selection_id_combobox,
            'lineEdit',
        ):
            line_edit = (
                self.selection_input_widget.phasor_selection_id_combobox.lineEdit()
            )
            if line_edit:
                line_edit.editingFinished.connect(self.on_selection_id_changed)

    def _create_cursor_controls(self):
        """Create the elliptical cursor control widgets."""
        cursor_group = QWidget()
        cursor_layout = QVBoxLayout(cursor_group)
        
        # Title
        title_label = QLabel("Select Cursors")
        title_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        cursor_layout.addWidget(title_label)
        
        # Cursor management buttons
        button_layout = QHBoxLayout()
        self.add_cursor_button = QPushButton("Add Cursor")
        self.add_cursor_button.clicked.connect(self._add_cursor)
        button_layout.addWidget(self.add_cursor_button)
        
        self.clear_cursors_button = QPushButton("Clear All")
        self.clear_cursors_button.clicked.connect(self._clear_all_cursors)
        button_layout.addWidget(self.clear_cursors_button)
        
        cursor_layout.addLayout(button_layout)
        
        # Create cursor table
        self.cursor_table = QTableWidget()
        self.cursor_table.setColumnCount(7)  # Added one column for delete button
        self.cursor_table.setHorizontalHeaderLabels([
            "Center X", "Center Y", "Radius Major", "Radius Minor", "Angle (°)", "Color", "Delete"
        ])
        
        # Configure table
        self.cursor_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cursor_table.setAlternatingRowColors(True)
        self.cursor_table.horizontalHeader().setStretchLastSection(False)
        self.cursor_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Make the delete column smaller
        self.cursor_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Fixed)
        self.cursor_table.setColumnWidth(6, 50)
        self.cursor_table.setMinimumHeight(150)
        
        # Connect table signals
        self.cursor_table.cellChanged.connect(self._on_cursor_table_changed)
        self.cursor_table.cellClicked.connect(self._on_cursor_selected)
        
        # Add context menu for removing cursors (keep as backup)
        self.cursor_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.cursor_table.customContextMenuRequested.connect(self._show_context_menu)
        
        cursor_layout.addWidget(self.cursor_table)
        
        self.layout().addWidget(cursor_group)

    def _show_context_menu(self, position):
        """Show context menu for removing cursors."""
        from qtpy.QtWidgets import QMenu, QAction
        
        if self.cursor_table.itemAt(position) is None:
            return
            
        menu = QMenu(self)
        remove_action = QAction("Remove Cursor", self)
        remove_action.triggered.connect(self._remove_selected_cursor)
        menu.addAction(remove_action)
        
        menu.exec_(self.cursor_table.mapToGlobal(position))

    def _remove_selected_cursor(self):
        """Remove the selected cursor from the table and plot."""
        current_row = self.cursor_table.currentRow()
        if current_row < 0:
            return
        self._remove_cursor_by_index(current_row)

    def _remove_cursor_by_index(self, row_index):
        """Remove cursor by index from both table and plot."""
        if row_index < 0 or row_index >= self.cursor_table.rowCount():
            return
            
        # Remove from plot
        if row_index < len(self._cursors):
            self._cursors[row_index].remove()
            self._cursors.pop(row_index)
            
        # Remove from table
        self.cursor_table.removeRow(row_index)
        
        # Update row indices for remaining delete buttons
        self._update_delete_button_indices()
        
        # Update selection
        self._delayed_selection_update()
        
        # Redraw canvas
        if self.parent_widget and self.parent_widget.canvas_widget:
            self.parent_widget.canvas_widget.figure.canvas.draw_idle()

    def _update_delete_button_indices(self):
        """Update the row indices for all delete buttons after a row is removed."""
        for row in range(self.cursor_table.rowCount()):
            delete_button = self.cursor_table.cellWidget(row, 6)
            if isinstance(delete_button, DeleteButton):
                delete_button.row_index = row

    def _add_cursor(self):
        """Add a new elliptical cursor to the plot and table."""
        if self.parent_widget is None or self.parent_widget.canvas_widget is None:
            notifications.WarningNotification("No plot available to add cursor")
            return
            
        # Default cursor parameters
        center_x = 0
        center_y = 0
        radius_major = 0.05
        radius_minor = 0.05
        angle = 0.0
        color = QColor(255, 0, 0)  # Red
        
        # Vary position slightly for multiple cursors
        row_count = self.cursor_table.rowCount()
        if row_count > 0:
            # Cycle through some colors
            colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), 
                     QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)]
            color = colors[row_count % len(colors)]
            
        # Create ellipse patch
        ellipse = Ellipse(
            (center_x, center_y),
            2 * radius_major,  # matplotlib expects width/height, not radius
            2 * radius_minor,
            angle=angle,
            fill=False,
            edgecolor=color.name(),
            linewidth=2,
            picker=True
        )
        
        # Add to plot
        ax = self.parent_widget.canvas_widget.axes
        ax.add_patch(ellipse)
        self._cursors.append(ellipse)
        
        # Add row to table
        row = self.cursor_table.rowCount()
        self.cursor_table.insertRow(row)
        
        # Add items to row
        self.cursor_table.setItem(row, 0, QTableWidgetItem(f"{center_x:.3f}"))
        self.cursor_table.setItem(row, 1, QTableWidgetItem(f"{center_y:.3f}"))
        self.cursor_table.setItem(row, 2, QTableWidgetItem(f"{radius_major:.3f}"))
        self.cursor_table.setItem(row, 3, QTableWidgetItem(f"{radius_minor:.3f}"))
        self.cursor_table.setItem(row, 4, QTableWidgetItem(f"{angle:.1f}"))
        
        # Add color button
        color_button = ColorButton(color)
        self.cursor_table.setCellWidget(row, 5, color_button)
        
        # Add delete button
        delete_button = DeleteButton(row, self)
        self.cursor_table.setCellWidget(row, 6, delete_button)
        
        # Connect mouse events for interaction if not already connected
        self._connect_cursor_events()
        
        # Update selection
        self._delayed_selection_update()
        
        # Redraw canvas
        self.parent_widget.canvas_widget.figure.canvas.draw_idle()

    def _connect_cursor_events(self):
        """Connect mouse events for cursor interaction."""
        canvas = self.parent_widget.canvas_widget.figure.canvas
        
        # Disconnect existing connections to avoid duplicates
        try:
            canvas.mpl_disconnect(self._press_cid)
            canvas.mpl_disconnect(self._motion_cid)
            canvas.mpl_disconnect(self._release_cid)
        except (AttributeError, TypeError):
            pass
            
        self._press_cid = canvas.mpl_connect('button_press_event', self._on_cursor_press)
        self._motion_cid = canvas.mpl_connect('motion_notify_event', self._on_cursor_motion)
        self._release_cid = canvas.mpl_connect('button_release_event', self._on_cursor_release)

    def _on_cursor_press(self, event):
        """Handle mouse press on cursor."""
        if event.inaxes != self.parent_widget.canvas_widget.axes:
            return
            
        for i, cursor in enumerate(self._cursors):
            if cursor.contains(event)[0]:
                self._selected_cursor_index = i
                self._dragging = True
                self._updating_from_drag = True
                # Select the corresponding row in the table
                self.cursor_table.selectRow(i)
                break

    def _on_cursor_motion(self, event):
        """Handle mouse motion for cursor dragging."""
        if not self._dragging or self._selected_cursor_index is None:
            return
            
        if event.inaxes != self.parent_widget.canvas_widget.axes:
            return
            
        # Update cursor position
        cursor = self._cursors[self._selected_cursor_index]
        cursor.center = (event.xdata, event.ydata)
        
        # Update table (temporarily disable change signal to avoid recursion)
        self.cursor_table.blockSignals(True)
        try:
            self.cursor_table.setItem(
                self._selected_cursor_index, 0, 
                QTableWidgetItem(f"{event.xdata:.3f}")
            )
            self.cursor_table.setItem(
                self._selected_cursor_index, 1, 
                QTableWidgetItem(f"{event.ydata:.3f}")
            )
        finally:
            self.cursor_table.blockSignals(False)
        
        # Redraw
        self.parent_widget.canvas_widget.figure.canvas.draw_idle()

    def _on_cursor_release(self, event):
        """Handle mouse release - update selection when dragging ends."""
        if self._dragging:
            self._dragging = False
            self._updating_from_drag = False
            # Update selection after drag is complete
            self._delayed_selection_update()

    def _on_cursor_selected(self, row, column):
        """Handle cursor selection in table."""
        if row < len(self._cursors):
            self._selected_cursor_index = row

    def _on_cursor_table_changed(self):
        """Handle changes to the cursor table."""
        if self._updating_from_drag:
            return  # Don't update during drag operations
        
        # Add additional guard to prevent recursive calls
        if getattr(self, '_updating_table', False):
            return
            
        self._updating_table = True
        
        try:
            # Update cursors based on table data
            for row in range(self.cursor_table.rowCount()):
                if row >= len(self._cursors):
                    continue
                    
                try:
                    center_x = float(self.cursor_table.item(row, 0).text())
                    center_y = float(self.cursor_table.item(row, 1).text())
                    radius_major = float(self.cursor_table.item(row, 2).text())
                    radius_minor = float(self.cursor_table.item(row, 3).text())
                    angle = float(self.cursor_table.item(row, 4).text())
                    
                    # Get color from color button
                    color_button = self.cursor_table.cellWidget(row, 5)
                    if isinstance(color_button, ColorButton):
                        color = color_button.color
                    else:
                        color = QColor(255, 0, 0)  # Default red
                    
                    # Update cursor patch
                    cursor = self._cursors[row]
                    cursor.center = (center_x, center_y)
                    cursor.width = 2 * radius_major
                    cursor.height = 2 * radius_minor
                    cursor.angle = angle
                    cursor.set_edgecolor(color.name())
                    
                except (ValueError, AttributeError):
                    continue  # Skip invalid entries
                    
            # Update selection
            self._delayed_selection_update()
            
            # Redraw canvas
            if self.parent_widget and self.parent_widget.canvas_widget:
                self.parent_widget.canvas_widget.figure.canvas.draw_idle()
                
        finally:
            self._updating_table = False

    def _clear_all_cursors(self):
        """Remove all cursors from the plot and table."""
        # Remove from plot
        for cursor in self._cursors:
            cursor.remove()
            
        self._cursors.clear()
        self._selected_cursor_index = None
        
        # Clear table
        self.cursor_table.setRowCount(0)
        
        # Update selection (clear it)
        self._delayed_selection_update()
        
        if self.parent_widget and self.parent_widget.canvas_widget:
            self.parent_widget.canvas_widget.figure.canvas.draw_idle()

    def _delayed_selection_update(self):
        """Update selection with a small delay to avoid excessive updates."""
        self._selection_update_timer.start(100)  # 100ms delay

    def _apply_cursor_selection(self):
        """Apply cursor selection to create a mask and update selection."""
        # Add guard to prevent recursive calls during selection updates
        if getattr(self, '_applying_selection', False):
            return

        self._applying_selection = True
        
        try:
            if not self._cursors or self.parent_widget._labels_layer_with_phasor_features is None:
                # If no cursors, clear selection
                if self.parent_widget._labels_layer_with_phasor_features is not None:
                    self.manual_selection_changed(None)
                return
                
            # Get current features
            features = self.parent_widget.get_features()
            if features is None:
                return
                
            x_data, y_data, _ = features
            
            if len(x_data) == 0:
                return
                
            # Initialize selection array with zeros (no selection)
            selection_indices = np.zeros(len(x_data), dtype=int)
            
            # Process each cursor individually and assign unique values
            for cursor_idx, row in enumerate(range(self.cursor_table.rowCount())):
                try:
                    center_x = float(self.cursor_table.item(row, 0).text())
                    center_y = float(self.cursor_table.item(row, 1).text())
                    radius_major = float(self.cursor_table.item(row, 2).text())
                    radius_minor = float(self.cursor_table.item(row, 3).text())
                    angle = float(self.cursor_table.item(row, 4).text())
                    
                    # Create mask for this specific cursor
                    try:
                        cursor_mask = mask_from_elliptic_cursor(
                            x_data,
                            y_data,
                            center_x,
                            center_y,
                            radius=radius_major,
                            radius_minor=radius_minor,
                            angle=np.radians(angle)
                        )
                        
                        # Assign unique cursor value (cursor_idx + 1, since 0 means no selection)
                        cursor_value = cursor_idx + 1
                        selection_indices[cursor_mask] = cursor_value
                        
                    except Exception as e:
                        print(f"Cursor {cursor_idx} failed: {e}")
                        continue
                    
                except (ValueError, AttributeError):
                    continue  # Skip invalid entries
            
            # Apply selection - now each cursor has its own unique value
            self.manual_selection_changed(selection_indices)
            
        finally:
            self._applying_selection = False

    @property
    def selection_id(self):
        """Gets or sets the selection id from the phasor selection id combobox.

        Value should not be one of `DATA_COLUMNS`.

        Returns
        -------
        str or None
            The selection id. Returns `None` if no selection id is available, "None" is selected, or empty string.

        """
        if (
            self.selection_input_widget.phasor_selection_id_combobox.count()
            == 0
        ):
            return None
        else:
            current_text = (
                self.selection_input_widget.phasor_selection_id_combobox.currentText()
            )
            return (
                None
                if current_text == "None" or current_text == ""
                else current_text
            )

    @selection_id.setter
    def selection_id(self, new_selection_id: str):
        """Sets the selection id from the phasor selection id combobox."""
        if new_selection_id is None or new_selection_id == "":
            new_selection_id = "None"

        if new_selection_id != "None" and new_selection_id in DATA_COLUMNS:
            notifications.WarningNotification(
                f"{new_selection_id} is not a valid selection column. It must not be one of {DATA_COLUMNS}."
            )
            return
        else:
            if new_selection_id not in [
                self.selection_input_widget.phasor_selection_id_combobox.itemText(
                    i
                )
                for i in range(
                    self.selection_input_widget.phasor_selection_id_combobox.count()
                )
            ]:
                self.selection_input_widget.phasor_selection_id_combobox.addItem(
                    new_selection_id
                )
            self.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
                new_selection_id
            )
            # Update the internal tracking variable
            self._current_selection_id = new_selection_id
            # Only add to features if it's not "None"
            if new_selection_id != "None":
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

    def _find_phasors_layer_by_name(self, layer_name):
        """Find a phasors layer by name in the viewer.

        Parameters
        ----------
        layer_name : str
            The name of the layer to find.

        Returns
        -------
        napari.layers.Layer or None
            The layer if found, None otherwise.
        """
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer
        return None

    def _on_show_color_overlay(self, visible: bool):
        """Slot to show/hide the current phasors_selected_layer."""
        if self._phasors_selected_layer is not None:
            self._phasors_selected_layer.visible = visible

    def _connect_show_overlay_signal(self):
        """Ensure show_color_overlay_signal is connected only to the current layer's visibility."""
        try:
            self.parent_widget.canvas_widget.show_color_overlay_signal.disconnect(
                self._on_show_color_overlay
            )
        except (TypeError, RuntimeError):
            pass
        self.parent_widget.canvas_widget.show_color_overlay_signal.connect(
            self._on_show_color_overlay
        )

    def on_selection_id_changed(self):
        """Callback function when the selection id combobox is changed.

        This function updates the selection and recreates/updates the phasors layer.
        """
        raw_combobox_text = (
            self.selection_input_widget.phasor_selection_id_combobox.currentText()
        )

        if raw_combobox_text == "":
            self.selection_id = ""

        new_selection_id = self.selection_id

        new_selection_id_for_comparison = (
            "None" if new_selection_id is None else new_selection_id
        )

        if self._current_selection_id != new_selection_id_for_comparison:

            # Set flag to prevent manual_selection_changed from firing
            self._switching_selection_id = True

            self._current_selection_id = new_selection_id_for_comparison
            if new_selection_id_for_comparison != "None":
                self.add_selection_id_to_features(
                    new_selection_id_for_comparison
                )

            # Check if we need to recreate a missing selection layer
            if (
                new_selection_id_for_comparison != "None"
                and self.parent_widget._labels_layer_with_phasor_features
                is not None
            ):

                layer_name = f"Selection: {new_selection_id_for_comparison}"
                existing_layer = self._find_phasors_layer_by_name(layer_name)

                # If layer doesn't exist but column exists in features, recreate it
                if (
                    existing_layer is None
                    and new_selection_id_for_comparison
                    in self.parent_widget._labels_layer_with_phasor_features.features.columns
                ):
                    self.create_phasors_selected_layer()
                else:
                    # If layer exists, just update the reference
                    self._phasors_selected_layer = existing_layer
            else:
                # If "None" is selected, set phasors_selected_layer to None
                self._phasors_selected_layer = None

            # Always (re)connect the overlay signal to the current layer
            self._connect_show_overlay_signal()

            processed_selection_id = new_selection_id

            # Only update the plot if we're not processing an initial selection
            if not getattr(self, '_processing_initial_selection', False):
                self.update_phasor_plot_with_selection_id(
                    processed_selection_id
                )
                # update phasor_selected_layer (needed if filtering was applied)
                if self._phasors_selected_layer is not None:
                    self.update_phasors_layer()

            self._switching_selection_id = False

    def update_phasor_plot_with_selection_id(self, selection_id):
        """Update the phasor plot with the selected ID and show/hide label layers."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        # Prevent this from running during plot updates
        if getattr(self.parent_widget, '_updating_plot', False):
            return

        # If selection_id is None, hide all selection layers and clear color indices
        if selection_id is None or selection_id == "":
            for layer in self.viewer.layers:
                if layer.name.startswith("Selection: "):
                    layer.visible = False

            # Clear color indices only for the active artist
            active_plot_type = self.parent_widget.plot_type
            if active_plot_type in self.parent_widget.canvas_widget.artists:
                self.parent_widget.canvas_widget.artists[
                    active_plot_type
                ].color_indices = 0

            # Trigger plot update to refresh the display
            self.parent_widget.plot()
            return

        # Check if the selection_id column exists in the features table
        if (
            selection_id
            not in self.parent_widget._labels_layer_with_phasor_features.features.columns
        ):
            # Don't create the column or update anything until there's actual selection data
            return

        target_layer_name = f"Selection: {selection_id}"
        for layer in self.viewer.layers:
            if layer.name.startswith("Selection: "):
                layer.visible = layer.name == target_layer_name

        harmonic_mask = (
            self.parent_widget._labels_layer_with_phasor_features.features[
                'harmonic'
            ]
            == self.parent_widget.harmonic
        )
        # Filter rows where 'G' and 'S' is not NaN
        valid_rows = (
            ~self.parent_widget._labels_layer_with_phasor_features.features[
                "G"
            ].isna()
            & ~self.parent_widget._labels_layer_with_phasor_features.features[
                "S"
            ].isna()
            & harmonic_mask
        )

        selection_data = (
            self.parent_widget._labels_layer_with_phasor_features.features.loc[
                valid_rows, selection_id
            ].values
        )

        # Update the color indices only for the active artist
        active_plot_type = self.parent_widget.plot_type
        if active_plot_type in self.parent_widget.canvas_widget.artists:
            self.parent_widget.canvas_widget.artists[
                active_plot_type
            ].color_indices = selection_data

        # Trigger plot update
        self.parent_widget.plot()

    def _get_next_available_selection_id(self):
        """Get the next available manual selection ID.

        Returns
        -------
        str
            The next available selection ID (e.g., "MANUAL SELECTION #1", "MANUAL SELECTION #2", etc.)
        """
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return "MANUAL SELECTION #1"

        existing_columns = (
            self.parent_widget._labels_layer_with_phasor_features.features.columns
        )
        counter = 1
        while True:
            candidate_name = f"MANUAL SELECTION #{counter}"
            if candidate_name not in existing_columns:
                return candidate_name
            counter += 1

    def manual_selection_changed(self, manual_selection):
        """Update the manual selection in the labels layer with phasor features."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        # Add guard to prevent recursive calls
        if getattr(self.parent_widget, '_updating_plot', False):
            return

        # Check if we're in the middle of switching selection IDs
        if getattr(self, '_switching_selection_id', False):
            return
            
        # Add guard to prevent recursive calls during manual selection updates
        if getattr(self, '_updating_manual_selection', False):
            return
            
        self._updating_manual_selection = True
        
        try:
            current_combobox_text = (
                self.selection_input_widget.phasor_selection_id_combobox.currentText()
            )

            # If "None" is selected in combobox, automatically switch to new selection ID
            if current_combobox_text == "None":
                new_selection_id = self._get_next_available_selection_id()

                # Set a flag to indicate we're processing the original manual selection
                self._processing_initial_selection = True
                self._initial_manual_selection = manual_selection

                self._current_selection_id = new_selection_id
                self.selection_id = new_selection_id

            self.add_selection_id_to_features(self.selection_id)
            column = self.selection_id

            self.parent_widget._labels_layer_with_phasor_features.features[
                column
            ] = 0

            # Filter rows where 'G' and 'S' is not NaN
            valid_rows = (
                ~self.parent_widget._labels_layer_with_phasor_features.features[
                    "G"
                ].isna()
                & ~self.parent_widget._labels_layer_with_phasor_features.features[
                    "S"
                ].isna()
            )
            num_valid_rows = valid_rows.sum()

            selection_to_use = manual_selection
            if (
                hasattr(self, '_processing_initial_selection')
                and self._processing_initial_selection
            ):
                selection_to_use = self._initial_manual_selection
                self._processing_initial_selection = False
                delattr(self, '_initial_manual_selection')

            # Handle case where selection_to_use is None
            if selection_to_use is None:
                # Set all values to 0 (no selection)
                self.parent_widget._labels_layer_with_phasor_features.features.loc[
                    valid_rows, column
                ] = 0
            else:
                tiled_manual_selection = np.tile(
                    selection_to_use, (num_valid_rows // len(selection_to_use)) + 1
                )[:num_valid_rows]

                self.parent_widget._labels_layer_with_phasor_features.features.loc[
                    valid_rows, column
                ] = tiled_manual_selection

            self.update_phasors_layer()
            
        finally:
            self._updating_manual_selection = False

    def create_phasors_selected_layer(self):
        """Create the phasors selected layer."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if self.selection_id is None or self.selection_id == "":
            return

        input_array = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.data
        )
        input_array_values = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.features[
                "label"
            ].values
        )

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

        layer_name = f"Selection: {self.selection_id}"
        phasors_selected_layer = Labels(
            mapped_data,
            name=layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap=DirectLabelColormap(
                color_dict=color_dict, name="cat10_mod"
            ),
        )

        self._phasors_selected_layer = self.viewer.add_layer(
            phasors_selected_layer
        )

        # Always (re)connect the overlay signal to the new layer
        self._connect_show_overlay_signal()

    def update_phasors_layer(self):
        """Update the existing phasors layer data without recreating it."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        layer_name = f"Selection: {self.selection_id}"
        existing_phasors_selected_layer = self._find_phasors_layer_by_name(
            layer_name
        )

        if existing_phasors_selected_layer is None:
            self.create_phasors_selected_layer()
            return

        input_array = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.data
        )
        input_array_values = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.features[
                "label"
            ].values
        )

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
            ).copy()
        valid_rows = (
            ~self.parent_widget._labels_layer_with_phasor_features.features[
                "G"
            ].isna()
            & ~self.parent_widget._labels_layer_with_phasor_features.features[
                "S"
            ].isna()
        )
        phasors_layer_data[~valid_rows] = 0
        mapped_data = map_array(
            input_array, input_array_values, phasors_layer_data
        )
        existing_phasors_selected_layer.data = mapped_data
        self._phasors_selected_layer = existing_phasors_selected_layer