import napari
from napari_phasors._widget import CalibrationWidget

viewer = napari.Viewer()


file_path = '/Users/bruno/Documents/UBA/phasorpy/test_data/FBDfiles-DIVER/BUENOS/convallaria_000$EI0S.fbd'
calibration_path = '/Users/bruno/Documents/UBA/phasorpy/test_data/FBDfiles-DIVER/BUENOS/RH110CALIBRATION_000$EI0S.fbd'

viewer.open(file_path, plugin='napari-phasors')
viewer.open(calibration_path, plugin='napari-phasors')

calibration_widget = CalibrationWidget(viewer)
viewer.window.add_dock_widget(calibration_widget)
napari.run()