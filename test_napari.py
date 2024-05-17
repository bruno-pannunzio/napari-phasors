#%%
import napari.viewer
import napari_phasors
import phasorpy.io as io
from phasorpy.phasor import phasor_from_signal
import pandas as pd
import numpy as np

data_path = '/Users/bruno/convallaria_000$EI0S.fbd'
calibration_path = '/Users/bruno/RH110CALIBRATION_000$EI0S.fbd'


data = io.read_fbd(data_path, frame=-1,keepdims=False)
for channel in range(data.shape[0]):
    print(channel)
    phasor = phasor_from_signal(data[channel])
    labels_id = range(phasor[0].size)
    data_table = pd.DataFrame({'label': labels_id,'DC': np.ravel(phasor[0]),'G': np.ravel(phasor[1]),'S': np.ravel(phasor[2])})
    print(data_table)
# %%
