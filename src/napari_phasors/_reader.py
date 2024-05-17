"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import os
import numpy as np
import phasorpy.io as io
from phasorpy.phasor import phasor_from_signal
import pandas as pd
from napari.layers import Labels, Image

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    """
    supported_raw_formats = [
        ".ptu",
        ".fbd",
        ".flif",
        ".sdt",
        ".bh",
        ".bhz",
        # ".ifli",
        ".lsm"
    ]
    supported_processed_formats = [
        ".tif",
        # ".b64",
        # ".r64",
        # ".ref"
    ]
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension.lower()
    if file_extension in supported_processed_formats:
        return phasor_file_reader
    elif file_extension in supported_raw_formats:
        return raw_file_reader
    else:
        return unkonwn_reader_function

def unkonwn_reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))
    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data[0], add_kwargs, layer_type)]


def raw_file_reader(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    extension_mapping = {
        ".ptu": lambda path: io.read_ptu(path, frame=-1, keepdims=False),
        ".fbd": lambda path: io.read_fbd(path, frame=-1, keepdims=False),
        ".flif": lambda path: io.read_flif(path),
        ".sdt": lambda path: io.read_sdt(path),
        ".bh": lambda path: io.read_bh(path),
        ".bhz": lambda path: io.read_bhz(path),
        # ".ifli": lambda path: io.read_ifli(),
        ".lsm": lambda path: io.read_lsm(path),
    }
    filename = os.path.basename(path)
    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lower()
    raw_data = extension_mapping[file_extension](path)
    layers = []
    for channel in range(raw_data.shape[0]):
        mean_intensity_image, G_image, S_image = phasor_from_signal(raw_data[channel])
        pixel_id = np.arange(1, mean_intensity_image.size + 1)
        if len(G_image.shape) > 2:
            table = pd.DataFrame([])
            for i in range(G_image.shape[0]):
                sub_table = pd.DataFrame({'label': pixel_id, 'G': G_image[i].ravel(), 'S': S_image[i].ravel(), 'harmonic': i+1})  
                table = pd.concat([table, sub_table])
        else:
            table = pd.DataFrame({'label': pixel_id, 'G': G_image.ravel(), 'S': S_image.ravel(), 'harmonic': 1})
        labels_data = pixel_id.reshape(mean_intensity_image.shape)
        labels_layer = Labels(labels_data, name=filename + ' Phasor Features Layer', scale=(1, 1), features=table)
        add_kwargs = {'name': filename + ' Intensity Image', 'metadata':{'phasor_features_labels_layer': labels_layer}}
        layers.append((mean_intensity_image, add_kwargs))
    return layers

def phasor_file_reader(path):
    extension_mapping = {
        ".tif": lambda path: io.read_ometiff_phasor(path),
        # ".b64": lambda path: io.read_b64(path),
        # ".r64": lambda path: io.read_r64(path),
        # ".ref": lambda path: io.read_ref(path)
    }
    filename = os.path.basename(path)
    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lower()
    mean_intensity_image, G_image, S_image = extension_mapping[file_extension](path)
    mean_intensity_image, G_image, S_image = mean_intensity_image.values, G_image.values, S_image.values
    pixel_id = np.arange(1, mean_intensity_image.size + 1)
    layers = []
    if len(G_image.shape) > 2:
        table = pd.DataFrame([])
        for i in range(G_image.shape[0]):
            sub_table = pd.DataFrame({'label': pixel_id, 'G': G_image[i].ravel(), 'S': S_image[i].ravel(), 'harmonic': i+1})  
            table = pd.concat([table, sub_table])
    else:
        table = pd.DataFrame({'label': pixel_id, 'G': G_image.ravel(), 'S': S_image.ravel(), 'harmonic': 1})
    labels_data = pixel_id.reshape(mean_intensity_image.shape)
    labels_layer = Labels(labels_data, name=filename + ' Phasor Features Layer', scale=(1, 1), features=table)
    add_kwargs = {'name': filename + ' Intensity Image', 'metadata':{'phasor_features_labels_layer': labels_layer}}
    layers.append((mean_intensity_image, add_kwargs))
    return layers