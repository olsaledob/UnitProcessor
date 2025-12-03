import h5py
import numpy as np

def load_h5_to_dict(filename):
    with h5py.File(filename, 'r') as f:
        data = {}
        for key in f.keys():
            value = f[key][()]
            if isinstance(value, np.ndarray):
                data[key] = value
            else:
                data[key] = value[0]
        return data