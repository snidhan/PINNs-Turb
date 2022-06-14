import numpy as np
import struct as st

def find_nearest1d(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def find_nearest2d(array, value):
    array = np.asarray(array)
    idx   = np.zeros((np.size(value)))
    for j in range(0,np.size(value)):
        idx[j] = (np.abs(array - value[j])).argmin()
    return idx
