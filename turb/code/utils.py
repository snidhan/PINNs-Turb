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

def clean_float(s):
    return s.rstrip('0').rstrip('.') if '.' in s else s

def p2s(p1,p2): #param to string
    p = [a+clean_float(b) for a,b in zip(p1,p2.astype(str))]
    return '_'.join(str(e) for e in p)