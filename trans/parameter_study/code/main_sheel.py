import deepxde as dde
import numpy as np
import torch
import scipy
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import KDTree
import tensorflow.compat.v1 as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
from pde import *
from data_anchor import *
from xde import call_xde
import numpy as np
import sys
from multiprocessing import Process

inp = np.loadtxt('inp')
if inp.ndim == 1:
    inp = np.reshape(inp,(1,inp.shape[0]))
act = {1:"tanh", 2:"relu"}
lr, lyr, wid, actv, epch, n, lossw, nxm, nym, alpha, beta = inp.T

#batch = int(sys.argv[1])
#c0    = int(batch)*8
#c1    = min([c0 + 8, lr.shape[0]])

def p2s(p,sym): #param to string
    return sym.join(str(e) for e in p)

#Params that don't change across runs
Re = 800
ndim = 2
nvar = 6

proc = []
c=0
#for c in range(c0,c1):
nn_params   = [lr[c], int(lyr[c]), int(wid[c]), act[actv[c]]]
data_params = [Re, int(nxm[c]), int(nym[c]), alpha[c], beta[c]]
xde_params  = [ndim, nvar, int(epch[c]), int(n[c]), lossw[c]]
fold_name   = p2s(xde_params,'_') +'___'+ p2s(data_params,'_')
file_name   = p2s(nn_params,'_')
pr = Process(target=call_xde, args=(xde_params, data_params, nn_params, fold_name, file_name, 0))
proc.append(pr); pr.start()
#for ip in proc:
#    i
