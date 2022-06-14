import deepxde as dde
import numpy as np
import torch
import scipy
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import KDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
from data_turbBL import *
from utils import *

def get_data(xmin_idx, xmax_idx):

    #Reading rectangle size and the x and y for deepxde setup
    xmin, xmax, ymin, ymax, x_coor, y_coor, X, Y, Ygridded = turbBL_read(xmin_idx, xmax_idx)   # Problem configuration dependent

    #Scale input and ouput data
    sf = 2.0
    ys = sf*np.max(np.abs(Y),axis=0)
    xs = np.max(X,axis=0)
    Y /= ys.T
    Ygridded /= ys.T
    X /= xs.T
    x_coor /= xs[0]
    y_coor /= xs[1]

    return X, Y, Ygridded, xs, ys, x_coor, y_coor

def get_anchor(Nx_anchor, Ny_anchor, alpha, beta, xs, ys, x_coor, y_coor, X, Y, Ygridded):   # Problem configuration dependent

    Xloc = X.copy()
    Yval = Y.copy()
    xcor = x_coor.copy()
    ycor = y_coor.copy()

    Xloc[:,0] = Xloc[:,0]*xs[0]
    Xloc[:,1] = Xloc[:,1]*xs[1]

    xcor = xcor*xs[0]
    ycor = ycor*xs[1]

    U = ys[0]*Ygridded[:,:,0]
    Uaux = U/U[-1,:]*(1-U/U[-1,:])
    theta = np.trapz(Uaux, ycor, axis=0)

    #Setup anchors
    x_coor_anchors = np.linspace(xcor[0], xcor[-1], Nx_anchor)   # Concentrating anchor points in the transitional layer only
    Ygrid_anchors  = np.zeros((Ny_anchor,Nx_anchor))

    for i in range(0, Nx_anchor):
        val, idx1  = find_nearest1d(xcor, x_coor_anchors[i])
        ThetaQuery = alpha*theta[idx1]
        Ygrid_anchors[0:int(Ny_anchor*beta),i] = np.linspace(0, ThetaQuery, int(Ny_anchor*beta))
        Ygrid_anchors[int(Ny_anchor*beta):,i]  = np.linspace(Ygrid_anchors[int(Ny_anchor*beta),i], ycor[-1], int(Ny_anchor*(1-beta)))

    Xgrid_anchors   = np.tile(x_coor_anchors,(Ny_anchor, 1))
    X_anchors  = np.hstack((Xgrid_anchors.flatten()[:,None], Ygrid_anchors.flatten()[:,None]))
    kdtree = KDTree(Xloc)
    d, idx_anchors = kdtree.query(X_anchors)

    return idx_anchors

#    def anc(i):
#        return x[idx_anchors],y[i,idx_anchors]
#
#    return x, y, xs, ys, dbc, anc

def get_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor):    # Problem configuration dependent
    turbBL_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor)
