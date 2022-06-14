import numpy as np
import scipy
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import KDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
from data_pmxl import *
from utils import *

def get_data():

    #Reading rectangle size and the x and y for deepxde setup
    xmin, xmax, ymin, ymax, x_coor, y_coor, X, Y, Ygridded, y_coor_gridded = pmxl_read()   # Problem configuration dependent

    y_coor_gridded_unnormalized = y_coor_gridded.copy()
    #Scale input and ouput data
    sf = 2.0
    ys = sf*np.max(np.abs(Y), axis=0)
    xs = np.max(X, axis=0)

    Y /= ys.T
    Ygridded /= ys.T
    X /= xs.T

    x_coor /= xs[0]
    y_coor /= xs[1]

    return X, Y, Ygridded, xs, ys, x_coor, y_coor, y_coor_gridded_unnormalized

def get_anchor(nxflag, Ny_anchor, xs, ys, x_coor, y_coor, X, Y, Ygridded, y_coor_gridded):   # Problem configuration dependent

    if nxflag == 1:
        xanchor = [0.15, 0.2, 0.25]
        counter = [0, 1, 2]
    elif nxflag == 2:
        xanchor = [0.65, 0.8, 0.95]
        counter = [3, 4, 5]
    elif nxflag == 3:
        xanchor = [0.15, 0.95]
        counter = [0, 5]

    Nx_anchor = len(xanchor)

    Xloc = X.copy()
    Yval = Y.copy()
    xcor = x_coor.copy()
    ycor = y_coor.copy()

    Xloc[:,0] = Xloc[:,0]*xs[0]
    Xloc[:,1] = Xloc[:,1]*xs[1]

    xcor = xcor*xs[0]
    ycor = ycor*xs[1]

    #Setup anchors
    x_coor_anchors = xanchor   # Concentrating anchor points in the transitional layer only
    Ygrid_anchors  = np.zeros((Ny_anchor, np.size(x_coor_anchors)))

    ############# Getting vorticity thickness
    Xdelom = [150, 200, 250, 650, 800, 950]
    delom = np.loadtxt('../../data_central/delom-ml1.dat')
    delom_X = np.zeros((6,2))
    for i in range(0,len(Xdelom)):
        for j in range(0,len(delom)):
            if Xdelom[i] == delom[j,0]:
                delom_X[i,:] = delom[j,:]
    delom_X = delom_X*(1e-3)   # Converts delom in meters
    #######################

    beta = 0.5
    counter2 = 0
    for i in counter:
        Ygrid_anchors[:, counter2] = np.concatenate((np.linspace(y_coor_gridded[0,i], -delom_X[i,1], np.int((1-beta)*0.5*Ny_anchor)),
                                     np.linspace(-delom_X[i,1], delom_X[i,1], np.int((beta)*Ny_anchor)),
                                     np.linspace(delom_X[i,1], y_coor_gridded[-1,i], np.int((1-beta)*0.5*Ny_anchor))))
        counter2 = counter2 + 1

    Xgrid_anchors   = np.tile(x_coor_anchors, (Ny_anchor, 1))
    #Ygrid_anchors   = np.tile(y_coor_anchors,(Nx_anchor, 1)).T

    X_star_anchors  = np.hstack((Xgrid_anchors.flatten()[:,None], Ygrid_anchors.flatten()[:,None]))
    kdtree          = KDTree(Xloc)
    d, idx_anchors  = kdtree.query(X_star_anchors)

    return idx_anchors

def data_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor, y_coor_gridded):    # Problem configuration dependent
    pmxl_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor, y_coor_gridded)
