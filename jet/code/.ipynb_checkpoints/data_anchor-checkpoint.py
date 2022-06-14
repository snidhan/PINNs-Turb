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
from data_turbjet import *
from utils import *

def get_data():

    #Reading rectangle size and the x and y for deepxde setup
    xmin, xmax, ymin, ymax, x_coor, y_coor, X, Y, Ygridded = turbjet_read()   # Problem configuration dependent

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

def get_anchor(Nx_anchor, Ny_anchor, xs, ys, x_coor, y_coor, X, Y, Ygridded):   # Problem configuration dependent

    Xloc      = X.copy()
    Yval      = Y.copy()
    xcor      = x_coor.copy()
    ycor      = y_coor.copy()

    Xloc[:,0] = Xloc[:,0]*xs[0]
    Xloc[:,1] = Xloc[:,1]*xs[1]

    xcor      = xcor*xs[0]
    ycor      = ycor*xs[1]

    #Setup anchors
    x_coor_anchors  = np.linspace(xcor[0], xcor[-1], Nx_anchor)   # Concentrating anchor points in the transitional layer only

    #y_coor_anchors  = np.linspace(ycor[0], ycor[-1], Ny_anchor)
    beta = 0.5
    ylower = 0.4; yupper = 0.6
    y_coor_anchors = np.concatenate((np.linspace(y_coor[0], ylower - (y_coor[1]-y_coor[0]), np.int((1-beta)*0.5*Ny_anchor)), 
                                     np.linspace(ylower, yupper, np.int((beta)*Ny_anchor)),
                                     np.linspace(yupper + (y_coor[1]-y_coor[0]), y_coor[-1], np.int((1-beta)*0.5*Ny_anchor)))) 

    Xgrid_anchors   = np.tile(x_coor_anchors,(Ny_anchor, 1))
    Ygrid_anchors   = np.tile(y_coor_anchors,(Nx_anchor, 1)).T

    X_star_anchors  = np.hstack((Xgrid_anchors.flatten()[:,None], Ygrid_anchors.flatten()[:,None]))
    kdtree          = KDTree(Xloc)

    d, idx_anchors  = kdtree.query(X_star_anchors)

    return idx_anchors

def data_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor):    # Problem configuration dependent
    turbjet_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor)
