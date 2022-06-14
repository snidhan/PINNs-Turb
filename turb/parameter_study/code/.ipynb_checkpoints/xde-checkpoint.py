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
from sys import getsizeof

from pde import *
from data_anchor import *

def call_xde( xde_params, data_params, nn_params, fold_name, file_name, cudev=0):

    put_anchors = 1
    ndim, nvar, epochs, n, w = xde_params
    Re, nxm, nym, alpha, beta = data_params
    learning_rate, num_dense_layers, num_dense_nodes, activation = nn_params

    os.environ["CUDA_VISIBLE_DEVICES"]=str(cudev)

    precision_train  = 10
    precision_test   = 30
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    ## Data input
    xmin_idx = 2085
    xmax_idx = 3310
    X, Y, Ygridded, xs, ys, x_coor, y_coor = get_data(xmin_idx, xmax_idx)   # X == size(Npoints, 2), Y = size(Npoints, nvar), Ygridded = size(Ny, Nx, nvar)

    ## Boundary conditions
    bc = []
    def getbc(xbc,ybc,c):
        return lambda x: interpolate.interp1d(xbc,ybc,kind ='cubic')(x[:,c:c+1])
    for i in range(nvar):
            if i != 2:
               bc.append(dde.DirichletBC(geom, lambda x: 0,  lambda x, on_boundary:  np.isclose(x[1], y_coor[0]), component = i))
            bc.append(dde.DirichletBC(geom, getbc(x_coor,Ygridded[-1,:,i],0),  lambda x, on_boundary:  np.isclose(x[1], y_coor[-1]), component = i))
            #bc.append(dde.DirichletBC(geom, getbc(y_coor,Ygridded[:,0,i],1),  lambda x, on_boundary:  np.isclose(x[0], x_coor[ 0]), component = i))
            #bc.append(dde.DirichletBC(geom, getbc(y_coor,Ygridded[:,-1,i],1),  lambda x, on_boundary:  np.isclose(x[0], x_coor[-1]), component = i))

    ## Anchor points
    idx_anchors = get_anchor(nxm, nym, alpha, beta, xs, ys, x_coor, y_coor, X, Y, Ygridded)
    fig = plt.figure(figsize=(4,2.2), dpi=200)
    gsp = gridspec.GridSpec(1,1)
    ax0 = plt.subplot(gsp[0,0])
    cs0 = plt.plot(X[idx_anchors,0], X[idx_anchors,1], 'rs', markersize=1)
    plt.grid()
    fig.tight_layout()
    plt.savefig("./anchor.png")
    plt.close(fig)

    #anchors are appended in bc itself
    for i in range(nvar):
        bc.append(dde.PointSetBC(X[idx_anchors], Y[idx_anchors,i:i+1], component=i))

    # PDE
    pde = incom_pde(ndim, nvar, xs, ys, Re)

    # Neural Network
    data = dde.data.PDE(
        geom,
        pde,
        bc,
        train_distribution = 'uniform',
        anchors = X[idx_anchors],
        num_domain         = n ** 2,
        num_boundary       = n * 10,
        num_test           = n ** 2,
    )

    net = dde.maps.FNN([ndim] + [num_dense_nodes] * num_dense_layers + [nvar], activation, "Glorot uniform")
    model = dde.Model(data, net)
    loss_weights = [1]*3 + [w]*len(bc)
    model.compile("adam",lr=learning_rate,loss_weights=loss_weights)


    isExist = os.path.exists('./' + fold_name)
    if not isExist:
        os.makedirs('./' + fold_name)
    file_name = fold_name + '/' + file_name + '.ckpt'

    checkpointer = dde.callbacks.ModelCheckpoint(
    filepath = file_name,
    verbose=1,
    save_better_only=True,
    period=np.int(epochs/10))

    losshistory, train_state = model.train(epochs=epochs, callbacks=[checkpointer])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=fold_name)

    ## Postprocessing
    get_postproc(fold_name, X, Y, model.predict(X), xs, ys, x_coor, y_coor)
