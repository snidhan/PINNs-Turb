from data_anchor import *
import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf
from xde import load_xde
import os
import pickle
def run_xde(xde_params, data_params, nn_params, fold_name, file_name, cudev=0):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cudev)
    ndim, nvar, _, _, _, _, _, _       = xde_params
    Re, nxm, nym, bctype               = data_params

    fold_name = '../runs/' + fold_name
    isExist   = os.path.exists('./' + fold_name)
    if not isExist:
        os.makedirs('./' + fold_name)
    file_name = fold_name + '/' + file_name + '.ckpt'

    #geometry, data and anchor points
    X, Y, Ygrid, xs, ys, xc, yc = get_data()
    geom                        = dde.geometry.Rectangle([np.min(xc), np.min(yc)], [np.max(xc), np.max(yc)])
    
    #set bcs and anchors
    bc = []

    #train model
    model = load_xde(xde_params, [Re,], nn_params, [xs,ys], geom, bc, [fold_name, file_name])
    
    return model, X, Y, xs, ys, xc, yc