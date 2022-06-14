from data_anchor import *
import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf
from xde import call_xde
import os

def run_xde( xde_params, data_params, nn_params, fold_name, file_name, cudev=0):

    os.environ["CUDA_VISIBLE_DEVICES"]=str(cudev)
    ndim, nvar, _, _, _         = xde_params
    Re, nxm, nym, alpha, beta   = data_params


    fold_name = '../runs/' + fold_name
    isExist = os.path.exists('./' + fold_name)
    if not isExist:
        os.makedirs('./' + fold_name)
    file_name = fold_name + '/' + file_name + '.ckpt'

    #geometry, data and anchor points
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    X, Y, Ygrid, xs, ys, xc, yc = get_data(60,1885)
    idx_anchors = get_anchor(nxm, nym, alpha, beta, xs, ys, xc, yc, X, Y, Ygrid)

    #set bcs and anchors
    bc = []
    def getibc(xbc,ybc,c):
            return lambda x: interpolate.interp1d(xbc,ybc,kind ='cubic')(x[:,c:c+1])
    def getvbc(n_bc,c):
            vbc = [[0,0,0,0,0,0],[1,0,0,0,0,0]]
            return lambda x: vbc[n_bc][c]
    sw = {
            "ful_bcs":0, #interp all and 0 bot
            "con_bcs":1, #constant top and 0 bot
            "int_bcs":0, #interp top and 0 bot
            "put_anc":1
    }

    for i in range(nvar):

            if sw["ful_bcs"]:
             if i != 2:
                bc.append(dde.DirichletBC(geom, lambda x: 0,  lambda x, on_boundary:  np.isclose(x[1], yc[0]), component = i))
             bc.append(dde.DirichletBC(geom, getibc(xc,Ygrid[-1, :,i],0),  lambda x, on_boundary:  np.isclose(x[1], yc[-1]), component = i))
             bc.append(dde.DirichletBC(geom, getibc(yc,Ygrid[ :, 0,i],1),  lambda x, on_boundary:  np.isclose(x[0], xc[ 0]), component = i))
             bc.append(dde.DirichletBC(geom, getibc(yc,Ygrid[ :,-1,i],1),  lambda x, on_boundary:  np.isclose(x[0], xc[-1]), component = i))

            if sw["con_bcs"]:
             if i != 2:
                bc.append(dde.DirichletBC(geom, getvbc(0,i),  lambda x, on_boundary:  np.isclose(x[1], yc[0] ), component = i))
                bc.append(dde.DirichletBC(geom, getvbc(1,i),  lambda x, on_boundary:  np.isclose(x[1], yc[-1]), component = i))

            if sw["int_bcs"]:
             if i != 2:
                bc.append(dde.DirichletBC(geom, getvbc(0,i),  lambda x, on_boundary:  np.isclose(x[1], yc[0] ), component = i))
                bc.append(dde.DirichletBC(geom, getibc(xc,Ygrid[-1, :,i],0),  lambda x, on_boundary:  np.isclose(x[1], yc[-1]), component = i))

            if sw["put_anc"]:
               bc.append(dde.PointSetBC(X[idx_anchors], Y[idx_anchors,i:i+1], component=i))

    #train model
    model = call_xde(xde_params, [Re,], nn_params, [xs,ys], geom, bc, [fold_name, file_name])
    #pp model
    data_postproc(fold_name, X, Y, model.predict(X), xs, ys, xc, yc)
