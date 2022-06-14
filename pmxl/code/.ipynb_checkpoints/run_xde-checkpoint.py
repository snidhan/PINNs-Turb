from data_anchor import *
import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf
from xde import call_xde
import os
import pickle

def run_xde(xde_params, data_params, nn_params, fold_name, file_name, cudev=0):

    os.environ["CUDA_VISIBLE_DEVICES"]         = str(cudev)
    ndim, nvar, _, _, _, _, _, _               = xde_params
    Re, xanchor, nym, y_loc, bctype            = data_params


    fold_name = '../runs/' + fold_name
    isExist = os.path.exists('./' + fold_name)
    if not isExist:
        os.makedirs('./' + fold_name)
    file_name = fold_name + '/' + file_name + '.ckpt'

    #geometry, data and anchor points
    geom = dde.geometry.Rectangle([np.min(xc), np.min(yc)], [np.max(xc), np.max(yc)])
    X, Y, Ygrid, xs, ys, xc, yc = get_data()
    
    if len(xanchor) > 0:
        idx_anchors = get_anchor(xanchor, nym, xs, ys, xc, yc, X, Y, Ygrid)

        fig = plt.figure(figsize=(4,2.2), dpi=200)
        gsp = gridspec.GridSpec(1,1)
        ax0 = plt.subplot(gsp[0,0])
        cs0 = plt.plot(X[idx_anchors,0]*xs[0], X[idx_anchors,1]*xs[1], 'rs', markersize=1)
        plt.grid()
        plt.savefig('anchors.png')

    #set bcs and anchors
    bc = []
    def getibc(xbc, ybc, c):
            return lambda x: interpolate.interp1d(xbc, ybc, kind ='cubic')(x[:, c:c+1])
    def getvbc(n_bc,c):
            vbc = [[1.175,0,0,0,0,0], [2.275,0,0,0,0,0]]
            return lambda x: vbc[n_bc][c]

    bc_loss_idx = []              ## indices for BC and anchor losses of different vars
    counter_prev = 0              ## counter to track length of a var loss
    for i in range(nvar):
        #match bctype:
            if nvar > 1:
                counter_prev = len(bc)

            if bctype == 0: #All 4 BCS
                if i != 2:
                    bc.append(dde.DirichletBC(geom, lambda x: 0,  lambda x, on_boundary:  np.isclose(x[1], yc[0]), component = i))
                bc.append(dde.DirichletBC(geom, getibc(xc,Ygrid[-1, :,i],0),  lambda x, on_boundary:  np.isclose(x[1], yc[-1]), component = i))
                bc.append(dde.DirichletBC(geom, getibc(yc,Ygrid[ :, 0,i],1),  lambda x, on_boundary:  np.isclose(x[0], xc[ 0]), component = i))
                bc.append(dde.DirichletBC(geom, getibc(yc,Ygrid[ :,-1,i],1),  lambda x, on_boundary:  np.isclose(x[0], xc[-1]), component = i))

            if bctype == 1: #Constant top and bottom
                if i != 1: #skip v
                    bc.append(dde.DirichletBC(geom, getvbc(0,i),  lambda x, on_boundary:  np.isclose(x[1], yc[0] ), component = i))
                    bc.append(dde.DirichletBC(geom, getvbc(1,i),  lambda x, on_boundary:  np.isclose(x[1], yc[-1]), component = i))

            if bctype == 2: #Exact top and 0 bottom
                if i != 1: #skip v
                    bc.append(dde.DirichletBC(geom, getvbc(0,i),  lambda x, on_boundary:  np.isclose(x[1], yc[0] ), component = i))
                    bc.append(dde.DirichletBC(geom, getibc(xc,Ygrid[-1, :,i],0),  lambda x, on_boundary:  np.isclose(x[1], yc[-1]), component = i))

            if len(xanchor) > 0:
                bc.append(dde.PointSetBC(X[idx_anchors], Y[idx_anchors,i:i+1], component=i))

            counter_nex = len(bc)
            bc_loss_idx.append(np.linspace(counter_prev, counter_nex-1, counter_nex-counter_prev))

    with open(fold_name + '/bc_loss_idx.bin', 'wb') as fp:    # Writes a bin file with different lines corresponding to indices of bc+anchor loss in loss.dat
        pickle.dump(bc_loss_idx, fp)

    #train model
    model = call_xde(xde_params, [Re,], nn_params, [xs,ys], geom, bc, [fold_name, file_name])
    #pp model
    data_postproc(fold_name, X, Y, model.predict(X), xs, ys, xc, yc)
