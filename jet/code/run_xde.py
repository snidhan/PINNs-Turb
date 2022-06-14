from data_anchor import *
import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf
from xde import call_xde
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

    if nxm > 0 :
        idx_anchors = get_anchor(nxm, nym, xs, ys, xc, yc, X, Y, Ygrid)

    ## Plotting anchors
    fig = plt.figure(figsize=(4,2.2), dpi=200)
    gsp = gridspec.GridSpec(1,1)
    ax0 = plt.subplot(gsp[0,0])
    cs0 = plt.plot(X[idx_anchors,0], X[idx_anchors,1], 'rs', markersize=1)
    plt.grid()
    plt.savefig('./anchors.png')

    #set bcs and anchors
    bc = []
    def getibc(xbc, ybc, c):
            return lambda x: interpolate.interp1d(xbc, ybc, kind ='cubic')(x[:, c:c+1])

    bc_loss_idx = []              ## indices for BC and anchor losses of different vars
    counter_prev = 0              ## counter to track length of a var loss
    for i in range(nvar-1):
        #match bctype:
            if nvar > 1:
                counter_prev = len(bc)

            if bctype == 0: #All BCs at all the boundary
                bc.append(dde.DirichletBC(geom, getibc(xc,Ygrid[0,:,i],0),  lambda x, on_boundary:  np.isclose(x[1], yc[0]),  component = i)) # Axis
                bc.append(dde.DirichletBC(geom, getibc(xc,Ygrid[-1,:,i],0), lambda x, on_boundary:  np.isclose(x[1], yc[-1]), component = i)) # r top
                #bc.append(dde.DirichletBC(geom, getibc(yc,Ygrid[:,0,i],1),  lambda x, on_boundary:  np.isclose(x[0], xc[ 0]), component = i)) # Inlet
                #bc.append(dde.DirichletBC(geom, getibc(yc,Ygrid[:,-1,i],1), lambda x, on_boundary:  np.isclose(x[0], xc[-1]), component = i)) # Outlet

            if bctype == 1: #Constant top and bottom
                bc.append(dde.DirichletBC(geom, getvbc(0,i),  lambda x, on_boundary:  np.isclose(x[1], yc[0] ), component = i))
                bc.append(dde.DirichletBC(geom, getvbc(1,i),  lambda x, on_boundary:  np.isclose(x[1], yc[-1]), component = i))

            if bctype == 2: #Exact bottom and no top
                pass
                #bc.append(dde.DirichletBC(geom, getvbc(0,i),  lambda x, on_boundary:  np.isclose(x[1], yc[0] ), component = i))
                #bc.append(dde.DirichletBC(geom, getibc(xc,Ygrid[0, :,i],0),  lambda x, on_boundary:  np.isclose(x[1], yc[0]), component = i))

            if nxm>0:
                bc.append(dde.PointSetBC(X[idx_anchors], Y[idx_anchors,i:i+1], component=i))

            counter_nex = len(bc)
            bc_loss_idx.append(np.linspace(counter_prev, counter_nex-1, counter_nex-counter_prev))

    with open(fold_name + '/bc_loss_idx.bin', 'wb') as fp:    # Writes a bin file with different lines corresponding to indices of bc+anchor loss in loss.dat
        pickle.dump(bc_loss_idx, fp)

    #train model
    model = call_xde(xde_params, [Re,], nn_params, [xs,ys], geom, bc, [fold_name, file_name])
    #pp model
    data_postproc(fold_name, X, Y, model.predict(X), xs, ys, xc, yc)
