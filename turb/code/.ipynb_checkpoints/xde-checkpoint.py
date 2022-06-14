import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf
from pde import *

#def call_xde( xde_params, data_params, nn_params, fold_name, file_name, cudev=0):
def call_xde( xde_params, pde_params, nn_params, scalings,  geom, bc, fnames):

    ndim, nvar, epochs, n, w = xde_params
    Re,     = pde_params
    learning_rate, num_dense_layers, num_dense_nodes, activation = nn_params
    xs, ys = scalings
    fold_name, file_name = fnames

    pde = incom_pde(ndim, nvar, xs, ys, Re)
    data = dde.data.PDE(
        geom,
        pde,
        bc,
        train_distribution = 'uniform',
        num_domain         = n ** 2,
        num_boundary       = n * 10,
        num_test           = n ** 2,
    )

    net = dde.maps.FNN([ndim] + [num_dense_nodes] * num_dense_layers + [nvar], activation, "Glorot uniform")
    model = dde.Model(data, net)
    loss_weights = [1]*3 + [w]*len(bc)
    model.compile("adam",lr=learning_rate,loss_weights=loss_weights)

    checkpointer = dde.callbacks.ModelCheckpoint(
    filepath = file_name,
    verbose=1,
    save_better_only=True,
    period=np.int(epochs/10))

    losshistory, train_state = model.train(epochs=epochs, model_save_path=file_name)
    #losshistory, train_state = model.train(epochs=epochs, callbacks=[checkpointer])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=fold_name)
    return model    
