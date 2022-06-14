import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf
from pde import *

#def call_xde( xde_params, data_params, nn_params, fold_name, file_name, cudev=0):
def call_xde( xde_params, pde_params, nn_params, scalings,  geom, bc, fnames):

    ndim, nvar, epochs, n, w1, w2, w3, wa = xde_params
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
    loss_weights = [w1]*1 + [w2]*1 + [w3]*1 + [wa]*len(bc)    # Different weights for PDEs and losses
    print('Loss weights ', loss_weights)

    ###################### Adam step ######################################

    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)

    #checkpointer = dde.callbacks.ModelCheckpoint(
    #filepath = file_name,
    #verbose=1,
    #save_better_only=True,
    #period=np.int(epochs/10))

    losshistory, train_state = model.train(epochs=epochs, model_save_path=file_name)
    #losshistory, train_state = model.train(epochs=epochs, callbacks=[checkpointer])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=fold_name)

    ####################### LBFGS step ####################################

    #dde.optimizers.config.set_LBFGS_options(maxiter=np.int(epochs/2))
    #model.compile("L-BFGS", lr=learning_rate, loss_weights=loss_weights)

    #losshistory, train_state = model.train(epochs=np.int(epochs/2), model_save_path=file_name)
    #dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=fold_name)

    return model
