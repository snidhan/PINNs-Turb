import numpy as np
import os
from run_xde import run_xde
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Process
from utils import *
import pickle

############################## Inputs ############################
inpt = np.genfromtxt('inp',dtype='str',max_rows=1)
inp  = np.loadtxt('inp',skiprows=2)

if inp.ndim == 1:
    inp = np.reshape(inp,(1,inp.shape[0]))
act = {1:"tanh", 2:"relu"}
lr, lyr, wid, actv, epch, n, lossw1, lossw2, lossw3, losswa, nxm, nym, bctype = inp.T

batch = int(sys.argv[1])
c0    = int(batch)*8
c1    = min([c0 + 8, lr.shape[0]])

#Params that don't change across runs
Re   = 560100   # UjetDjet/nu = 172*2inch/(1.48*10^-5), Ma_ref = 0.5
ndim = 2
nvar = 7

############################## Runs ############################
proc = []
fold_name = []
for c in range(c0,c1):
    xde_params  = [ndim, nvar, int(epch[c]), int(n[c]), lossw1[c], lossw2[c], lossw3[c], losswa[c]]
    data_params = [Re, int(nxm[c]), int(nym[c]), bctype[c]]
    nn_params   = [lr[c], int(lyr[c]), int(wid[c]), act[actv[c]]]

    fold_name.append(p2s(inpt, inp[c,:]))
    file_name      = 'file'
    pr = Process(target=run_xde, args=(xde_params, data_params, nn_params, fold_name[c-c0], file_name, c))
    proc.append(pr); pr.start()

for ip in proc:
    ip.join()

############################## On-the-fly postproc ############################

uniq_idx  = np.where(np.std(inp.T,axis=1)==0)
uniq_inp  = np.delete(inp.T,  uniq_idx, axis=0)
uniq_inpt = np.delete(inpt, uniq_idx)

fig, axs  = plt.subplots(4, 2, figsize=(40,20), dpi=300)
gsp       = gridspec.GridSpec(1,1)
axs       = axs.flatten()

glob_error = open('./unique_glob_error.txt', 'a')
phy_vars   = ['u', 'v', 'uu' ,'vv' ,'ww' ,'uv']
np.savetxt(glob_error, np.concatenate((uniq_inpt, phy_vars),axis=0).reshape(1,len(uniq_inpt)+len(phy_vars)), fmt = '% 10s')  #print(uniq_inpt,'u v p uu uv vv')

for i,f in enumerate(fold_name):
    ge = np.loadtxt('../runs/' + f + '/GlobalErrors.txt')
    #print(uniq_inp[:,i].T,ge)
    np.savetxt(glob_error, np.concatenate((uniq_inp[:,i].T, ge),axis=0).reshape(1,len(uniq_inp[:,i])+len(ge)), fmt = '% 10.5g')

    loss = np.loadtxt('../runs/' + f +'/loss.dat')
    axs[0].semilogy(loss[:,0], loss[:,1]/lossw1[i] + loss[:,2]/lossw2[i] + loss[:,3]/lossw3[i], '--', linewidth=1.5, label=str(uniq_inp[:,i])) # Training Residual Losses

    with open ('../runs/' + f + '/bc_loss_idx.bin', 'rb') as fp: # Reading bin file for indices of bc/anchor losses of different vars
            bc_loss_idx = pickle.load(fp)
    print('bc_loss_idx ', bc_loss_idx)
    c = 1
    for j in range(nvar-1):    # Skipping nvar == pressure
        idx_plot = 4 + bc_loss_idx[j].astype(int)
        if nxm[i] == 0:
            axs[c].semilogy(loss[:,0], np.sum(loss[:,idx_plot],1)/losswa[i], '--', linewidth=1.5, label=str(uniq_inp[:,i])) # BC losses if anchors ! given
        elif nxm[i] > 0:
            axs[c].semilogy(loss[:,0], loss[:,idx_plot[-1]]/losswa[i], '--', linewidth=1.5, label=str(uniq_inp[:,i])) # Anchor losses if anchors given
        c=c+1

glob_error.close()
axs[0].legend(), axs[0].grid(), axs[0].set_ylabel('Residual Loss', rotation=90, labelpad=0, fontsize=20)
axs[1].legend(), axs[1].grid(), axs[1].set_ylabel('U Loss',  rotation=90, labelpad=0, fontsize=20)
axs[2].legend(), axs[2].grid(), axs[2].set_ylabel('V Loss',  rotation=90, labelpad=0, fontsize=20)
axs[3].legend(), axs[3].grid(), axs[3].set_ylabel('uu Loss', rotation=90, labelpad=0, fontsize=20)
axs[4].legend(), axs[4].grid(), axs[4].set_ylabel('vv Loss', rotation=90, labelpad=0, fontsize=20)
axs[5].legend(), axs[5].grid(), axs[5].set_ylabel('ww Loss', rotation=90, labelpad=0, fontsize=20)
axs[6].legend(), axs[6].grid(), axs[6].set_ylabel('uv Loss', rotation=90, labelpad=0, fontsize=20)
axs[7].legend(), axs[7].grid(), axs[7].set_ylabel('Placeholder', rotation=90, labelpad=0, fontsize=20)
plt.savefig('./unique_losses.png', bbox_inches='tight', pad_inches=0.04, dpi=300)
plt.close()
