import numpy as np
import os
from run_xde import run_xde
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Process

inpt = np.genfromtxt('inp',dtype='str',max_rows=1)
inp  = np.loadtxt('inp',skiprows=2)

if inp.ndim == 1:
    inp = np.reshape(inp,(1,inp.shape[0]))
act = {1:"tanh", 2:"relu"}
lr, lyr, wid, actv, epch, n, lossw, nxm, nym, alpha, beta, bctype = inp.T

def clean_float(s):
    return s.rstrip('0').rstrip('.') if '.' in s else s

def p2s(p1,p2): #param to string
    p = [a+clean_float(b) for a,b in zip(p1,p2.astype(str))]
    return '_'.join(str(e) for e in p)

#Params that don't change across runs
Re = 800
ndim = 2
nvar = 6

proc = []
fold_name = []
for c in range(0,4):
    xde_params  = [ndim, nvar, int(epch[c]), int(n[c]), lossw[c]]
    data_params = [Re, int(nxm[c]), int(nym[c]), alpha[c], beta[c], bctype[c]]
    nn_params   = [lr[c], int(lyr[c]), int(wid[c]), act[actv[c]]]

    fold_name.append(p2s(inpt, inp[c,:]))
    file_name      = 'file'

uniq_idx  = np.where(np.std(inp.T,axis=1)==0)
uniq_inp  = np.delete(inp.T,  uniq_idx, axis=0)
uniq_inpt = np.delete(inpt, uniq_idx)
print(uniq_inpt,'u v p uu uv vv')

fig, axs = plt.subplots(6, 1, figsize=(15,10), dpi=600)
gsp = gridspec.GridSpec(1,1)

for i,f in enumerate(fold_name):

    loss = np.loadtxt('../runs/' + f +'/loss.dat')
    axs[0].loglog(loss[:,0], np.sum(loss[:,1:3],1), '--',linewidth=1,label=str(uniq_inp[:,i])) # Training Residual Losses

    with open ('../runs/' + f + '/bc_loss_idx.bin', 'rb') as fp: # Reading bin file for indices of bc/anchor losses of different vars
            bc_loss_idx = pickle.load(fp)

    c = 1
    for j in range(nvar):
        idx_plot = 4 + bc_loss_idx[j].astype(int)
        if j != 2:
            if nxm[i] == 0:
                axs[c].loglog(loss[:,0], np.sum(loss[:,idx_plot],1), '--',linewidth=1,label=str(uniq_inp[:,i])) # BC losses if anchors ! given
            elif nxm[i] > 0:
                axs[c].loglog(loss[:,0], loss[:,idx_plot[-1]], '--',linewidth=1,label=str(uniq_inp[:,i])) # Anchor losses if anchors given
            c=c+1


axs.grid()
axs[0].legend(), axs[0].grid(), axs[0].set_ylabel('Residual Loss', rotation=90, labelpad=0,fontsize=12)
axs[1].legend(), axs[1].grid(), axs[1].set_ylabel('U Loss', rotation=90, labelpad=0,fontsize=12)
axs[2].legend(), axs[2].grid(), axs[2].set_ylabel('V Loss', rotation=90, labelpad=0,fontsize=12)
axs[3].legend(), axs[3].grid(), axs[3].set_ylabel('uu Loss', rotation=90, labelpad=0,fontsize=12)
axs[4].legend(), axs[4].grid(), axs[4].set_ylabel('uv Loss', rotation=90, labelpad=0,fontsize=12)
axs[5].legend(), axs[5].grid(), axs[5].set_ylabel('vv Loss', rotation=90, labelpad=0,fontsize=12)
plt.savefig('../runs/' + 'unique_losses.png', bbox_inches='tight', pad_inches=0.04, dpi=600)
plt.close()
