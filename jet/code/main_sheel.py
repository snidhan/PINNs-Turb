import numpy as np
import os
from run_xde import run_xde
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from multiprocessing import Process

inpt = np.genfromtxt('inp',dtype='str',max_rows=1)
inp  = np.loadtxt('inp',skiprows=2)

if inp.ndim == 1:
    inp = np.reshape(inp,(1,inp.shape[0]))
act = {1:"tanh", 2:"relu"}
lr, lyr, wid, actv, epch, n, lossw, nxm, nym, bctype = inp.T

def clean_float(s):
    return s.rstrip('0').rstrip('.') if '.' in s else s

def p2s(p1,p2): #param to string
    p = [a+clean_float(b) for a,b in zip(p1,p2.astype(str))]
    return '_'.join(str(e) for e in p)

#Params that don't change across runs
Re = 5601
ndim = 2
nvar = 7

proc = []
fold_name = []
for c in range(0,1):
    xde_params  = [ndim, nvar, int(epch[c]), int(n[c]), lossw[c]]
    data_params = [Re, int(nxm[c]), int(nym[c]), bctype[c]]
    nn_params   = [lr[c], int(lyr[c]), int(wid[c]), act[actv[c]]]

    fold_name.append(p2s(inpt, inp[c,:]))
    file_name      = 'file'
    pr = Process(target=run_xde, args=(xde_params, data_params, nn_params, fold_name[c], file_name, c))
    proc.append(pr); pr.start()

#uniq_idx  = np.where(np.std(inp.T,axis=1)==0)
#uniq_inp  = np.delete(inp.T,  uniq_idx, axis=0)
#uniq_inpt = np.delete(inpt, uniq_idx)
#print(uniq_inpt,'u v p uu uv vv')
#
#idx_plot = np.linspace(4,24,24-4+1).astype(int)  # Indices to sum over for loss plots for BC + anchor losses
#
#fig, axs = plt.subplots(1, 2, figsize=(14,7), dpi=600)
#gsp = gridspec.GridSpec(1,1)
#
#for i,f in enumerate(fold_name):
#    ge = np.loadtxt('../runs/' + f + '/GlobalErrors.txt')
#    print(uniq_inp[:,i].T,ge)
#
#    loss = np.loadtxt('../runs/' + f +'/loss.dat')
#    ax1.loglog(loss[:,0], np.sum(loss[:,1:3],1), '--',linewidth=1,label=str(unique_inp[:,i])) # Training Eq.  Residual Losses
#    ax2.loglog(loss[:,0], np.sum(loss[:,idx_plot],1), '--',linewidth=1,label=str(unique_inp[:,i])) # Training BC Residual Losses
#
#axs[1].legend(); axs[0].legend()
#axs[1].grid(); axs[0].grid()
#axs[0].set_ylabel('Residual Loss', rotation=90, labelpad=0,fontsize=12)
#axs[1].set_ylabel('BC+Anchors Loss', rotation=90, labelpad=0,fontsize=12)
#plt.savefig('./' + 'unique_losses.png', bbox_inches='tight', pad_inches=0.04, dpi=600)
#plt.close()
