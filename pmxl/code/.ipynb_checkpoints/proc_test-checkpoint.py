import numpy as np
import os
#from pde import *
#from data_anchor import *
#from run_xde import run_xde
import numpy as np
import sys
import time
from multiprocessing import Process,current_process

inpt = np.genfromtxt('inp',dtype='str',max_rows=1)
inp  = np.loadtxt('inp',skiprows=2)

if inp.ndim == 1:
    inp = np.reshape(inp,(1,inp.shape[0]))
act = {1:"tanh", 2:"relu"}
lr, lyr, wid, actv, epch, n, lossw, nxm, nym, alpha, beta, bctype = inp.T

batch = int(sys.argv[1])
c0    = int(batch)*8
c1    = min([c0 + 8, lr.shape[0]])


def sit_there(c):
    if(c==1):
      time.sleep(0.1)
    rank = current_process()._identity[0]
    print(rank,c)

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
for c in range(c0,c1):
    xde_params  = [ndim, nvar, int(epch[c]), int(n[c]), lossw[c]]
    data_params = [Re, int(nxm[c]), int(nym[c]), alpha[c], beta[c], bctype[c]]
    nn_params   = [lr[c], int(lyr[c]), int(wid[c]), act[actv[c]]]

    fold_name.append(p2s(inpt, inp[c,:]))
    file_name   = 'file'
    #pr = Process(target=run_xde, args=(xde_params, data_params, nn_params, fold_name, file_name, c))
    pr = Process(target=sit_there, args=(c+1,))
    proc.append(pr); pr.start()
    print('some done')

for ip in proc:
    ip.join()

uniq_idx  = np.where(np.std(inp.T,axis=1)==0)
uniq_inp  = np.delete(inp.T,  uniq_idx, axis=0)
uniq_inpt = np.delete(inpt, uniq_idx)
print(uniq_inpt,'u v p uu uv vv')

idx_plot      = np.linspace(4,24,24-4+1).astype(int)  # Indices to sum over for loss plots

fig, axs = plt.subplots(1, 2, figsize=(14,7), dpi=600)
gsp = gridspec.GridSpec(1,1)

for i,f in enumerate(fold_name):
    ge = np.loadtxt('../runs/' + f + '/GlobalErrors.txt')
    print(uniq_inp[:,i].T,ge)

    loss = np.loadtxt('../runs/' + f +'/loss.dat')
    ax1.loglog(loss[:,0], np.sum(loss[:,1:3],1), '--',linewidth=1,label=str(unique_inp[:,i])) # Training Residual Losses
    ax2.loglog(loss[:,0], np.sum(loss[:,idx_plot],1), '--',linewidth=1,label=str(unique_inp[:,i])) # Training Residual Losses

axs[1].legend(); axs[0].legend()
axs[1].grid(); axs[0].grid()
axs[0].set_ylabel('Residual Loss', rotation=90, labelpad=0,fontsize=12)
axs[1].set_ylabel('BC+Anchors Loss', rotation=90, labelpad=0,fontsize=12)
plt.savefig('./' + 'unique_losses.png', bbox_inches='tight', pad_inches=0.04, dpi=600)
plt.close()
