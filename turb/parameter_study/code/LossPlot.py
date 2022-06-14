#!/usr/bin/env python
# coding: utf-8

# In[1]:


import deepxde as dde
import numpy as np
import torch
import scipy
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
from scipy.spatial import KDTree
import tensorflow.compat.v1 as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
from pde import *
from data_anchor import *
from xde import call_xde
import numpy as np
import sys
from multiprocessing import Process
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('mathtext', fontset='stix')
plt.rc('text', usetex=True)


# In[2]:


def p2s(p,sym): #param to string
    return sym.join(str(e) for e in p)


# In[42]:


inp = np.loadtxt('inp')

if inp.ndim == 1:
     inp = np.reshape(inp,(1,inp.shape[0]))

act = {1:"tanh", 2:"relu"}

lr, lyr, wid, actv, epch, n, lossw, nxm, nym, alpha, beta = inp.T

lr    = np.reshape(lr,   (len(lr),1))
lyr   = np.reshape(lyr,  (len(lr),1))
wid   = np.reshape(wid,  (len(lr),1))
actv  = np.reshape(actv, (len(lr),1))
epch  = np.reshape(epch, (len(lr),1))
n     = np.reshape(n,    (len(lr),1))
lossw = np.reshape(lossw,(len(lr),1))
nxm   = np.reshape(nxm,  (len(lr),1))
nym   = np.reshape(nym,  (len(lr),1))
alpha = np.reshape(alpha,(len(lr),1))
beta  = np.reshape(beta, (len(lr),1))

Re       = 800
ndim     = 2
nvar     = 6
Re_ar    = Re*np.ones((len(lr),1))
ndim_ar  = ndim*np.ones((len(lr),1))
nvar_ar  = nvar*np.ones((len(lr),1))

# nn_params   = [lr, int(lyr), int(wid), act[actv]]
data_params = np.hstack((Re_ar, nxm, nym, alpha, beta))         # All set of runs in data_params
xde_params  = np.hstack((ndim_ar, nvar_ar, (epch), (n), lossw)) # All set of runs in xde_params

xde_fix       = 0        # Fixing index for varying data_params plots
data_fix      = 0        # Fixing index for varying xde_params plots
idx_plot      = np.linspace(1,20,20).astype(int)  # Indices to sum over for loss plots

fold = 'Plots'
isExist = os.path.exists('./' + fold)
if not isExist:
    os.makedirs('./' + fold)


# In[43]:


## Plotting variation as changes in xde_params
# Fixing data_params at one set of values
dp_temp   = data_params[data_fix,:]
dp_temp   = [int(dp_temp[0]), int(dp_temp[1]), int(dp_temp[2]), dp_temp[3], dp_temp[4]]

fig = plt.figure(figsize=(5,5), dpi=600)
gsp = gridspec.GridSpec(1,1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4)

for i in range(0,np.size(xde_params,0)):
    
    xde_temp  = xde_params[i,:]
    xde_temp  = [int(xde_temp[0]), int(xde_temp[1]), int(xde_temp[2]), int(xde_temp[3]), xde_temp[4]]

    fold_name = p2s(xde_temp,'_') +'___'+ p2s(dp_temp,'_')
    print('Reading folder ', fold_name)
    loss = np.loadtxt(fold_name +'/loss.dat')
    plt.loglog(loss[:,0], np.sum(loss[:,idx_plot],1), '--',linewidth=1,label=str(xde_temp))

plt.legend()
plt.grid()
plt.ylabel(r'Loss', rotation=90, labelpad=0,fontsize=12)
plt.savefig('./' + fold + '/Loss' + '_xde_params.png', bbox_inches='tight', pad_inches=0.04, dpi=600)
plt.close()


# In[45]:


xde_temp   = xde_params[xde_fix,:]
xde_temp   = [int(xde_temp[0]), int(xde_temp[1]), int(xde_temp[2]), int(xde_temp[3]), xde_temp[4]]

fig = plt.figure(figsize=(5,5), dpi=600)
gsp = gridspec.GridSpec(1,1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4)

for i in range(0,np.size(data_params,0)):

    dp_temp  = data_params[i,:]
    dp_temp    = [int(dp_temp[0]), int(dp_temp[1]), int(dp_temp[2]), dp_temp[3], dp_temp[4]]

    fold_name = p2s(xde_temp,'_') +'___'+ p2s(dp_temp,'_')
    print('Reading folder ', fold_name)
    loss = np.loadtxt(fold_name +'/loss.dat')
    plt.loglog(loss[:,0], np.sum(loss[:,idx_plot],1), '--',linewidth=1,label=str(dp_temp))

plt.legend()
plt.grid()
plt.ylabel(r'Loss', rotation=90, labelpad=0,fontsize=12)
plt.savefig('./' + fold + '/Loss' + '_data_params.png', bbox_inches='tight', pad_inches=0.04, dpi=600)
plt.close()

