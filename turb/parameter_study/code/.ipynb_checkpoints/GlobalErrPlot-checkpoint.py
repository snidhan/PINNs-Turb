#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
from scipy.spatial import KDTree
import tensorflow.compat.v1 as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
import sys
from multiprocessing import Process
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('mathtext', fontset='stix')
plt.rc('text', usetex=True)


# In[2]:

def p2s(p,sym): #param to string
    return sym.join(str(e) for e in p)

# In[3]:

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

xde_fix  = 0        # Fixing index for varying data_params plots
data_fix = 0        # Fixing index for varying xde_params plots

fold = 'Plots'
isExist = os.path.exists('./' + fold)
if not isExist:
    os.makedirs('./' + fold)


# In[4]:


string = ['U', 'V', 'P', 'uu', 'uv', 'vv']

################################################################# Plotting variation as changes in xde_params
# Fixing data_params at one set of values
dp_temp   = data_params[data_fix,:]
dp_temp   = [int(dp_temp[0]), int(dp_temp[1]), int(dp_temp[2]), dp_temp[3], dp_temp[4]]

global_error = np.zeros((np.size(data_params,0),nvar))

for i in range(0,np.size(xde_params,0)):

    xde_temp  = xde_params[i,:]
    xde_temp  = [int(xde_temp[0]), int(xde_temp[1]), int(xde_temp[2]), int(xde_temp[3]), xde_temp[4]]

    fold_name = p2s(xde_temp,'_') +'___'+ p2s(dp_temp,'_')
    print('Reading folder ', fold_name)
    global_error_temp = np.loadtxt(fold_name +'/GlobalErrors.txt')
    global_error[:,i] = global_error_temp

for i in range(0,nvar):
    fig = plt.figure(figsize=(5,5), dpi=600)
    gsp = gridspec.GridSpec(1,1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.4)
    ax = plt.subplot(gsp[0,0])
    plt.plot(np.linspace(1,len(lr), len(lr)), global_error[i,:],'ks')
    ax.set_ylim(np.min(global_error[i,:])/2,1.05*np.max(global_error[i,:]))
    ax.set_xlim(1, len(lr))
    plt.grid()

    ax2=ax.twiny()
    ax2.set_xscale("linear")
    # sns.despine(ax=ax2,top=True, right=False, left=False, bottom=False, offset=0)
    ax2.spines['top'].set_color('tab:red')
    ax2.tick_params(axis='x',which='major', colors='tab:red')
    xt2 = xde_params[:,2]
    ax2.set_xticklabels((xde_params[:,2]))
    # ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel('Epochs')

    ax3=ax.twiny()
    ax3.set_xscale("linear")
    sns.despine(ax=ax3,top=False, right=True, left=True, bottom=True, offset=50)
    ax3.spines['top'].set_color('#2166AC')
    ax3.tick_params(axis='x',which='major', colors='#2166AC')
    xt3 = xde_params[:,3]
    ax3.set_xticklabels((xde_params[:,3]))
    # ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.set_xlabel('N')

    ax4=ax.twiny()
    ax4.set_xscale("linear")
    sns.despine(ax=ax4,top=False, right=True, left=True, bottom=True, offset=100)
    ax4.spines['top'].set_color('k')
    ax4.tick_params(axis='x',which='major', colors='k')
    xt4 = xde_params[:,4]
    ax4.set_xticklabels((xde_params[:,4]))
    # ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax4.set_xlabel('$w$')
    ax.set_ylabel(r'Error \ '+ string[i] + '(in \%)', rotation=90, labelpad=0,fontsize=12)
    plt.savefig('./' + fold + '/Error_' + string[i] + '_xde_params.png', bbox_inches='tight', pad_inches=0.04, dpi=600)
    plt.close()


# In[5]:


################################################################# Plotting variation as changes in data_params
# Fixing xde_params at one set of values

xde_temp   = xde_params[xde_fix,:]
xde_temp   = [int(xde_temp[0]), int(xde_temp[1]), int(xde_temp[2]), int(xde_temp[3]), xde_temp[4]]

global_error = np.zeros((np.size(data_params,0),nvar))

for i in range(0,np.size(data_params,0)):

    dp_temp  = data_params[i,:]
    dp_temp    = [int(dp_temp[0]), int(dp_temp[1]), int(dp_temp[2]), dp_temp[3], dp_temp[4]]

    fold_name = p2s(xde_temp,'_') +'___'+ p2s(dp_temp,'_')
    print('Reading folder ', fold_name)
    global_error_temp = np.loadtxt(fold_name +'/GlobalErrors.txt')
    global_error[:,i] = global_error_temp

for i in range(0,nvar):
    fig = plt.figure(figsize=(5,5), dpi=600)
    gsp = gridspec.GridSpec(1,1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.4)
    ax = plt.subplot(gsp[0,0])
    plt.plot(np.linspace(1,len(lr), len(lr)), global_error[i,:],'ks')
    ax.set_ylim(np.min(global_error[i,:])/2,1.05*np.max(global_error[i,:]))
    ax.set_xlim(1, len(lr))
    plt.grid()

    ax2=ax.twiny()
    ax2.set_xscale("linear")
    sns.despine(ax=ax2,top=True, right=False, left=False, bottom=False, offset=0)
    ax2.spines['top'].set_color('tab:red')
    ax2.tick_params(axis='x',which='major', colors='tab:red')
    xt2 = data_params[:,1]
    ax2.set_xticklabels((data_params[:,1]))
    # ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel('$Nx_m$')

    ax3=ax.twiny()
    ax3.set_xscale("linear")
    sns.despine(ax=ax3,top=False, right=True, left=True, bottom=True, offset=50)
    ax3.spines['top'].set_color('#2166AC')
    ax3.tick_params(axis='x',which='major', colors='#2166AC')
    xt3 = data_params[:,2]
    ax3.set_xticklabels((data_params[:,2]))
    # ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.set_xlabel('$Ny_m$')

    ax4=ax.twiny()
    ax4.set_xscale("linear")
    sns.despine(ax=ax4,top=False, right=True, left=True, bottom=True, offset=100)
    ax4.spines['top'].set_color('k')
    ax4.tick_params(axis='x',which='major', colors='k')
    xt4 = data_params[:,3]
    ax4.set_xticklabels((data_params[:,3]))
    # ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax4.set_xlabel(r'$\alpha$')

    ax5=ax.twiny()
    ax5.set_xscale("linear")
    sns.despine(ax=ax5,top=False, right=True, left=True, bottom=True, offset=150)
    ax5.spines['top'].set_color('k')
    ax5.tick_params(axis='x',which='major', colors='k')
    xt5 = data_params[:,4]
    ax5.set_xticklabels((data_params[:,4]))
    # ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax5.set_xlabel(r'$\beta$')

    ax.set_ylabel(r'Error '+ string[i] + '(in \%)', rotation=90, labelpad=0,fontsize=12)
    plt.savefig('./' + fold + '/Error_' + string[i] + '_data_params.png', bbox_inches='tight', pad_inches=0.04, dpi=600)
    plt.close()

