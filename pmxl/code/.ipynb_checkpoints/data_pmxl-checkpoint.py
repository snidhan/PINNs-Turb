import numpy as np
import struct as st
import h5py
from utils import *
import os
import matplotlib.pyplot as plt

def pmxl_read():

    moments_pmxl = np.load('../../data_central/moments_pmxl.npz')
    U  = moments_pmxl['U'].T
    V  = moments_pmxl['V'].T
    uu = moments_pmxl['uu'].T
    uv = moments_pmxl['uv'].T
    vv = moments_pmxl['vv'].T
    
    grid_pmxl = np.load('../../data_central/grid_pmxl.npz')
    Xgrid = grid_pmxl['xgrid'].T
    Ygrid = grid_pmxl['ygrid'].T
    
    x_coor = Xgrid[0,:]
    y_coor = Ygrid[:,-1]

    xmin = np.min(Xgrid)
    xmax = np.max(Xgrid)

    ymin = np.min(Ygrid)
    ymax = np.max(Ygrid)
    
    Ustar    = U.flatten()[:,None]
    Vstar    = V.flatten()[:,None]
    uustar   = uu.flatten()[:,None]
    vvstar   = vv.flatten()[:,None]
    uvstar   = uv.flatten()[:,None]

    X_star  = np.hstack((Xgrid.flatten()[:,None], Ygrid.flatten()[:,None]))

    DeltaU = (41.50 - 22.41)
    y          = np.zeros((np.size(Ustar,0), 5))
    y[:,0:1]   = Ustar/DeltaU
    y[:,1:2]   = Vstar/DeltaU
    y[:,2:3]   = uustar/DeltaU**2
    y[:,3:4]   = uvstar/DeltaU**2
    y[:,4:5]   = vvstar/DeltaU**2
    x          = X_star

    ygridded          = np.zeros((np.size(U,0),np.size(U,1),5))
    ygridded[:,:,0]   = U
    ygridded[:,:,1]   = V
    ygridded[:,:,2]   = uuprime
    ygridded[:,:,3]   = uvprime
    ygridded[:,:,4]   = vvprime

    return xmin, xmax, ymin, ymax, x_coor, y_coor x, y, ygridded

def pmxl_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor):
    ############# Postprocessing the output during the run itself ##################################

    Upred   = ys[0]*output_pred[:,0:1]
    Vpred   = ys[1]*output_pred[:,1:2]
    uupred  = ys[2]*output_pred[:,2:3]
    uvpred  = ys[3]*output_pred[:,3:4]
    vvpred  = ys[4]*output_pred[:,4:5]
    Ppred   = ys[0]*ys[0]*output_pred[:,5:6]
    
    U       = ys[0]*Y[:,0:1]
    V       = ys[1]*Y[:,1:2]
    uu      = ys[2]*Y[:,2:3]
    uv      = ys[3]*Y[:,3:4]
    vv      = ys[4]*Y[:,4:5]
    
    error_U  = np.linalg.norm(Upred-U,2)/np.linalg.norm(U,2)*100
    error_V  = np.linalg.norm(Vpred-V,2)/np.linalg.norm(V,2)*100
    error_uu = np.linalg.norm(uupred-uu,2)/np.linalg.norm(uu,2)*100
    error_vv = np.linalg.norm(vvpred-vv,2)/np.linalg.norm(vv,2)*100
    error_uv = np.linalg.norm(np.abs(uvpred)-np.abs(uv),2)/np.linalg.norm(uv,2)*100
    
    path = './' + fold + '/'
    np.savetxt(path+'GlobalErrors.txt', [error_U, error_V, error_P, error_uu, error_vv, error_uv])
    
    Ugrid    =  np.reshape(Upred,   [np.size(y_coor,0), np.size(x_coor,0)])
    Vgrid    =  np.reshape(Vpred,   [np.size(y_coor,0), np.size(x_coor,0)])
    Pgrid    =  np.reshape(Ppred,   [np.size(y_coor,0), np.size(x_coor,0)])
    uugrid   =  np.reshape(uupred,  [np.size(y_coor,0), np.size(x_coor,0)])
    uvgrid   =  np.reshape(uvpred,  [np.size(y_coor,0), np.size(x_coor,0)])
    vvgrid   =  np.reshape(vvpred,  [np.size(y_coor,0), np.size(x_coor,0)])

    U        =  np.reshape(U,    [np.size(y_coor,0), np.size(x_coor,0)])
    V        =  np.reshape(V,    [np.size(y_coor,0), np.size(x_coor,0)])
    P        =  np.reshape(P,    [np.size(y_coor,0), np.size(x_coor,0)])
    uu       =  np.reshape(uu,   [np.size(y_coor,0), np.size(x_coor,0)])
    uv       =  np.reshape(uv,   [np.size(y_coor,0), np.size(x_coor,0)])
    vv       =  np.reshape(vv,   [np.size(y_coor,0), np.size(x_coor,0)])

    y_coor   = y_coor*xs[1]
    x_coor   = x_coor*xs[0]
    
    DeltaU   = (41.50) - (22.41)
 
    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor, U[:,idx[count]], 'k-', linewidth=1.5)
            axs[l, m].plot(, Ugrid[:,idx[count]], 'k--', linewidth=1.5)
            axs[l, m].set_title('U')
            axs[l, m].set_ylim((0, ys[0]))
            axs[l, m].set_xlim((0, 15))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$U$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"U_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor/theta[idx[count]], V[:,idx[count]], 'k-', linewidth=1.5)
            axs[l, m].plot(y_coor/theta[idx[count]], Vgrid[:,idx[count]], 'k--', linewidth=1.5)
            axs[l, m].set_title('V')
            axs[l, m].set_ylim((-ys[1]/2, ys[1]/2))
            axs[l, m].set_xlim((0, 15))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$V$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"V_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor/theta[idx[count]], uu[:,idx[count]], 'k-', linewidth=1.5)
            axs[l, m].plot(y_coor/theta[idx[count]], uugrid[:,idx[count]], 'k--', linewidth=1.5)
            axs[l, m].set_title('uu')
            axs[l, m].set_ylim((0, ys[3]))
            axs[l, m].set_xlim((0, 15))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$\langle uu \rangle$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"uu_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor/theta[idx[count]], vv[:,idx[count]], 'k-', linewidth=1.5)
            axs[l, m].plot(y_coor/theta[idx[count]], vvgrid[:,idx[count]], 'k--', linewidth=1.5)
            axs[l, m].set_title('vv')
            axs[l, m].set_ylim((0, ys[5]))
            axs[l, m].set_xlim((0, 10))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$\langle vv \rangle$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"vv_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor/theta[idx[count]], -uv[:,idx[count]], 'k-', linewidth=1.5)
            axs[l, m].plot(y_coor/theta[idx[count]], -uvgrid[:,idx[count]], 'k--', linewidth=1.5)
            axs[l, m].set_title('uv')
            axs[l, m].set_ylim((0, ys[4]))
            axs[l, m].set_xlim((0, 10))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$\langle -uv \rangle$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"uv_comp.png")
    plt.close(fig)
