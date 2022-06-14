import numpy as np
import struct as st
import h5py
from utils import *
import os
import matplotlib.pyplot as plt

def turbBL_read(xmin_idx, xmax_idx):
    hf = h5py.File('./Transition_BL_Time_Averaged_Profiles.h5', 'r')

    x_coor1  = np.array(hf['x_coor'])
    y_coor1  = np.array(hf['y_coor'])
    y_coor1 = np.insert(y_coor1,0,0)   # Coordinate of the wall
    z_coor1  = np.array(hf['z_coor'])

    idx_xmin = xmin_idx
    idx_xmax = xmax_idx
    idx_ymin = 0
    idx_ymax = -1

    xmin = x_coor1[idx_xmin]
    xmin, idx_xmin = find_nearest1d(x_coor1, xmin)
    xmax = x_coor1[idx_xmax]
    xmax, idx_xmax = find_nearest1d(x_coor1, xmax)

    ymin = y_coor1[idx_ymin]
    ymin, idx_ymin = find_nearest1d(y_coor1, ymin)
    ymax = y_coor1[idx_ymax]
    ymax, idx_ymax = find_nearest1d(y_coor1, ymax)

    x_coor  = np.single(x_coor1[idx_xmin:idx_xmax+1]) - np.single(xmin)
    y_coor  = np.single(y_coor1[idx_ymin:idx_ymax+1]) - np.single(ymin)

    xmin = x_coor[0]
    xmax = x_coor[-1]

    ymin = y_coor[0]
    ymax = y_coor[-1]

    U        = np.array(hf['um'])
    uuprime  = np.array(hf['uum'] - np.multiply(U,U))
    V        = np.array(hf['vm'])
    vvprime  = np.array(hf['vvm'] - np.multiply(V,V))
    P        = np.array(hf['pm'])
    uvprime  = np.array(hf['uvm'] - np.multiply(U,V))

    U        = np.pad(U, ((1,0),(0,0)), 'constant')
    V        = np.pad(V, ((1,0),(0,0)), 'constant')
    P        = np.pad(P, ((1,0),(0,0)), 'edge')
    uuprime  = np.pad(uuprime, ((1,0),(0,0)), 'constant')
    vvprime  = np.pad(vvprime, ((1,0),(0,0)), 'constant')
    uvprime  = np.pad(uvprime, ((1,0),(0,0)), 'constant')

    U        = U[idx_ymin:idx_ymax+1,idx_xmin:idx_xmax+1]
    V        = V[idx_ymin:idx_ymax+1,idx_xmin:idx_xmax+1]
    P        = P[idx_ymin:idx_ymax+1,idx_xmin:idx_xmax+1]

    uuprime  = uuprime[idx_ymin:idx_ymax+1,idx_xmin:idx_xmax+1]
    uvprime  = uvprime[idx_ymin:idx_ymax+1,idx_xmin:idx_xmax+1]
    vvprime  = vvprime[idx_ymin:idx_ymax+1,idx_xmin:idx_xmax+1]

    Ustar    = U.flatten()[:,None]
    Vstar    = V.flatten()[:,None]
    Pstar    = P.flatten()[:,None]
    uustar   = uuprime.flatten()[:,None]
    vvstar   = vvprime.flatten()[:,None]
    uvstar   = uvprime.flatten()[:,None]

    Xgrid   = np.tile(x_coor,(y_coor.shape[0], 1))
    Ygrid   = np.tile(y_coor,(x_coor.shape[0], 1)).T
    X_star  = np.hstack((Xgrid.flatten()[:,None], Ygrid.flatten()[:,None]))

    y          = np.zeros((np.size(Ustar,0), 6))
    y[:,0:1]   = Ustar
    y[:,1:2]   = Vstar
    y[:,2:3]   = Pstar
    y[:,3:4]   = uustar
    y[:,4:5]   = uvstar
    y[:,5:6]   = vvstar
    x = X_star

    ygridded   = np.zeros((np.size(U,0),np.size(U,1),6))
    ygridded[:,:,0]   = U
    ygridded[:,:,1]   = V
    ygridded[:,:,2]   = P
    ygridded[:,:,3]   = uuprime
    ygridded[:,:,4]   = uvprime
    ygridded[:,:,5]   = vvprime

    return xmin, xmax, ymin, ymax, x_coor, y_coor, x, y, ygridded

def turbBL_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor):
    ############# Postprocessing the output during the run itself ##################################

    Upred   = ys[0]*output_pred[:,0:1]
    Vpred   = ys[1]*output_pred[:,1:2]
    Ppred   = ys[2]*output_pred[:,2:3]
    uupred  = ys[3]*output_pred[:,3:4]
    uvpred  = ys[4]*output_pred[:,4:5]
    vvpred  = ys[5]*output_pred[:,5:6]

    U       = ys[0]*Y[:,0:1]
    V       = ys[1]*Y[:,1:2]
    P       = ys[2]*Y[:,2:3]
    uu      = ys[3]*Y[:,3:4]
    uv      = ys[4]*Y[:,4:5]
    vv      = ys[5]*Y[:,5:6]

    error_U  = np.linalg.norm(Upred-U,2)/np.linalg.norm(U,2)*100
    error_V  = np.linalg.norm(Vpred-V,2)/np.linalg.norm(V,2)*100
    error_P  = np.linalg.norm(Ppred-P,2)/np.linalg.norm(P,2)*100
    error_uu = np.linalg.norm(uupred-uu,2)/np.linalg.norm(uu,2)*100
    error_vv = np.linalg.norm(vvpred-vv,2)/np.linalg.norm(vv,2)*100
    error_uv = np.linalg.norm(np.abs(uvpred)-np.abs(uv),2)/np.linalg.norm(uv,2)*100

    Ugrid  =  np.reshape(Upred,   [np.size(y_coor,0), np.size(x_coor,0)])
    Vgrid  =  np.reshape(Vpred,   [np.size(y_coor,0), np.size(x_coor,0)])
    Pgrid  =  np.reshape(Ppred,   [np.size(y_coor,0), np.size(x_coor,0)])
    uugrid =  np.reshape(uupred,  [np.size(y_coor,0), np.size(x_coor,0)])
    uvgrid =  np.reshape(uvpred,  [np.size(y_coor,0), np.size(x_coor,0)])
    vvgrid =  np.reshape(vvpred,  [np.size(y_coor,0), np.size(x_coor,0)])

    U       =  np.reshape(U,    [np.size(y_coor,0), np.size(x_coor,0)])
    V       =  np.reshape(V,    [np.size(y_coor,0), np.size(x_coor,0)])
    P       =  np.reshape(P,    [np.size(y_coor,0), np.size(x_coor,0)])
    uu      =  np.reshape(uu,   [np.size(y_coor,0), np.size(x_coor,0)])
    uv      =  np.reshape(uv,   [np.size(y_coor,0), np.size(x_coor,0)])
    vv      =  np.reshape(vv,   [np.size(y_coor,0), np.size(x_coor,0)])

    y_coor = y_coor*xs[1]
    Uinf   = 1

    Uaux = U/U[-1,:]*(1-U/U[-1,:])
    theta = np.trapz(Uaux, y_coor, axis=0)
    ReTheta = Uinf*theta/(1.25e-3)

    path = './' + fold + '/'
    np.savetxt(path+'GlobalErrors.txt', [error_U, error_V, error_P, error_uu, error_vv, error_uv])

    rethetaloc = [100, 200, 400, 600, 800, 1000]
    idx   = find_nearest2d(ReTheta, rethetaloc)
    idx = idx.astype(int)
    ThetaQuery = theta[idx]

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor/theta[idx[count]], U[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor/theta[idx[count]], Ugrid[:,idx[count]], 'k--', linewidth=1)
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
            axs[l, m].plot(y_coor/theta[idx[count]], V[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor/theta[idx[count]], Vgrid[:,idx[count]], 'k--', linewidth=1)
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
            axs[l, m].plot(y_coor/theta[idx[count]], uu[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor/theta[idx[count]], uugrid[:,idx[count]], 'k--', linewidth=1)
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
            axs[l, m].plot(y_coor/theta[idx[count]], vv[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor/theta[idx[count]], vvgrid[:,idx[count]], 'k--', linewidth=1)
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
            axs[l, m].plot(y_coor/theta[idx[count]], -uv[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor/theta[idx[count]], -uvgrid[:,idx[count]], 'k--', linewidth=1)
            axs[l, m].set_title('uv')
            axs[l, m].set_ylim((0, ys[4]))
            axs[l, m].set_xlim((0, 10))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$\langle -uv \rangle$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"uv_comp.png")
    plt.close(fig)
