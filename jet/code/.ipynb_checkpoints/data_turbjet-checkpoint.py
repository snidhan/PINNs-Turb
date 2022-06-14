import numpy as np
import struct as st
import h5py
from utils import *
import os
import matplotlib.pyplot as plt

def turbjet_read():

    with open('../../data_central/JetCopy.dat', 'r') as f:
        a = np.array([float(field) for field in f.read().split()])

    N  = 241*141

    Xg  = a[0:1*N]
    Xfull  = np.reshape(Xg,(241, 141),order='F').T

    Yg  = a[1*N:2*N]
    Yfull  = np.reshape(Yg,(241, 141),order='F').T

    Ug  = a[2*N:3*N]
    Ufull  = np.reshape(Ug,(241, 141),order='F').T

    Vg  = a[3*N:4*N]
    Vfull  = np.reshape(Vg,(241, 141),order='F').T

    Wg  = a[4*N:5*N]
    Wfull  = np.reshape(Wg,(241, 141),order='F').T

    uug = a[5*N:6*N]
    uufull = np.reshape(uug,(241, 141),order='F').T

    vvg = a[6*N:7*N]
    vvfull = np.reshape(vvg,(241, 141),order='F').T

    wwg = a[7*N:8*N]
    wwfull = np.reshape(wwg,(241, 141),order='F').T

    uvg = a[8*N:9*N]
    uvfull = np.reshape(uvg,(241, 141),order='F').T

    uwg = a[9*N:10*N]
    uwfull = np.reshape(uwg,(241, 141),order='F').T

    vwg = a[10*N:11*N]
    vwfull = np.reshape(vwg,(241, 141),order='F').T

    Xr  = Xfull[70:,:]
    Yr  = Yfull[70:,:]
    Ur  = Ufull[70:,:]
    Vr  = Vfull[70:,:]
    Wr  = Wfull[70:,:]
    uur = uufull[70:,:]
    vvr = vvfull[70:,:]
    wwr = wwfull[70:,:]
    uvr = uvfull[70:,:]
    uwr = uwfull[70:,:]
    vwr = vwfull[70:,:]

    X  = Xr[4:,:]
    Y  = Yr[4:,:]
    U  = Ur[4:,:];  Ustar  =  U.flatten()[:,None]
    V  = Vr[4:,:];  Vstar  =  V.flatten()[:,None]
    W  = Wr[4:,:];  Wstar  =  W.flatten()[:,None]
    uu = uur[4:,:]; uustar =  uu.flatten()[:,None]
    vv = vvr[4:,:]; vvstar =  vv.flatten()[:,None]
    ww = wwr[4:,:]; wwstar =  ww.flatten()[:,None]
    uv = uvr[4:,:]; uvstar =  uv.flatten()[:,None]
    uw = uwr[4:,:]; uwstar =  uw.flatten()[:,None]
    vw = vwr[4:,:]; vwstar =  vw.flatten()[:,None]

    xmin = X[0,0]; xmax = X[0,-1]
    ymin = Y[0,0]; ymax = Y[-1,0]

    Xgrid   = np.single(X[:,:])
    Ygrid   = np.single(Y[:,:])

    X_star = np.hstack((Xgrid.flatten()[:,None], Ygrid.flatten()[:,None]))

    x_coor = np.single(X[0,:])
    y_coor = np.single(Y[:,0])

    y          = np.zeros((np.size(Ustar,0), 6))
    y[:,0:1]   = Ustar
    y[:,1:2]   = Vstar
    y[:,2:3]   = uustar
    y[:,3:4]   = vvstar
    y[:,4:5]   = wwstar
    y[:,5:6]   = uvstar
    x          = X_star

    ygridded          = np.zeros((np.size(U,0),np.size(U,1),6))
    ygridded[:,:,0]   = U
    ygridded[:,:,1]   = V
    ygridded[:,:,2]   = uu
    ygridded[:,:,3]   = vv
    ygridded[:,:,4]   = ww
    ygridded[:,:,5]   = uv

    return xmin, xmax, ymin, ymax, x_coor, y_coor, x, y, ygridded

def turbjet_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor):
    ############# Postprocessing the output during the run itself ##################################

    Upred   = ys[0]*output_pred[:,0:1]
    Vpred   = ys[1]*output_pred[:,1:2]
    uupred  = ys[2]*output_pred[:,2:3]
    vvpred  = ys[3]*output_pred[:,3:4]
    wwpred  = ys[4]*output_pred[:,4:5]
    uvpred  = ys[5]*output_pred[:,5:6]
    Ppred   = ys[0]*ys[0]*output_pred[:,6:7]

    U       = ys[0]*Y[:,0:1]
    V       = ys[1]*Y[:,1:2]
    uu      = ys[2]*Y[:,2:3]
    vv      = ys[3]*Y[:,3:4]
    ww      = ys[4]*Y[:,4:5]
    uv      = ys[5]*Y[:,5:6]

    error_U  = np.linalg.norm(Upred-U,2)/np.linalg.norm(U,2)*100
    error_V  = np.linalg.norm(Vpred-V,2)/np.linalg.norm(V,2)*100
    error_uu = np.linalg.norm(uupred-uu,2)/np.linalg.norm(uu,2)*100
    error_vv = np.linalg.norm(vvpred-vv,2)/np.linalg.norm(vv,2)*100
    error_ww = np.linalg.norm(wwpred-ww,2)/np.linalg.norm(ww,2)*100
    error_uv = np.linalg.norm(np.abs(uvpred)-np.abs(uv),2)/np.linalg.norm(uv,2)*100

    Ugrid  =  np.reshape(Upred,   [np.size(y_coor,0), np.size(x_coor,0)])
    Vgrid  =  np.reshape(Vpred,   [np.size(y_coor,0), np.size(x_coor,0)])
    uugrid =  np.reshape(uupred,  [np.size(y_coor,0), np.size(x_coor,0)])
    vvgrid =  np.reshape(vvpred,  [np.size(y_coor,0), np.size(x_coor,0)])
    wwgrid =  np.reshape(wwpred,  [np.size(y_coor,0), np.size(x_coor,0)])
    uvgrid =  np.reshape(uvpred,  [np.size(y_coor,0), np.size(x_coor,0)])

    U       =  np.reshape(U,    [np.size(y_coor,0), np.size(x_coor,0)])
    V       =  np.reshape(V,    [np.size(y_coor,0), np.size(x_coor,0)])
    uu      =  np.reshape(uu,   [np.size(y_coor,0), np.size(x_coor,0)])
    vv      =  np.reshape(vv,   [np.size(y_coor,0), np.size(x_coor,0)])
    ww      =  np.reshape(ww,   [np.size(y_coor,0), np.size(x_coor,0)])
    uv      =  np.reshape(uv,   [np.size(y_coor,0), np.size(x_coor,0)])

    y_coor = y_coor*xs[1]

    path = './' + fold + '/'
    np.savetxt(path+'GlobalErrors.txt', [error_U, error_V, error_uu, error_vv, error_ww, error_uv])

    xloc = [1, 2, 5, 10, 15, 25]/xs[0]
    idx   = find_nearest2d(x_coor, xloc)
    idx = idx.astype(int)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor, U[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor, Ugrid[:,idx[count]], 'k--', linewidth=1)
            axs[l, m].set_title('U')
            axs[l, m].set_ylim((0, ys[0]))
            axs[l, m].set_xlim((0, 2))
            axs[l, m].set_xlabel(r'$r/D$',labelpad=0)
            axs[l, m].set_ylabel(r'$U/U_{j}$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"U_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor, V[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor, Vgrid[:,idx[count]], 'k--', linewidth=1)
            axs[l, m].set_title('V')
            axs[l, m].set_ylim((-ys[1]/2, ys[1]/2))
            axs[l, m].set_xlim((0, 2))
            axs[l, m].set_xlabel(r'$r/D$',labelpad=0)
            axs[l, m].set_ylabel(r'$V/U_{j}$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"V_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor, uu[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor, uugrid[:,idx[count]], 'k--', linewidth=1)
            axs[l, m].set_title('uu')
            axs[l, m].set_ylim((0, ys[2]))
            axs[l, m].set_xlim((0, 2))
            axs[l, m].set_xlabel(r'$r/D$',labelpad=0)
            axs[l, m].set_ylabel(r'$\langle uu \rangle/U_{j}U_{j}$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"uu_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor, vv[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor, vvgrid[:,idx[count]], 'k--', linewidth=1)
            axs[l, m].set_title('vv')
            axs[l, m].set_ylim((0, ys[3]))
            axs[l, m].set_xlim((0, 2))
            axs[l, m].set_xlabel(r'$r/D$',labelpad=0)
            axs[l, m].set_ylabel(r'$\langle vv \rangle/U_{j}U_{j}$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"vv_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor, ww[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor, wwgrid[:,idx[count]], 'k--', linewidth=1)
            axs[l, m].set_title('ww')
            axs[l, m].set_ylim((0, ys[4]))
            axs[l, m].set_xlim((0, 2))
            axs[l, m].set_xlabel(r'$r/D$',labelpad=0)
            axs[l, m].set_ylabel(r'$\langle ww \rangle/U_{j}U_{j}$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"ww_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor, uv[:,idx[count]], 'k-', linewidth=1)
            axs[l, m].plot(y_coor, uvgrid[:,idx[count]], 'k--', linewidth=1)
            axs[l, m].set_title('uv')
            axs[l, m].set_ylim((0, ys[5]))
            axs[l, m].set_xlim((0, 2))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$\langle -uv \rangle$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()
    fig.tight_layout()
    plt.savefig(path+"uv_comp.png")
    plt.close(fig)
