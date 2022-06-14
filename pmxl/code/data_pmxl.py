import numpy as np
import struct as st
import h5py
from utils import *
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def pmxl_read():

    moments_pmxl = np.load('../../data_central/moments_pmxl.npz')
    U  = moments_pmxl['U']
    V  = moments_pmxl['V']
    uu = moments_pmxl['uu']
    uv = moments_pmxl['uv']
    vv = moments_pmxl['vv']

    grid_pmxl = np.load('../../data_central/grid_pmxl.npz')
    Xgrid = grid_pmxl['X']
    Ygrid = grid_pmxl['Y']

    x_coor = Xgrid[0,:]
    y_coor = Ygrid[:,-1]


    xmin = np.min(Xgrid)
    xmax = np.max(Xgrid)

    ymin = np.min(Ygrid)
    ymax = np.max(Ygrid)


    Ustar    = U.flatten()[:,None]
    Vstar    = V.flatten()[:,None]
    uustar   = uu.flatten()[:,None]
    uvstar   = uv.flatten()[:,None]
    vvstar   = vv.flatten()[:,None]

    X_star  = np.hstack((Xgrid.flatten()[:,None], Ygrid.flatten()[:,None]))

    DeltaU     = (41.50 - 22.41)
    y          = np.zeros((np.size(Ustar,0), 5))
    y[:,0:1]   = Ustar/(DeltaU)
    y[:,1:2]   = Vstar/(DeltaU)
    y[:,2:3]   = uustar/(DeltaU**2)
    y[:,3:4]   = uvstar/(DeltaU**2)
    y[:,4:5]   = vvstar/(DeltaU**2)
    x          = X_star

    ygridded          = np.zeros((np.size(U,0),np.size(U,1),5))
    ygridded[:,:,0]   = U
    ygridded[:,:,1]   = V
    ygridded[:,:,2]   = uu
    ygridded[:,:,3]   = uv
    ygridded[:,:,4]   = vv

    y_coor_gridded    = Ygrid

    return xmin, xmax, ymin, ymax, x_coor, y_coor, x, y, ygridded, y_coor_gridded

def pmxl_postproc(fold, X, Y, output_pred, xs, ys, x_coor, y_coor, y_coor_gridded):
    ############# Postprocessing the output during the run itself ##################################

    Upred   = ys[0]*output_pred[:,0:1]
    Vpred   = ys[1]*output_pred[:,1:2]
    uupred  = ys[2]*output_pred[:,2:3]
    uvpred  = ys[3]*output_pred[:,3:4]
    vvpred  = ys[4]*output_pred[:,4:5]

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
    np.savetxt(path+'GlobalErrors.txt', [error_U, error_V, error_uu, error_vv, error_uv])

    Ugrid    =  np.reshape(Upred,   [np.size(y_coor,0), np.size(x_coor,0)])
    Vgrid    =  np.reshape(Vpred,   [np.size(y_coor,0), np.size(x_coor,0)])
    uugrid   =  np.reshape(uupred,  [np.size(y_coor,0), np.size(x_coor,0)])
    uvgrid   =  np.reshape(uvpred,  [np.size(y_coor,0), np.size(x_coor,0)])
    vvgrid   =  np.reshape(vvpred,  [np.size(y_coor,0), np.size(x_coor,0)])

    U        =  np.reshape(U,    [np.size(y_coor,0), np.size(x_coor,0)])
    V        =  np.reshape(V,    [np.size(y_coor,0), np.size(x_coor,0)])
    uu       =  np.reshape(uu,   [np.size(y_coor,0), np.size(x_coor,0)])
    uv       =  np.reshape(uv,   [np.size(y_coor,0), np.size(x_coor,0)])
    vv       =  np.reshape(vv,   [np.size(y_coor,0), np.size(x_coor,0)])

    y_coor           = y_coor*xs[1]
    x_coor           = x_coor*xs[0]

    DeltaU   = (41.50) - (22.41)

    ## Plotting on top and bottom boundary

    fig = plt.figure(figsize=(4,2.2), dpi=200)
    gsp = gridspec.GridSpec(1,1)
    ax0 = plt.subplot(gsp[0,0])
    cs0 = plt.plot(x_coor, uugrid[-1,:], 'rs', markersize=1)
    plt.grid()
    plt.savefig('top_bc_uu.png')

    fig = plt.figure(figsize=(4,2.2), dpi=200)
    gsp = gridspec.GridSpec(1,1)
    ax0 = plt.subplot(gsp[0,0])
    cs0 = plt.plot(x_coor, uugrid[0,:], 'bs', markersize=1)
    plt.grid()
    plt.savefig('bottom_bc_uu.png')

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor_gridded[:,count], U[:, count], 'k-', linewidth=1.5)
            axs[l, m].plot(y_coor_gridded[:,count], Ugrid[:,count], 'k--', linewidth=1.5)
            axs[l, m].set_title('U')
            #axs[l, m].set_xlim((y_coor_gridded[0,count], y_coor_gridded[-1,count]))
            #axs[l, m].set_ylim((np.min(U[:,count]), np.max(U[:,count])))
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
            axs[l, m].plot(y_coor_gridded[:,count], V[:, count], 'k-', linewidth=1.5)
            axs[l, m].plot(y_coor_gridded[:,count], Vgrid[:,count], 'k--', linewidth=1.5)
            axs[l, m].set_title('V')
            #axs[l, m].set_xlim((y_coor_gridded[0,count], y_coor_gridded[-1,count]))
            #axs[l, m].set_ylim((np.min(V[:,count]), np.max(V[:,count])))
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
            axs[l, m].plot(y_coor_gridded[:,count], uu[:, count], 'k-', linewidth=1.5)
            axs[l, m].plot(y_coor_gridded[:,count], uugrid[:,count], 'k--', linewidth=1.5)
            axs[l, m].set_title('U')
            #axs[l, m].set_xlim((y_coor_gridded[0,count], y_coor_gridded[-1,count]))
            #axs[l, m].set_ylim((np.min(uu[:,count]), np.max(uu[:,count])))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$uu$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()

    fig.tight_layout()
    plt.savefig(path+"uu_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor_gridded[:,count], vv[:, count], 'k-', linewidth=1.5)
            axs[l, m].plot(y_coor_gridded[:,count], vvgrid[:,count], 'k--', linewidth=1.5)
            axs[l, m].set_title('vv')
            #axs[l, m].set_xlim((y_coor_gridded[0,count], y_coor_gridded[-1,count]))
            #axs[l, m].set_ylim((np.min(vv[:,count]), np.max(vv[:,count])))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$vv$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()

    fig.tight_layout()
    plt.savefig(path+"vv_comp.png")
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12,12), dpi=300)
    for l in range(0,axs.shape[0]):
        for m in range(0, axs.shape[1]):
            count = l*(axs.shape[0]-1) + m
            axs[l, m].plot(y_coor_gridded[:,count], uv[:, count], 'k-', linewidth=1.5)
            axs[l, m].plot(y_coor_gridded[:,count], uvgrid[:,count], 'k--', linewidth=1.5)
            axs[l, m].set_title('uv')
            #axs[l, m].set_xlim((y_coor_gridded[0,count], y_coor_gridded[-1,count]))
            #axs[l, m].set_ylim((np.min(uv[:,count]), np.max(uv[:,count])))
            axs[l, m].set_xlabel(r'$y/\theta$',labelpad=0)
            axs[l, m].set_ylabel(r'$uv$', rotation=90, labelpad=0,fontsize=10)
            axs[l, m].grid()

    fig.tight_layout()
    plt.savefig(path+"uv_comp.png")
    plt.close(fig)
