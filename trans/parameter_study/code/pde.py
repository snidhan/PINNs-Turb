import deepxde as dde
import numpy as np
import torch
import tensorflow.compat.v1 as tf
import scipy
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import KDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os

def d1(x, y, xs, ys, i, j):
    return (ys[i]/xs[j])*dde.grad.jacobian(y, x, i=i, j=j)

def d2(x, y, xs, ys, num, dnm1, dnm2):
    return (ys[num]/(xs[dnm1]*xs[dnm2])) * dde.grad.hessian(y, x, component=num, i=dnm1, j=dnm2)

def incom_pde(ndim, nvar, xs, ys, Re):
   def pde(x,y):

    print('xs ', xs)
    print('ys ', ys)

    #Name variables here
    u  = ys[0]*y[:,0:1]
    v  = ys[1]*y[:,1:2]
    p  = ys[2]*y[:,2:3]
    uu = ys[3]*y[:,3:4]
    uv = ys[4]*y[:,4:5]
    vv = ys[5]*y[:,5:6]

    drv = []
    #A lot of 1st derivatives are required
    for yid in range(nvar):
        for xid in range(ndim):
            print('xid, yid ', xid, yid)
            aux = (ys[yid]/xs[xid])*dde.grad.jacobian(y, x, i=yid, j=xid)
            drv.append(aux)
    print(len(drv))

    #Name derivatives here
    u_x,u_y,v_x,v_y,p_x,p_y,uu_x,uu_y,uv_x,uv_y,vv_x,vv_y = drv

    #Only a handful 2nd derivatives are required
    u_xx      = d2(x, y, xs, ys, 0, 0, 0)
    u_yy      = d2(x, y, xs, ys, 0, 1, 1)
    v_xx      = d2(x, y, xs, ys, 1, 0, 0)
    v_yy      = d2(x, y, xs, ys, 1, 1, 1)

    #Non-dimensionalized residuals
    l1 = u * u_x + v*u_y + p_x - (1/Re) * (u_xx + u_yy) + uu_x + uv_y
    l2 = u * v_x + v*v_y + p_y - (1/Re) * (v_xx + v_yy) + uv_x + vv_y
    l3 = u_x + v_y

    #Non-dimensionalized residuals
    return l1/ys[0],l2/ys[1],l3
   return pde
