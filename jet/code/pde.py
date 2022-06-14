import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf

def d1(x, y, xs, ys, i, j):
    return (ys[i]/xs[j])*dde.grad.jacobian(y, x, i=i, j=j)

def d2(x, y, xs, ys, num, dnm1, dnm2):
    return (ys[num]/(xs[dnm1]*xs[dnm2])) * dde.grad.hessian(y, x, component=num, i=dnm1, j=dnm2)

def incom_pde(ndim, nvar, xs, ys_temp, Re):
    def pde(x,y):

        ys = np.append(ys_temp, ys_temp[0]*ys_temp[0]) # Pressure scaling for the PDEs

        #Name variables here
        u  = ys[0]*y[:,0]
        v  = ys[1]*y[:,1]
        uu = ys[2]*y[:,2]
        vv = ys[3]*y[:,3]
        ww = ys[4]*y[:,4]
        uv = ys[5]*y[:,5]
        p  = ys[6]*y[:,6]
        r  = x[:,1:2]

        drv = []
        #A lot of 1st derivatives are required
        for ii in np.ndindex((nvar,ndim)):
            drv.append(d1(x, y, xs, ys, ii[0], ii[1]))

        #Name derivatives here
        u_x,u_r, v_x,v_r, uu_x,uu_r, vv_x,vv_r, ww_x,ww_r, uv_x,uv_r, p_x,p_r = drv

        #Only a handful 2nd derivatives are required
        u_xx      = d2(x, y, xs, ys, 0, 0, 0)
        u_rr      = d2(x, y, xs, ys, 0, 1, 1)
        v_xx      = d2(x, y, xs, ys, 1, 0, 0)
        v_rr      = d2(x, y, xs, ys, 1, 1, 1)

        #Non-dimensionalized residuals
        l1 = u * u_x + v * u_r + p_x - (1/Re) * (u_xx + u_rr + u_r/r)            + (uu_x + uv_r + uv/r)
        l2 = u * v_x + v * v_r + p_r - (1/Re) * (v_xx + v_rr + v_r/r - v/(r**2)) + (uv_x + vv_r + vv/r - ww/r)
        l3 = u_x + v_r + v/r
        #Non-dimensionalized residuals
        return l1,l2,l3
    return pde
