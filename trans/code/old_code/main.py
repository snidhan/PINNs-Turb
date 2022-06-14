import numpy as np
import os
#from pde import *
#from data_anchor import *
#from run_xde import run_xde
import numpy as np
import sys
from multiprocessing import Process

inpt = np.genfromtxt('inp',dtype='str',max_rows=1)
inp  = np.loadtxt('inp',skiprows=2)

if inp.ndim == 1:
    inp = np.reshape(inp,(1,inp.shape[0]))
act = {1:"tanh", 2:"relu"}
lr, lyr, wid, actv, epch, n, lossw, nxm, nym, alpha, beta = inp.T

batch = int(sys.argv[1])
c0    = int(batch)*8
c1    = min([c0 + 8, lr.shape[0]])

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
for c in range(c0,c1):
    xde_params  = [ndim, nvar, int(epch[c]), int(n[c]), lossw[c]]
    data_params = [Re, int(nxm[c]), int(nym[c]), alpha[c], beta[c]]
    nn_params   = [lr[c], int(lyr[c]), int(wid[c]), act[actv[c]]]

    fold_name   = p2s(inpt, inp[c,:])
    file_name   = 'file'
    pr = Process(target=run_xde, args=(xde_params, data_params, nn_params, fold_name, file_name, c))
    proc.append(pr); pr.start()

for ip in proc:
    ip.join()
