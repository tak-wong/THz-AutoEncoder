#   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
#
#   The Code is created based on the method described in the following paper 
#   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
#   German Conference on Pattern Recognition (GCPR), 2019.
#   
#   If you use this code in your scientific publication, please cite the mentioned paper.
#   The code and the algorithm are for non-comercial use only.

import torch as pt
import numpy as np
import math

from THzModel import *

def scaleUp(params):

    sigma = params[:,0].item()
    mu = params[:,1].item()
    phi = params[:,2].item()
    A = params[:,3].item()

    sigma = sigma / os_rate
    #mu = mu*os_rate+z_center
    mu = mu * os_rate + Min
    #phi = np.unwrap([np.pi,phi*2*math.pi],axis=0)[1]
    phi = phi*2*math.pi
    A = A * normAFactor

    return sigma, mu, phi, A

def scaleUpAll(params, grad=True, device='cpu'):

    B = pt.tensor([1/os_rate, os_rate, 2*math.pi, normAFactor], requires_grad=grad).to(device)
    #C = pt.tensor([0, z_center,0, 0],requires_grad=grad).to(device)
    C = pt.tensor([0, Min,0, 0],requires_grad=grad).to(device)

    B = B.unsqueeze(0)
    C = C.unsqueeze(0)

    params = pt.add(pt.mul(params,B),C)
    return params