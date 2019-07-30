#   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
#
#   The Code is created based on the method described in the following paper 
#   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
#   German Conference on Pattern Recognition (GCPR), 2019.
#   
#   If you use this code in your scientific publication, please cite the mentioned paper.
#   The code and the algorithm are for non-comercial use only.

import numpy as np
import torch as pt
import torch.nn as nn
import scipy.stats as st
import math

import matplotlib.pyplot as plt

class curveLoss(nn.Module):
    def __init__(self):
        super(curveLoss, self).__init__()

    def forward(self,prediction,target):
        cost = pt.mean(pt.pow(prediction - target,2))   
        return cost

