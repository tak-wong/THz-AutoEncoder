#   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
#
#   The Code is created based on the method described in the following paper 
#   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
#   German Conference on Pattern Recognition (GCPR), 2019.
#   
#   If you use this code in your scientific publication, please cite the mentioned paper.
#   The code and the algorithm are for non-comercial use only.

import torch.utils.data as data
import torch as pt
from someNumbers import *
import numpy as np
from curveCreator import getCurves

import math

import h5py


class DatasetLoader(data.Dataset): 
    def __init__(self, cAmount, batchSize): 
        super(DatasetLoader, self).__init__()
        self.cAmount = cAmount
        self.batchSize = batchSize
        
    def __getitem__(self, index): 
        input,_,_ = getCurves(self.batchSize)
        return input.float()
  
    def __len__(self): 
          return self.cAmount
