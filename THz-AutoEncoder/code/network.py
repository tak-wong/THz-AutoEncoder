#   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
#
#   The Code is created based on the method described in the following paper 
#   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
#   German Conference on Pattern Recognition (GCPR), 2019.
#   
#   If you use this code in your scientific publication, please cite the mentioned paper.
#   The code and the algorithm are for non-comercial use only.

import torch as pt
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.LeReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()

        self.ConvLayer_ToOne = nn.Conv3d(2,10,(5,1,1), groups=2)
        self.ConvLayer_RealImag = nn.Conv3d(10,20,(10,1,1))
        self.ConvLayer_Both1 = nn.Conv3d(20,20,(15,1,1))
        self.ConvLayer_Both2 = nn.Conv3d(20,30,(31,1,1))

        self.BatchNorm10 = nn.BatchNorm3d(10)
        self.BatchNorm20_1 = nn.BatchNorm3d(20)
        self.BatchNorm20_2 = nn.BatchNorm3d(20)
        self.BatchNorm30 = nn.BatchNorm3d(30)
 
        self.FC_Last1 = nn.Linear(1020,4)

        self._initialize_weights()

    def forward(self, x):

        network = self.ConvLayer_ToOne(x)
        network = self.BatchNorm10(network)
        network = self.LeReLU(network)
        
        network = self.ConvLayer_RealImag(network)
        network = self.BatchNorm20_1(network)
        network = self.LeReLU(network)
        
        network = self.ConvLayer_Both1(network)
        network = self.BatchNorm20_2(network)
        network = self.LeReLU(network)

        network = self.ConvLayer_Both2(network)
        network = self.BatchNorm30(network)
        network = self.LeReLU(network)

        network = network.view(-1,1020)
        network = self.FC_Last1(network)

        network = self.LeReLU(network)

        return network

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.ConvLayer_ToOne.weight)
        nn.init.kaiming_uniform_(self.ConvLayer_RealImag.weight)
        nn.init.kaiming_uniform_(self.ConvLayer_Both1.weight)
        nn.init.kaiming_uniform_(self.ConvLayer_Both2.weight)
        nn.init.kaiming_uniform_(self.FC_Last1.weight)

        nn.init.constant_(self.ConvLayer_ToOne.bias, 0.0) 
        nn.init.constant_(self.ConvLayer_RealImag.bias, 0.0)
        nn.init.constant_(self.ConvLayer_Both1.bias, 0.0)
        nn.init.constant_(self.ConvLayer_Both2.bias, 0.0)
        nn.init.constant_(self.FC_Last1.bias, 0.0)
