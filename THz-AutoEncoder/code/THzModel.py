#   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
#
#   The Code is created based on the method described in the following paper 
#   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
#   German Conference on Pattern Recognition (GCPR), 2019.
#   
#   If you use this code in your scientific publication, please cite the mentioned paper.
#   The code and the algorithm are for non-comercial use only.

import numpy as np

os_rate = 9
window_width = os_rate * 5
period = 2 * os_rate
freq = 1 / period
omega = 2 * np.pi * freq
z_center = ( 1400 * os_rate ) / 2

Min = z_center-window_width
Max = z_center+window_width

z = np.linspace(Min, Max, num=2*window_width+1)

normAFactor = 2000
normZ = (z - Min) / os_rate
normOmega = omega * os_rate

#--------------
realChannel = 0
imagChannel = 1
valuesCurve = 91
countParam = 4


