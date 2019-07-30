#   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
#
#   The Code is created based on the method described in the following paper 
#   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
#   German Conference on Pattern Recognition (GCPR), 2019.
#   
#   If you use this code in your scientific publication, please cite the mentioned paper.
#   The code and the algorithm are for non-comercial use only.

import numpy as np
import matplotlib.pyplot as plt
import torch as pt
import math
import time
from datetime import datetime
from network import Net
from scaling import scaleUpAll
from GPUSelector import GPUInfo
from THzModel import *

import h5py
import hdf5storage
from numpy.fft import fft, fftshift


def predict():

    #------Settings---------------------------
    weights = "./full_epoch_final.pth"
    device = True
    #-----------------------------------------

    #------ Use GPU computation
    # gpuInfo = GPUInfo()
    # if device and not pt.cuda.is_available():
    #     raise Exception("No GPU found, please run without --cuda")
    # else:
    #     GPU = gpuInfo.getGPUs()
    # if device:
    #     pt.cuda.set_device(GPU)
    #     print("")
    #     print("Using --> GPU:{}".format(pt.cuda.current_device()))
    # else:
    #     print("Using --> CPU")
    #device = pt.device("cuda:0" if device else "cpu")

    #------ Use CPU computation
    device = pt.device("cpu")

    print("Open File")        
    f = h5py.File('../../../thz-dataset/MetalPCB_446x446x91.mat', 'r')
    d = f.keys()
    dset = f['data_complex_roi']

    x,y,z = dset.shape

    params = np.ndarray(shape=(x,y,countParam),dtype=float)
    params2 = np.ndarray(shape=(x,y,countParam),dtype=float)
    curves = np.ndarray(shape=(x,y,valuesCurve),dtype=complex)
    curvesPred = np.ndarray(shape=(x,y,valuesCurve),dtype=complex)

    print("Network setup:\n")
    net = Net()

    #------ save model weight 
    matfiledata = {}
    for name, param in net.named_parameters():
        matfiledata[name.replace('.','_')] = param.detach().numpy()
    hdf5storage.write(matfiledata, '.', 'Model.mat', matlab_compatible=True)

    # Load trained CNN weights
    print("\nLoading trained parameters from '%s'..."%weights)
    net = pt.load(weights, map_location=lambda storage, loc: storage)
    net = net.to(device)
    net = net.eval()
    print("Done!")
	
    learnedEpochs = weights.split('_')[-1]
    learnedEpochs = learnedEpochs.split('.')[0]

    predict_curve_time = time.time()

    t2 = 0
    FileContent = "y,StartTime,EndTime\n"
    print("y,StartTime,EndTime")

    for i in range(y):
        curveData = dset[0:x, i, 0:z]
        curveData = curveData['real'] + 1j*curveData['imag']

        t1 = time.time()

        curves[0:x,i,0:z] = curveData

        curve = np.ndarray( shape=(len(curveData[:,0]), len(curveData[0,:]), 2), dtype=float )

        curve[:,:,0] = curveData.real
        curve[:,:,1] = curveData.imag

        curve = pt.from_numpy(curve)
        curve = curve.float().permute(0,2,1)
        curve = curve.unsqueeze(3).unsqueeze(3)

        start_time = datetime.now()
        y_predict = net(curve.to(device))
        end_time = datetime.now()
        elapsed_time = end_time - start_time

        start_tstr = start_time.strftime("%Y/%m/%d:%H:%M:%S.%f")
        end_tstr = end_time.strftime("%Y/%m/%d:%H:%M:%S.%f")

        t2 = t2+(time.time()-t1)

        output_line = "{0}, {1}, {2}".format(str(i+1), start_tstr, end_tstr)
        print(output_line)
        FileContent = FileContent + output_line + "\n"

        params[:,i,:] = scaleUpAll(y_predict, False, device).cpu().detach()
        params2[:,i,:] = y_predict.cpu().detach()

    for i in range(x):
        for j in range(y):
            curvesPred[i,j,:] = math.fabs(params2[i,j,3]) * normAFactor * np.sinc(params2[i,j,0]*(normZ-params2[i,j,1])) * np.exp(-1j*(normOmega*normZ-2*np.pi*params2[i,j,2]))


    print(time.strftime("Finish at: %H:%M:%S", time.gmtime(time.time())))
    print(time.strftime("Inference Time:\t %H:%M:%S", time.gmtime(t2)))
    print(time.strftime("Overall Time:\t %H:%M:%S", time.gmtime(time.time()-predict_curve_time)))

    epochFile = open( "predict_measure_result.txt", "w" )
    epochFile.write(FileContent)
    epochFile.close()

    print("Writing mat File")
    matfiledata = {} # make a dictionary to store the MAT data in
    matfiledata[u'A'] = params[:,:,3]
    matfiledata[u'Sigma'] = params[:,:,0]
    matfiledata[u'Mu'] = params[:,:,1]
    matfiledata[u'Phi'] = params[:,:,2]
    matfiledata[u'curvesIn'] = curves
    matfiledata[u'curvesOut'] = curvesPred
    matfiledata[u'rmse'] = np.mean(np.power((curves.real - curvesPred.real),2) + np.power((curves.imag - curvesPred.imag),2))
    
    outputName = 'OutputMeasure_'+learnedEpochs+'.mat'
	
    hdf5storage.write(matfiledata, '.', outputName, matlab_compatible=True)

    print("Done")
predict()
