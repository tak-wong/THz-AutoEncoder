# THz-AutoEncoder
_**Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction**_

_Abstract_
<p align="justify">Terahertz (THz) sensing is a promising imaging technology for a wide variety of different applications. Extracting the interpretable and physically meaningful parameters for such applications, however, requires solving an inverse problem in which a model function determined by these parameters needs to be fitted to the measured data. Since the underlying optimization problem is nonconvex and very costly to solve, we propose learning the prediction of suitable parameters from the measured data directly. More precisely, we develop a model-based autoencoder in which the encoder network predicts suitable parameters and the decoder is fixed to a physically meaningful model function, such that we can train the encoding network in an unsupervised way. We illustrate numerically that the resulting network is more than 140 times faster than classical optimization techniques while making predictions with only slightly higher objective values. Using such predictions as starting points of local optimization techniques allows us to converge to better local minima about twice as fast as optimizing without the network-based initialization.</p>

## How to download dataset
1. Download the source code
2. Extract the folder "THz-AutoEncoder"
3. Download dataset file from url https://www.cg.informatik.uni-siegen.de/data/THz-AutoEncoder/thz-dataset.zip
4. Extract the folder "thz-dataset" into the folder "THz-AutoEncoder"

## How to train
1. Install PyTorch
2. Open folder "code"
3. Run "train.py" by PyTorch

## How to infer after training
1. Open the subfolder folder "code/result/result-{time-stamp}", where {time-stamp} is the start time of the training
2. Run "predictMeasure.py" by PyTorch

## Result
full_epoch_final.pth: the trained autoencoder

epoch.txt: the epoch loss and information during training. Read by "readEpochFile.m" in MATLAB.

OutputMeasure_final.mat: the inference result in MATLAB format. Load by "loadOutput.m" in MATLAB.



## How to Cite
If you use this code in your scientific publication, please cite the paper

   **Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction**<br/>
     *Tak Ming Wong, Matthias Kahl, Peter Haring Bolívar, Andreas Kolb, Michael Möller,*<br/>
     In German Conference on Pattern Recognition (GCPR), 2019.
     
     @inproceedings{wong2019training,
      title={Training Auto-Encoder-Based Optimizers for Terahertz Image Reconstruction},
      author={Wong, Tak Ming and Kahl, Matthias and Haring-Bol{\'\i}var, Peter and Kolb, Andreas and M{\"o}ller, Michael},
      booktitle={German Conference on Pattern Recognition},
      pages={93--106},
      year={2019},
      organization={Springer}
    }
