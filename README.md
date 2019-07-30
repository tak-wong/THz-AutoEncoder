# THz-AutoEncoder
Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction

## How to run
1. Install PyTorch
2. Extract all files to a folder
3. Open folder "thz-autoencoder"
3. Run "train.py" by PyTorch
4. Open the subfolder folder "thz-autoencoder/result/result-{time-stamp}", where {time-stamp} is the start time of the training
5. Run "predictMeasure.py" by PyTorch


## Result
full_epoch_final.pth: the trained autoencoder
epoch.txt: the epoch loss and information during training. Read by "readEpochFile.m" in MATLAB.
OutputMeasure_final.mat: the inference result in MATLAB format. Load by "loadOutput.m" in MATLAB.


## How to Cite
If you use this code in your scientific publication, please cite the paper

   **Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction**
     (T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller),
     In German Conference on Pattern Recognition (GCPR), 2019.

Copyright (C) Tak Ming Wong 2019, University of Siegen. All rights reserved.
