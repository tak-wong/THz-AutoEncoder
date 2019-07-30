# THz-AutoEncoder
Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction

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

   **Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction**
     (T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller),
     In German Conference on Pattern Recognition (GCPR), 2019.
