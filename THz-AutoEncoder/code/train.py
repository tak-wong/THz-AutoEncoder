#   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
#
#   The Code is created based on the method described in the following paper 
#   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
#   German Conference on Pattern Recognition (GCPR), 2019.
#   
#   If you use this code in your scientific publication, please cite the mentioned paper.
#   The code and the algorithm are for non-comercial use only.

from data import THzImageDataset
from network import Net

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import time
import math
import datetime
import shutil
import os

from LossFunctions import curveLoss
from THzModel import *
from datetime import datetime

EPOCHS = 1200
DEVICE = True  # Enable GPU usage
BATCH_SIZE_TRAIN = 64 * 64
BATCH_SIZE_VALID = 64 * 64
LEARNING_RATE = 0.005
OS_RATE = 9
WINDOW_WIDTH = OS_RATE * 5
OMEGA = 2 * math.pi * 0.5 * (1/OS_RATE)
NORM_OMEGA = OMEGA * os_rate
Z_CENTER = (1400 * OS_RATE) / 2
MIN = Z_CENTER - WINDOW_WIDTH
MAX = Z_CENTER + WINDOW_WIDTH

def checkpoint(net, epoch, folder="./"):
    model_out_path = "{}full_epoch_{}.pth".format(folder,epoch)
    torch.save(net, model_out_path)

def train():

    # Generate time string
    tobj = datetime.now()
    tstr = tobj.strftime("%Y%m%d_%H%M%S") 

    # Create a sub-directory under output directory for the result on a run
    folder = "./result/result-{}/".format(tstr)
    os.makedirs(folder, exist_ok=True)
    shutil.copy2('./scaling.py', folder)
    shutil.copy2('./THzModel.py', folder)
    shutil.copy2('./GPUSelector.py', folder)
    shutil.copy2('./network.py', folder)
    shutil.copy2('./predictMeasure.py', folder)
    shutil.copy2('./readEpochFile.m', folder)
    shutil.copy2('./loadOutput.m', folder)

    # This dataset is preprocessed and reduced to 91 samples in the middle
    dataset = THzImageDataset('../thz-dataset/MetalPCB_446x446x91.mat', is_preprocessed=True)

    # Prepare training set and validation set
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_VALID, shuffle=True)

    # Select the most idle gpu, instanciate network, select optimizer and loss function
    device = torch.device("cuda" if DEVICE else "cpu")  # TODO: Select idle gpu
    net = Net()
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.99)
    criterion_real = curveLoss().to(device)
    criterion_imag = curveLoss().to(device)

    # Prepare epoch output file
    epochFileContent = "Epoch,LR,LossTrain,LossValid,StartTime,EndTime\n"
    print("Epoch,LR,LossTrain,LossValid,StartTime,EndTime")

    epoch_loss_threshold = 600
    epoch = 0
    epoch_loss_train = 99999999
    epoch_loss_valid = 99999999

    # Main training loop
    while (epoch <= EPOCHS) & (epoch_loss_train >= epoch_loss_threshold) & (epoch_loss_valid >= epoch_loss_threshold):
        epoch = epoch + 1

        start_epoch_time = datetime.now()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        batch_number = 0
        epoch_loss_train = 0
        scheduler.step()

        # Training
        for train_batch in train_loader:
            # Encoding
            train_batch = train_batch.to(device=device, dtype=torch.float)

            optimizer.zero_grad()
            prediction_train = net(train_batch)

            # Decoding
            # Model formula: Amp * sinc(sigma * (z - mu)) * e^(-j(omega*z - phi)
            normSigma = prediction_train[:, 0].unsqueeze(1)
            normMu = prediction_train[:, 1].unsqueeze(1)
            normPhi = prediction_train[:, 2].unsqueeze(1)
            normAmp = prediction_train[:, 3].unsqueeze(1)
            s = normAmp.size()[0]

            z = np.linspace(MIN, MAX, num=2 * WINDOW_WIDTH + 1)
            nz = (z - MIN) / os_rate
            nz = np.tile(nz, (s, 1))
            nz = torch.tensor(nz, requires_grad=True).float().to(device)

            norm_pi_sigma_mu = math.pi * normSigma * (nz - normMu)
            norm_sin_pi_sigma_mu = torch.sin( math.pi * normSigma * (nz - normMu) )

            norm_pi_sigma_mu[norm_pi_sigma_mu == 0] = 1
            norm_sin_pi_sigma_mu[norm_pi_sigma_mu == 0] = 1

            # Get loss
            yReal =      torch.abs(normAmp) * normAFactor * (norm_sin_pi_sigma_mu) / (norm_pi_sigma_mu) * torch.cos(NORM_OMEGA * nz - 2 * math.pi * normPhi)
            yImag = -1 * torch.abs(normAmp) * normAFactor * (norm_sin_pi_sigma_mu) / (norm_pi_sigma_mu) * torch.sin(NORM_OMEGA * nz - 2 * math.pi * normPhi)

            lossReal = criterion_real(yReal, train_batch[0:BATCH_SIZE_TRAIN, realChannel, 0:valuesCurve, :, :].squeeze())
            lossImag = criterion_imag(yImag, train_batch[0:BATCH_SIZE_TRAIN, imagChannel, 0:valuesCurve, :, :].squeeze())

            loss_train = lossReal + lossImag
            epoch_loss_train += loss_train.item()

            loss_train.backward()
            optimizer.step()

            batch_number += 1

        # Record the training time
        end_epoch_time = datetime.now()

        ## Validation
        epoch_loss_valid = 0
        for valid_batch in valid_loader:
            # Encoding
            valid_batch = valid_batch.to(device=device, dtype=torch.float)

            # Decoding
            prediction_valid = net(valid_batch)
            normSigma = prediction_valid[:, 0].unsqueeze(1)
            normMu = prediction_valid[:, 1].unsqueeze(1)
            normPhi = prediction_valid[:, 2].unsqueeze(1)
            normAmp = prediction_valid[:, 3].unsqueeze(1)
            s = normAmp.size()[0]

            z = np.linspace(MIN, MAX, num=2 * WINDOW_WIDTH + 1)
            nz = (z - MIN) / os_rate
            nz = np.tile(nz, (s, 1))
            nz = torch.tensor(nz, requires_grad=True).float().to(device)

            norm_pi_sigma_mu = math.pi * normSigma * (nz - normMu)
            norm_sin_pi_sigma_mu = torch.sin( math.pi * normSigma * (nz - normMu) )

            norm_pi_sigma_mu[norm_pi_sigma_mu == 0] = 1
            norm_sin_pi_sigma_mu[norm_pi_sigma_mu == 0] = 1

            # Get loss
            yReal =      torch.abs(normAmp) * normAFactor * (norm_sin_pi_sigma_mu) / (norm_pi_sigma_mu) * torch.cos(NORM_OMEGA * nz - 2 * math.pi * normPhi)
            yImag = -1 * torch.abs(normAmp) * normAFactor * (norm_sin_pi_sigma_mu) / (norm_pi_sigma_mu) * torch.sin(NORM_OMEGA * nz - 2 * math.pi * normPhi)

            lossReal = criterion_real(yReal, valid_batch[0:BATCH_SIZE_VALID, realChannel, 0:valuesCurve, :, :].squeeze())
            lossImag = criterion_imag(yImag, valid_batch[0:BATCH_SIZE_VALID, imagChannel, 0:valuesCurve, :, :].squeeze())

            loss_valid = lossReal + lossImag
            epoch_loss_valid += loss_valid.item()
        
        # Get the epoch time
        start_epoch_tstr = start_epoch_time.strftime("%Y/%m/%d:%H:%M:%S.%f")
        end_epoch_tstr = end_epoch_time.strftime("%Y/%m/%d:%H:%M:%S.%f")

        # Get the epoch loss
        epoch_loss_avg_train = float(epoch_loss_train) / float(len(train_loader))
        epoch_loss_avg_valid = float(epoch_loss_valid) / float(len(valid_loader))

        output_line = "{0},{1:12.10f},{2:12.4f},{3:12.4f},{4},{5}".format(epoch, lr, epoch_loss_avg_train, epoch_loss_avg_valid, start_epoch_tstr, end_epoch_tstr)
        print(output_line)

        epochFileContent = epochFileContent + output_line + "\n"

        if (epoch % 50 == 0):
            checkpoint(net, epoch, folder)
    
    checkpoint(net, "final", folder)

    epochFile = open( folder + "epoch.txt", "w" )
    epochFile.write(epochFileContent)
    epochFile.close()

def test():
    pass


if __name__ == "__main__":
    train()


