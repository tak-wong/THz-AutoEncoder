#   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
#
#   The Code is created based on the method described in the following paper 
#   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
#   German Conference on Pattern Recognition (GCPR), 2019.
#   
#   If you use this code in your scientific publication, please cite the mentioned paper.
#   The code and the algorithm are for non-comercial use only.

from torch.utils.data import Dataset, DataLoader
import torch
import h5py
import time
import numpy as np
from numpy.fft import fft, fftshift


class THzImageDataset(Dataset):
    def __init__(self, path, is_preprocessed=False):
        super(THzImageDataset, self).__init__()

        with h5py.File(path, 'r') as hf:
            self.data = hf['data_complex_roi' if is_preprocessed else 'data_complex_all'][:]

        self.shape = self.data.shape
        self.is_preprocessed = is_preprocessed

    def __len__(self):
        return self.shape[0] * self.shape[1]

    def __getitem__(self, index):
        x = index % self.shape[0]
        y = index // self.shape[0]
        curve_data = self.data[x, y, :]

        if not self.is_preprocessed:
            curve_data = curve_data['real'] + 1j * curve_data['imag']

            # Take 91 samples from the mid
            ft = fftshift(fft(curve_data, 1400*9))
            mid = int(len(ft)/2)
            ft = ft[mid-45:mid+46]

            curve = np.ndarray(shape=(2, len(ft)), dtype=float)

            curve[0, :] = ft.real
            curve[1, :] = ft.imag

            curve = torch.from_numpy(curve).unsqueeze(2).unsqueeze(2)
            return curve

        else:
            curve = np.ndarray(shape=(2, len(curve_data)), dtype=float)
            curve[0, :] = curve_data['real']
            curve[1, :] = curve_data['imag']

            curve = torch.from_numpy(curve).unsqueeze(2).unsqueeze(2)
            return curve
