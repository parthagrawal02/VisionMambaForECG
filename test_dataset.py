import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# import zarr
# import s3fs
import logging
import tempfile
import os

logger = logging.getLogger("default")
readme_logger = logging.getLogger("readme")

class ECGDataset(Dataset):
    def __init__(self, data_path: str):
        print(f"Loading data from {data_path} ...")
        self.X, self.Y = self.get_X_y_from_zarr(data_path)
        self.data_generator = self.Generator(self.X, self.Y)
        
    def get_X_y_from_zarr(self, input_path):
        X = np.random.rand(15000,12,1000)
        print(X.shape[0])
        return X, np.arange(0, 15000, 1)
    
    def Generator(self, X, Y):
        batch_size = 5000
        for i in range(X.shape[0]//batch_size):
            start = i*batch_size
            end = start + + batch_size
            X_t = X[start:end]
            y_t = Y[start:end]
            for sample, idz in zip(X_t, y_t):
                if np.isnan(sample).any():
                    continue
                yield sample, idz

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, idz = next(self.data_generator)
        # print(X[0][0])
        ecg_tensor = torch.from_numpy(X)
        idz = torch.from_numpy(idz)
        img_tensor = ecg_tensor[None, :, :]
        mean = img_tensor.mean(dim=-1, keepdim=True)
        var = img_tensor.var(dim=-1, keepdim=True)
        img_tensor = (img_tensor - mean) / (var + 1.0e-6) ** 0.5
        return img_tensor, idz