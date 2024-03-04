import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import zarr
import s3fs
import logging
import tempfile
import os

logger = logging.getLogger("default")
readme_logger = logging.getLogger("readme")


class ECGDataset(Dataset):
    def __init__(self, data_path: str):
        print(f"Loading data from {data_path} ...")
        self.X = self.get_X_y_from_zarr(data_path)
        self.data_generator = self.Generator(self.X)
        
    def get_X_y_from_zarr(self, input_path):
        
        s3 = s3fs.S3FileSystem()
        directory_path = '/opt/ml/data'
        os.mkdir(directory_path)
        if os.path.exists(directory_path):
            print(f"Directory '{directory_path}' created successfully.")
        else:
            print(f"Failed to create directory '{directory_path}'.") 
        # tmpdirname = tempfile.TemporaryDirectory()
        # print(tmpdirname.name)
        if input_path.endswith("/"):
            input_path = input_path + "*"
        elif not input_path.endswith("/*"):
            input_path = input_path + "/*"
            
        logger.info("Started data copy from %s to %s", input_path, directory_path)
        s3.get(input_path, directory_path + "/", recursive = True)
        logger.info("Data copied from %s to %s", input_path, directory_path)

        X = zarr.open(directory_path, mode="r")
        print(X.shape[0])

        logger.info("X Info: %s, Shape - X: %s", X.info, X.shape)

        return X
    
    def Generator(self, X):
        batch_size = 10000
        for i in range(X.shape[0]//batch_size):
            start = i*batch_size
            end = start + + batch_size
            X_t = X[start:end]
            for sample in X_t:
                if np.isnan(sample).any():
                    continue
                yield sample

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = next(self.data_generator)
        # print(X[0][0])
        ecg_tensor = torch.from_numpy(X)
        img_tensor = ecg_tensor[None, :, :]
        mean = img_tensor.mean(dim=-1, keepdim=True)
        var = img_tensor.var(dim=-1, keepdim=True)
        img_tensor = (img_tensor - mean) / (var + 1.0e-6) ** 0.5
        return img_tensor, idx
