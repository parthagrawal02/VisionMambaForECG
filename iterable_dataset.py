import pandas as pd
import torch
from torch.utils.data import IterableDataset

class ParquetDataset(IterableDataset):
    def __init__(self, parquet_file, batch_size=32):
        super(ParquetDataset).__init__()
        self.parquet_file = parquet_file
        self.batch_size = batch_size

    def __iter__(self):
        df = pd.read_parquet(self.parquet_file, engine='pyarrow')
        while True:
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                # Convert the batch to PyTorch tensors
                X = torch.tensor(batch.drop('label', axis=1).values, dtype=torch.float32)
                y = torch.tensor(batch['label'].values, dtype=torch.long)
                yield X, y