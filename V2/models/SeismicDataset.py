import numpy as np
import pandas as pd
import os
import pickle
import time
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from V2.config import WINDOW_SIZE, STEP_SIZE
from V2.preprocess import sliding_window
from V2.utility import load_train_data
from src.config import DATA_DIR
from tqdm import tqdm


class SeismicDataset(Dataset):
    def __init__(self, X_data, labels):
        self.X_data = X_data
        self.labels = labels

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        spectrogram = torch.tensor(self.X_data[idx], dtype=torch.float32)
        spectrogram = spectrogram.permute(2, 0, 1)  # Reshape from (129, 26, 1) to (1, 129, 26)

        label_start = torch.tensor(self.labels[idx], dtype=torch.float32)  # Start time label
        return spectrogram,  label_start


X_train, labels = load_train_data()
# Initialize the dataset and dataloader
batch_size = 32
train_dataset = SeismicDataset(X_train, labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

