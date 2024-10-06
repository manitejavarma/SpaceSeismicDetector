import glob
import os
import pickle
import time

from obspy import read
import numpy as np
from V2.config import WINDOW_SIZE, STEP_SIZE
from V2.preprocess import sliding_window
from src.config import DATA_DIR
from tqdm import tqdm
import pandas as pd


def create_train_data_lunar():
    X_train = None
    labels = None

    # Load the catalog using cuDF
    cat_directory = DATA_DIR + '/lunar/training/catalogs/'
    cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
    cat = pd.read_csv(cat_file)

    # Directory containing the training data
    data_directory = DATA_DIR + '/lunar/training/data/S12_GradeA/'

    # Iterate through each file in the catalog
    for index, row in tqdm(cat.iterrows()):
        filename = row['filename']

        # Read the corresponding .mseed file
        mseed_file = os.path.join(data_directory, f'{filename}.mseed')
        if os.path.exists(mseed_file):
            st = read(mseed_file)
            tr = st.traces[0].copy()
            tr_times = tr.times()
            tr_data = tr.data

            spec, labels_file = sliding_window(tr_data, tr_times, st[0].stats.sampling_rate, WINDOW_SIZE,
                                               STEP_SIZE)

            X_train_file = spec[..., np.newaxis]

            if X_train is None:
                X_train = X_train_file
            else:
                X_train = np.concatenate((X_train, X_train_file), axis=0)

            if labels is None:
                labels = labels_file
            else:
                labels = np.concatenate((labels, labels_file), axis=0)

    # Save the training data
    save_train_data(X_train, labels)


def save_train_data(X_train, labels):
    # Save the training data and labels to a compressed NumPy file
    timestamp = int(time.time())
    # Save all data to a single pickle file
    with open(os.path.join('./output/', f'train_data_lunar_{timestamp}.pkl'), 'wb') as f:
        pickle.dump((X_train, labels), f)

    print("Training data saved successfully to pickle file.")


def get_latest_train_data(directory='./output/'):
    # List all files in the directory that match the pattern
    files = glob.glob(os.path.join(directory, 'train_data_lunar_*.pkl'))

    # Find the latest file based on the timestamp in the filename
    latest_file = max(files, key=os.path.getctime) if files else None
    print("File to be loaded:", latest_file)
    return latest_file


def load_train_data():
    # Load the data from the pickle file

    X_train_loaded, start_labels_train_loaded = pickle.load(open(get_latest_train_data(),
                                                                 'rb'))

    return X_train_loaded, start_labels_train_loaded


if __name__ == '__main__':
    # create_train_data_lunar()
    load_train_data()
