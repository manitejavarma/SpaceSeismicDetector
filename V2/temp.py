import os
import pickle

import pandas as pd

from obspy import read
import numpy as np

from V2.config import WINDOW_SIZE, STEP_SIZE
from V2.preprocess import sliding_window
from src.config import DATA_DIR

from tqdm import tqdm

def create_train_data_lunar():

    X_train = None
    labels = None
    # Load the catalog
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

            spec, labels_file = sliding_window(tr_data, tr_times, st[0].stats.sampling_rate, WINDOW_SIZE, STEP_SIZE)

            X_train_file = spec[..., np.newaxis]

            if X_train is None:
                X_train = X_train_file
            else:
                X_train = np.concatenate((X_train, X_train_file), axis=0)

            if labels is None:
                labels = labels_file
            else:
                labels = np.concatenate((labels, labels_file), axis=0)




    #save the training data
    save_train_data(X_train, labels)

def save_train_data(X_train, labels):
    # Save the training data and labels to a compressed NumPy file
    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    # Save all data to a single pickle file
    with open(os.path.join('./output/', f'train_data_lunar_{timestamp}.pkl'), 'wb') as f:
        pickle.dump((X_train, labels), f)

    print("Training data saved successfully to pickle file.")

def load_train_data():
    # Load the data from the pickle file
    with open(os.path.join('./output/', 'train_data.pkl'), 'rb') as f:
        X_train_loaded, event_labels_train_loaded, start_labels_train_loaded = pickle.load(f)

    # Check the loaded data
    print("Loaded X_train shape:", X_train_loaded.shape)
    print("Loaded event_labels_train shape:", event_labels_train_loaded.shape)
    print("Loaded start_labels_train shape:", start_labels_train_loaded.shape)


if __name__ == '__main__':
    create_train_data_lunar()


'''for cpu max use'''
import glob
import os
import pickle
import time
import multiprocessing
from functools import partial

from obspy import read
import numpy as np
from V2.config import WINDOW_SIZE, STEP_SIZE
from V2.preprocess import sliding_window
from src.config import DATA_DIR
from tqdm import tqdm
import pandas as pd


def process_file(row, data_directory):
    filename = row['filename']
    mseed_file = os.path.join(data_directory, f'{filename}.mseed')
    if os.path.exists(mseed_file):
        st = read(mseed_file)
        tr = st.traces[0].copy()
        tr_times = tr.times()
        tr_data = tr.data

        spec, labels_file = sliding_window(tr_data, tr_times, st[0].stats.sampling_rate, WINDOW_SIZE, STEP_SIZE)
        return spec[..., np.newaxis], labels_file
    return None, None


def create_train_data_lunar():
    X_train = None
    labels = None

    # Load the catalog using cuDF
    cat_directory = DATA_DIR + '/lunar/training/catalogs/'
    cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
    cat = pd.read_csv(cat_file)

    # Directory containing the training data
    data_directory = DATA_DIR + '/lunar/training/data/S12_GradeA/'

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Use partial to pass additional arguments to the worker function
    worker = partial(process_file, data_directory=data_directory)

    # Process files in parallel
    results = list(tqdm(pool.imap(worker, [row for _, row in cat.iterrows()]), total=len(cat)))

    pool.close()
    pool.join()

    for spec, labels_file in results:
        if spec is not None and labels_file is not None:
            if X_train is None:
                X_train = spec
            else:
                X_train = np.concatenate((X_train, spec), axis=0)

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


if __name__ == '__main__':
    create_train_data_lunar()

