import os

import numpy as np
import pandas as pd

from config import *
from models.sta_lta import sta_lta_model
from utilities.utilities import load_single_file
from datetime import datetime, timedelta


# run through all the seed files in the directory. lunar or Mars!
# run the algo on all the files and return the results into a csv file - look into the catalogue format

def get_mseed_files(directory):
    print(f"directory is {directory}")
    """
    Get all MiniSEED files in the specified directory.

    Parameters:
    directory (str): Path to the directory.

    Returns:
    list: List of MiniSEED file paths.
    """
    mseed_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.mseed')]
    return mseed_files


def get_train_data(target='lunar'):
    ending_dir = 'S12_GradeA/'
    if target == 'mars':
        ending_dir = ''
    train_data = get_mseed_files(f'{DATA_DIR}/{target}/training/data/{ending_dir}')
    return train_data


def detect_events(target='lunar'):
    get_train_data('lunar')

    # for each file in the train_data, load the file and run the algo
    # save the results in a csv file with the same name as the mseed file
    detection_times = []
    relative_detection_times = []
    fnames = []
    for file in get_train_data('lunar'):
        stream_file = load_single_file(file, file_type="mseed")
        tr = stream_file.traces[0].copy()
        tr_times = tr.times()
        tr_data = tr.data
        on_off = sta_lta_model(sta_len=120, lta_len=600, stream_file=stream_file, plot=False)

        starttime = tr.stats.starttime.datetime

        # Iterate through detection times and compile them

        for i in np.arange(0, len(on_off)):
            triggers = on_off[i]
            relative_time = tr_times[triggers[0]]
            relative_detection_times.append(relative_time)
            on_time = starttime + timedelta(seconds=relative_time)
            on_time_str = datetime.strftime(on_time, '%Y-%m-%dT%H:%M:%S.%f')
            detection_times.append(on_time_str)
            fnames.append(file)

        # Compile dataframe of detections
    detect_df = pd.DataFrame(data={'filename': [os.path.basename(file) for file in fnames],
                                   'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times,
                                   'time_rel(sec)': relative_detection_times})

    # save dataframe to csv
    detect_df.to_csv('../input/output.csv', index=False)


if __name__ == "__main__":
    detect_events(target='lunar')
