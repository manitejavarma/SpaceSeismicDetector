import os

import numpy as np
import pandas as pd

from config import *
from models.sta_lta import sta_lta_model
from utilities.utilities import *
from datetime import datetime, timedelta

from models.isolation_forest_detector import isolation_forest_detector


def get_data(target='lunar', type="training"):
    data = get_mseed_files(f'{DATA_DIR}\{target}\{type}\data')
    return data


def detect_events(target='lunar', plot=True, type="training", model = "sta_lta"):
    # for each file in the train_data, load the file and run the algo
    # save the results in a csv file with the same name as the mseed file
    detection_times = []
    relative_detection_times = []
    fnames = []
    for file in get_data(target, type):
        stream_file = load_single_file(file, file_type="mseed")
        tr = stream_file.traces[0].copy()
        tr_times = tr.times()
        tr_data = tr.data
        if model == "sta_lta":
            events = sta_lta_model(sta_len=120, lta_len=600, stream_file=stream_file, plot=plot)
        elif model == "isolation_forest":
            events = isolation_forest_detector(tr_data, tr_times, plot=plot)

        starttime = tr.stats.starttime.datetime

        # Iterate through detection times and compile them

        for i in np.arange(0, len(events)):
            event = events[i]
            relative_time = tr_times[event]
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
    detect_df.to_csv(f'../input/{model}-{target}-output.csv', index=False)
    print("Exported to output.csv")


if __name__ == "__main__":
    detect_events(target='lunar', plot=True, type="training", model="sta_lta")
