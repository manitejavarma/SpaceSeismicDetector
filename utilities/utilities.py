import numpy as np
from obspy.signal.trigger import trigger_onset
import matplotlib.pyplot as plt
import pandas as pd
from obspy import read
import os


def plot_characteristic_function(stream_file, cft):
    trace = stream_file.traces[0].copy()
    tr_times = trace.times()

    # Plot characteristic function
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(tr_times, cft)
    ax.set_xlim([min(tr_times), max(tr_times)])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Characteristic function')
    plt.savefig('characteristic_function.png')
    plt.show()


def plot_seismic_event_miniseed(cft, stream_file, on, off, on_off):
    trace = stream_file.traces[0].copy()
    tr_times = trace.times()
    tr_data = trace.data
    # The first column contains the indices where the trigger is turned "on".
    # The second column contains the indices where the trigger is turned "off".

    # Plot on and off triggers
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    for i in np.arange(0, len(on_off)):
        triggers = on_off[i]
        ax.axvline(x=tr_times[triggers[0]], color='red', label='Trig. On')
        ax.axvline(x=tr_times[triggers[1]], color='purple', label='Trig. Off')

    # Plot seismogram
    ax.plot(tr_times, tr_data)
    ax.set_xlim([min(tr_times), max(tr_times)])
    ax.legend()
    plt.savefig('seismic_event.png')
    plt.show()


def load_single_file(file_path, file_type="csv"):
    if file_type == "csv":
        data = pd.read_csv(file_path)
    elif file_type == "mseed":
        data = read(file_path)
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'mseed'.")

    return data


def is_trace_count_greater_than_one(stream_file):
    return len(stream_file.traces) > 1


def get_mseed_files(root_directory):
    """
    Find all MiniSEED files in the specified directory and its subdirectories.

    Parameters:
    root_directory (str): Path to the root directory.

    Returns:
    list: List of MiniSEED file paths.
    """
    mseed_files = []
    for dirpath, _, filenames in os.walk(root_directory):
        for file in filenames:
            if file.endswith('.mseed'):
                mseed_files.append(os.path.join(dirpath, file))
    return mseed_files


def check_trace_count():
    directory = [r'C:\Users\Mani Teja Varma\Documents\NASA\space_apps_2024_seismic_detection\data\lunar\test\data']

    for file in get_mseed_files(directory[0]):
        stream_file = load_single_file(file, file_type="mseed")
        if is_trace_count_greater_than_one(stream_file):
            print(f"Trace count is greater than 1 for file {file}")

    print("all are using only one trace")


if __name__ == "__main__":
    # Test if any of the files in train or test are using more than one trace. If using more than one trace,
    # we could somehow leverage it to make an accurate model
    #check_trace_count() # result: all are using only one trace
    pass
