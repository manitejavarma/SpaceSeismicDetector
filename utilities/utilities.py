import numpy as np
from obspy.signal.trigger import trigger_onset
import matplotlib.pyplot as plt
import pandas as pd
from obspy import read


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
