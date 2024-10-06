import numpy as np
from tqdm import tqdm
from feature_extractor import extract_spectrogram

def sliding_window(data, event_starts, sampling_rate, window_size=1000, step_size=500):
    """
    Generate overlapping windows from continuous seismic data and label them based on event start times.
    :param data: Continuous seismic data
    :param event_starts: List of event start times (in samples)
    :param window_size: Size of each sliding window (in samples)
    :param step_size: Step size (overlap between windows, in samples)
    :return: Tuple of windows and their labels (1 if event start is within the window, 0 otherwise)
    """
    num_windows = (len(data) - window_size) // step_size + 1
    labels = []
    spectrogram = []

    unique_event_starts = []

    for i in range(num_windows):
        window_start = i * step_size
        window_end = window_start + window_size

        # Extract the window from the data
        window = data[window_start:window_end]

        #Compute spectogram
        spec = extract_spectrogram(window, sampling_rate)
        spectrogram.append(spec)

        # Find event start within this window (if any)
        event_start_in_window = [start for start in event_starts if window_start <= start < window_end]

        # If there's an event start within the window, calculate its relative position
        if event_start_in_window:
            relative_event_time = int(event_start_in_window[0]) - window_start
            unique_event_starts.append(relative_event_time)
        else:
            relative_event_time = -1  # No event in this window

        labels.append(relative_event_time)
    print(f'Unique values of relative_event_time: {np.unique(labels)}')

    return np.array(spectrogram), np.array(labels)
