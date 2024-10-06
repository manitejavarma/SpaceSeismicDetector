import numpy as np


def sliding_window(data, event_starts, window_size=1000, step_size=500):
    """
    Generate overlapping windows from continuous seismic data and label them based on event start times.
    :param data: Continuous seismic data
    :param event_starts: List of event start times (in samples)
    :param window_size: Size of each sliding window (in samples)
    :param step_size: Step size (overlap between windows, in samples)
    :return: Tuple of windows and their labels (1 if event start is within the window, 0 otherwise)
    """
    num_windows = (len(data) - window_size) // step_size + 1
    windows = []
    labels = []

    for i in range(num_windows):
        window_start = i * step_size
        window_end = window_start + window_size

        # Extract the window from the data
        window = data[window_start:window_end]

        # Find event start within this window (if any)
        event_start_in_window = [start for start in event_starts if window_start <= start < window_end]

        # If there's an event start within the window, calculate its relative position
        if event_start_in_window:
            relative_event_time = event_start_in_window[0] - window_start
        else:
            relative_event_time = -1  # No event in this window

        windows.append(window)
        labels.append(relative_event_time)

    return np.array(windows), np.array(labels)


if __name__ == "__main__":
    # Example usage:
    seismic_data = np.random.randn(10000)  # Simulated seismic data
    event_starts = [1200, 3200, 8000]  # Example event start times (in samples)

    window_size = 2000  # Window size (number of samples)
    step_size = window_size//2  # Step size (overlap between windows)

    # Generate sliding windows and labels
    windows, labels = sliding_window(seismic_data, event_starts, window_size, step_size)

    print("Number of windows:", len(windows))
    print("First window label:", labels[0])  # Should print 0 or 1 depending on if the event starts in the first window
