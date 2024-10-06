import numpy as np
from utilitie import load_single_file
from config import DATA_DIR
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def dominant_freq(data, fs):
    length = len(data)
    fft_vals = fft(data)
    fft_freqs = np.fft.fftfreq(length, 1/fs)
    fft_power = np.abs(fft_vals)
    return fft_freqs[np.argmax(fft_power[:length//2])]

def butter_bandpass(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    low = max(0.01, min(1.0, low))
    high = max(0.01, min(1.0, high))

    print(f"Lowcut: {lowcut}, Highcut: {highcut}, Nyquist: {nyquist}, Normalized: low={low}, high={high}")

    if low >= high:
        raise ValueError(f"Invalid frequency range: low={low}, high={high}. Ensure low < high and both are > 0.")

    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(data, lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def filter_data(stream_file=None, plot=False):
    WINDOW_SIZE = 6000
    STEP_SIZE = WINDOW_SIZE // 2

    trace = stream_file.traces[0].copy() 
    all_velocity = trace.data
    all_time = trace.times()

    filtered_data = np.zeros_like(all_velocity)

    prev_lowcut, prev_highcut = 0.1, 10.0

    for point in range(0, len(all_velocity) - WINDOW_SIZE + 1, STEP_SIZE):
        window_data = all_velocity[point:point + WINDOW_SIZE]
        window_time = all_time[point:point + WINDOW_SIZE]

        delta_time = np.mean(np.diff(window_time)) 
        fs = 1 / delta_time 

        print(f"\nWindow start: {point}, Sampling Frequency: {fs}")

        try:
            dom_freq = dominant_freq(window_data, fs)
            print(f"Dominant Frequency: {dom_freq}")
        except ValueError:
            dom_freq = (prev_lowcut + prev_highcut) / 2
            print(f"Using previous frequency as dominant frequency: {dom_freq}")

        # Calculate lowcut and highcut based on dominant frequency
        lowcut = max(0.1, dom_freq - 2)
        highcut = min(fs / 2 - 0.01, dom_freq + 2)

        # Ensure valid frequency range
        if lowcut >= highcut:
            print(f"Adjusting invalid frequency range: lowcut={lowcut}, highcut={highcut}. Using previous values.")
            lowcut, highcut = prev_lowcut, prev_highcut

        print(f"Calculated Lowcut: {lowcut}, Highcut: {highcut}")

        # Normalize frequencies for the butter bandpass
        low = lowcut / (0.5 * fs)
        high = highcut / (0.5 * fs)

        # Adjust normalized values if necessary
        if low >= high:
            high = low + 0.01  # Increment high slightly to avoid overlap
            print(f"Adjusted normalized frequencies: low={low}, high={high} to avoid overlap.")

        print(f"Normalized: low={low}, high={high}")

        # Try filtering the window data
        try:
            filtered_window = bandpass_filter(window_data, lowcut, highcut, fs)
        except ValueError as e:
            print(f"Adjusting filter range due to error: {e}")
            filtered_window = bandpass_filter(window_data, prev_lowcut, prev_highcut, fs)

        filtered_data[point:point + WINDOW_SIZE] = filtered_window

        prev_lowcut, prev_highcut = lowcut, highcut

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(all_time, all_velocity, label='Original Velocity Data', color='red')
        plt.plot(all_time[:len(filtered_data)], filtered_data, label='Dynamically Filtered Data', color='blue')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity')
        plt.legend()
        plt.show()

#data_directory = f'{DATA_DIR}/lunar/training/data/S12_GradeA/'
#test_filename = 'xa.s12.00.mhz.1970-03-25HR00_evid00003'
#mseed_file = f'{data_directory}{test_filename}.mseed'
#stream_file = load_single_file(mseed_file, file_type="mseed")
#filter_data(stream_file=stream_file, plot=True)
