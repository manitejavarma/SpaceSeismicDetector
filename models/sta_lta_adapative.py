import numpy as np
import matplotlib.pyplot as plt


''' GPT generated adaptive STA/LTA event detection. Not tested. '''

# Simulated seismic data with noise and events
np.random.seed(0)
data = np.random.normal(0, 0.5, 1000)  # background noise
data[300:330] += np.sin(np.linspace(0, 3*np.pi, 30))  # seismic event 1
data[700:720] += np.sin(np.linspace(0, 3*np.pi, 20)) * 2  # seismic event 2

# Parameters
initial_sta = 5  # STA window (seconds)
initial_lta = 30  # LTA window (seconds)
sampling_rate = 10  # samples per second
base_threshold = 3  # Base STA/LTA threshold multiplier

# Adaptive STA/LTA calculation
sta_window = initial_sta * sampling_rate
lta_window = initial_lta * sampling_rate

# Calculate STA/LTA with adaptive windows
sta_lta_ratios = calculate_sta_lta(data, sta_window, lta_window)

def adaptive_threshold(lta_values, base_threshold=2.5):
    # Threshold scales with the long-term average noise level
    return base_threshold * np.median(lta_values)

# Adjust threshold dynamically based on LTA values
dynamic_threshold = adaptive_threshold(sta_lta_ratios, base_threshold)


def sliding_window_adaptive_sta_lta(data, sta_window, lta_window, base_threshold):
    sta_lta_ratio = calculate_sta_lta(data, sta_window, lta_window)

    # Dynamically update the threshold
    dynamic_threshold = adaptive_threshold(data[:lta_window], base_threshold)

    # Trigger event detection where the STA/LTA exceeds the dynamic threshold
    event_detected = np.where(sta_lta_ratio > dynamic_threshold, 1, 0)
    return event_detected, sta_lta_ratio


# Detect seismic events where STA/LTA exceeds dynamic threshold
detected_events, sta_lta_ratios = sliding_window_adaptive_sta_lta(data, sta_window, lta_window, base_threshold)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data, label="Seismic Data")
plt.plot(np.arange(len(sta_lta_ratios)), sta_lta_ratios, label="STA/LTA Ratio")
plt.axhline(y=dynamic_threshold, color='r', linestyle='--', label=f"Dynamic Threshold ({dynamic_threshold:.2f})")
plt.legend()
plt.title("Adaptive STA/LTA Seismic Event Detection")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude / STA-LTA Ratio")
plt.show()
