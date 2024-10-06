import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from obspy.signal.trigger import classic_sta_lta

# Example seismic data
sampling_rate = 100  # Hz, adjust based on your data
seismic_data = np.random.randn(10000)  # Simulated seismic data

# Feature 1: Spectrogram (STFT)
def extract_spectrogram(data, fs):
    """Extracts the spectrogram (STFT) from the seismic data."""
    f, t, Sxx = spectrogram(data, fs=fs)
    Sxx_log = np.log1p(Sxx)  # Use logarithmic scale for better feature representation
    return Sxx_log

# Feature 2: Wavelet Transform
def extract_wavelet_transform(data, wavelet='cmor', max_scale=128):
    """Extracts the Wavelet Transform (CWT) from the seismic data."""
    scales = np.arange(1, max_scale)
    coeffs, freqs = pywt.cwt(data, scales, wavelet)
    return coeffs, freqs

# Feature 3: STA/LTA
def extract_sta_lta(data, nsta, nlta, df):
    """Extracts the STA/LTA ratio from the seismic data."""
    sta_lta = classic_sta_lta(data, int(nsta * df), int(nlta * df))
    return sta_lta

# Feature 4: Energy
def extract_energy(data):
    """Extracts the energy (sum of squared amplitudes) from the seismic data."""
    energy = np.sum(data ** 2)
    return energy

# Extract all features from the data
def extract_features(data, fs):
    # Spectrogram
    f, t, Sxx = extract_spectrogram(data, fs)
    print("Spectrogram extracted.")

    # Wavelet Transform
    coeffs, freqs = extract_wavelet_transform(data)
    print("Wavelet Transform extracted.")

    # STA/LTA
    sta_lta = extract_sta_lta(data, nsta=5, nlta=30, df=fs)
    print("STA/LTA extracted.")

    # Energy
    energy = extract_energy(data)
    print(f"Energy extracted: {energy}")

    return {
        'spectrogram': (f, t, Sxx),
        'wavelet': (coeffs, freqs),
        'sta_lta': sta_lta,
        'energy': energy
    }
#
# # Run feature extraction
# features = extract_features(seismic_data, sampling_rate)
#
# # Plotting the features (optional)
# # 1. Plot Spectrogram
# f, t, Sxx = features['spectrogram']
# plt.figure(figsize=(10, 5))
# plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title('Spectrogram (STFT)')
# plt.colorbar(label='Power/Frequency (dB/Hz)')
# plt.show()
#
# # 2. Plot Wavelet Transform
# coeffs, freqs = features['wavelet']
# plt.figure(figsize=(10, 5))
# plt.imshow(np.abs(coeffs), extent=[0, len(seismic_data)/sampling_rate, 1, 128], aspect='auto', cmap='jet')
# plt.ylim([1, 128])
# plt.colorbar(label='Magnitude')
# plt.title('Wavelet Transform (CWT)')
# plt.ylabel('Scale')
# plt.xlabel('Time [sec]')
# plt.show()
#
# # 3. Plot STA/LTA
# plt.figure(figsize=(10, 5))
# plt.plot(features['sta_lta'])
# plt.title('STA/LTA Ratio')
# plt.xlabel('Sample Index')
# plt.ylabel('STA/LTA')
# plt.show()

