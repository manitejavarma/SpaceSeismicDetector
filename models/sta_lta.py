from obspy.signal.trigger import classic_sta_lta
from utilities.utilities import *


def sta_lta_model(sta_len=120,
                  lta_len=600, stream_file=None, plot=False):
    # Run Obspy's STA/LTA to obtain a characteristic function
    # This function basically calculates the ratio of amplitude between the short-term
    # and long-term windows, moving consecutively in time across the data

    sampling_rate = stream_file[0].stats.sampling_rate

    trace = stream_file.traces[0].copy()
    tr_data = trace.data

    cft = classic_sta_lta(tr_data, int(sta_len * sampling_rate), int(lta_len * sampling_rate))

    thr_on = 4
    thr_off = 1.5
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))

    if plot:
        # Plot characteristic function
        plot_characteristic_function(stream_file, cft)
        plot_seismic_event_miniseed(cft, stream_file, thr_on, thr_off, on_off)

    return on_off
