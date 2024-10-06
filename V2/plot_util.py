import numpy as np
from matplotlib import pyplot as plt


def plot_seismic_event_miniseed(tr_data, tr_time, events):
    # Plot on and off triggers
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    for i in np.arange(0, len(events)):
        event = events[i]
        ax.axvline(x=tr_time[event], color='red', label='Trig. On')

    # Plot seismogram
    ax.plot(tr_time, tr_data)
    ax.set_xlim([min(tr_time), max(tr_time)])
    ax.legend()
    plt.savefig('seismic_event.png')
    plt.show()
