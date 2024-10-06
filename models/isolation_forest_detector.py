from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from utilities.utilities import plot_seismic_event_miniseed


def isolation_forest_detector(tr_data, tr_time, plot=True):
    reshaped_data = tr_data.reshape(-1, 1)

    model = IsolationForest(contamination=0.1)  # Result: tried all combinations. Produces no anomalies

    # Fit the model
    model.fit(reshaped_data)

    # Predict anomalies
    predictions = model.predict(reshaped_data)

    event_indices = np.where(predictions == -1)[0]

    if plot:
        plot_seismic_event_miniseed(tr_data, tr_time, event_indices)

    return tr_time[event_indices]
