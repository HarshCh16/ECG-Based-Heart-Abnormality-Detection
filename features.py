import numpy as np
import pandas as pd
import wfdb
import os
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)


def extract_features_from_record(record_id, data_folder, window=5):
    path = os.path.join(data_folder, record_id)
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')

    fs = record.fs
    raw_signal = record.p_signal[:, 0]
    filtered_signal = bandpass_filter(raw_signal, fs=fs)

    peaks = annotation.sample
    labels = annotation.symbol
    normal_symbols = {'N'}

    rr_intervals = []
    rows = []

    for i in range(1, len(peaks)):
        symbol = labels[i]
        if symbol not in wfdb.io.annotation.ann_label_table['symbol'].values:
            continue

        rr_interval = (peaks[i] - peaks[i-1]) / fs
        if rr_interval <= 0 or rr_interval > 3:
            continue
        heart_rate = 60 / rr_interval
        rr_intervals.append(rr_interval)

        recent_rr = rr_intervals[-window:]
        hrv = np.std(recent_rr) if len(recent_rr) > 1 else 0.0

        peak_pos = peaks[i]
        half_win = int(0.1 * fs)
        start = max(0, peak_pos - half_win)
        end = min(len(filtered_signal), peak_pos + half_win)
        segment = filtered_signal[start:end]
        amplitude = filtered_signal[peak_pos] if peak_pos < len(filtered_signal) else 0.0

        if len(segment) > 0 and amplitude != 0:
            threshold = amplitude * 0.5
            above = np.where(np.abs(segment) > np.abs(threshold))[0]
            qrs_width = (above[-1] - above[0]) / fs if len(above) > 1 else 0.0
        else:
            qrs_width = 0.0

        label = 'Normal' if symbol in normal_symbols else 'Abnormal'

        rows.append({
            'sample_pos': peak_pos,
            'rr_interval': rr_interval,
            'heart_rate': heart_rate,
            'hrv': hrv,
            'amplitude': amplitude,
            'qrs_width': qrs_width,
            'true_label': label
        })

    return pd.DataFrame(rows), filtered_signal, fs