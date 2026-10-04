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


# --- Rhythm-model features ---------------------------------------------------
# Everything above is the BEAT-level path: one row per beat, 5 features, feeding
# ecg_model.pkl. Everything below is the RHYTHM-level path: one row per 60-second
# window, 7 features, feeding rhythm_model.pkl (NSR / AFib / AFlutter).
#
# Note `hrv` above and `hrv_sdnn` below are NOT the same quantity. `hrv` is
# np.std of the last 5 RR intervals (ddof=0) inside a 5-second buffer;
# `hrv_sdnn` is the SD of ~95 intervals (ddof=1) over a full minute. Same idea,
# different random variable. Do not feed one to a model trained on the other.
#
# These constants are the model's serving contract and must equal the ones in
# icentia_windows.py, which produced the training table. That file deliberately
# cannot import this one -- it must stay free of wfdb and network imports (a
# guard in test_notebook.py enforces it) and this module imports wfdb. So the
# maths exists twice on purpose, and test_rhythm_features.py is the only thing
# binding the two copies together. Do not edit either without running it.
RR_MIN_S, RR_MAX_S = 0.24, 2.0   # HR 250..30. Looser at the bottom than the
                                 # beat path's 0.3: rapid AF genuinely produces
                                 # sub-0.3s cycles, and that irregularity IS the
                                 # signal we are trying to detect.
PNN_THRESH_S = 0.05              # pNN50 convention
ENTROPY_BINS = 16                # FIXED bins spanning [RR_MIN_S, RR_MAX_S], so
                                 # entropy is comparable across windows
RHYTHM_FEATURES = ('mean_rr', 'hr', 'hrv_sdnn', 'rmssd', 'pnn50', 'cv_rr',
                   'rr_entropy')
RHYTHM_COUNTERS = ('n_rr_used', 'n_rr_rejected', 'n_drr_used')


def rhythm_features_from_beats(beat_samples, fs, rr_min_s=RR_MIN_S,
                               rr_max_s=RR_MAX_S, pnn_thresh_s=PNN_THRESH_S,
                               n_bins=ENTROPY_BINS):
    """The 7 rhythm features for ONE window, from integer beat sample positions.

    Single-window equivalent of icentia_windows.rr_window_features(), which
    computed the training table.

    beat_samples : integer sample positions, ascending. Absolute or
                   window-relative both work -- np.diff makes it offset
                   invariant. MUST be sample indices, not seconds; see below.
    fs           : sampling rate in Hz of those positions.

    Returns a dict of plain floats: the 7 features (nan where undefined) plus
    n_rr_used / n_rr_rejected / n_drr_used. A dict rather than a Series or
    DataFrame on purpose -- it has no implicit column order, so a caller is
    forced to project features by name and cannot silently feed a model the
    wrong order. Never raises: live_predict.py's loop swallows exceptions, so a
    raise here would become an invisible per-sample spin.

    Integer positions are a requirement, not a convenience. RR must be
    np.diff(beats)/fs and NOT np.diff(beats/fs). At 250 Hz an RR of 170 samples
    is 0.680 s, which lands exactly on entropy bin edge 4.0, so a last-bit
    float difference truncates it into bin 3 instead. Dividing first changes
    rr_entropy in 20 of the 69 windows of p00000_s00, by up to 0.294 bits. The
    other six features agree to ~1e-13 either way, which is exactly why this
    would slip through casual testing.
    """
    b = np.asarray(beat_samples, dtype=np.int64)
    out = {k: float('nan') for k in RHYTHM_FEATURES}
    out.update(n_rr_used=0, n_rr_rejected=0, n_drr_used=0)
    if b.size < 2:
        return out

    rr = np.diff(b).astype(np.float64) / fs        # diff FIRST, then divide
    ok = (rr >= rr_min_s) & (rr <= rr_max_s)
    rr_ok = rr[ok]
    out['n_rr_used'] = int(ok.sum())
    out['n_rr_rejected'] = int((~ok).sum())

    if rr_ok.size >= 1:
        mean_rr = float(rr_ok.mean())
        out['mean_rr'] = mean_rr
        out['hr'] = 60.0 / mean_rr

    if rr_ok.size >= 2:
        sd = float(np.std(rr_ok, ddof=1))
        out['hrv_sdnn'] = sd
        out['cv_rr'] = sd / out['mean_rr']
        idx = ((rr_ok - rr_min_s) / (rr_max_s - rr_min_s) * n_bins).astype(np.int64)
        np.clip(idx, 0, n_bins - 1, out=idx)
        counts = np.bincount(idx, minlength=n_bins)
        p = counts[counts > 0] / counts.sum()
        out['rr_entropy'] = float(-(p * np.log2(p)).sum())

    # Successive differences may ONLY span two RR intervals that are adjacent in
    # the original index and both accepted. Otherwise a rejected interval -- a
    # noise gap -- silently joins two non-consecutive cycles and inflates rmssd,
    # which is the feature AF detection leans on hardest.
    if rr.size >= 2:
        adj = ok[:-1] & ok[1:]
        d = np.diff(rr)[adj]
        out['n_drr_used'] = int(d.size)
        if d.size >= 1:
            out['rmssd'] = float(np.sqrt(np.mean(d * d)))
            out['pnn50'] = float(np.mean(np.abs(d) > pnn_thresh_s))

    return out