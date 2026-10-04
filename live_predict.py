"""Live ECG monitoring: per-beat ectopy AND per-minute rhythm, side by side.

Two models run together, answering different questions:

  ecg_model.pkl     one BEAT  -> Normal / Abnormal   (5 features, ~3s cadence)
  rhythm_model.pkl  one MINUTE -> NSR / AFib / AFlutter (7 RR features, 15s cadence)

A beat-level model cannot see atrial fibrillation: AF is defined by the pattern
across a minute of beats, not by the shape of any single beat. That is the gap
the rhythm model fills.

HOW FAST CAN THIS SEE AF?  Not fast. Read this before demoing it.

The 15s cadence is the HOP, not the window: every verdict analyses a full 60
seconds of beat history. So from the moment AF starts:

    45-60s   for the 60s window to fill with AF
     +30s    for AF_CONSEC=3 over-threshold windows at a 15s hop
    ------
     ~90s    to alarm   (measured ~90s on synth:nsr-af-nsr)

The 60s is NOT a tunable constant. rhythm_model.pkl carries window_sec=60 and
every feature is window-length dependent -- SDNN over 30 intervals is
systematically smaller than over 70, and rr_entropy is biased low with fewer
samples -- so a shorter window needs a RETRAIN, not an edit.

The hop is tunable and nearly free, but shrinking it is a bad trade: at a 5s hop
AF_CONSEC=3 would span 10s of real time instead of 30s, gutting the debounce
that absorbs artifact-driven false AF, and latency would still only fall to ~70s
because the 60s fill dominates.

Consequence: episodes shorter than ~60s are invisible to this. That is tolerable
because clinical practice only calls something AF at >=30s and real episodes run
minutes to hours -- a 5-minute episode presents ~20 windows. It is still a real
limit and should be stated when presenting results.

Run against hardware:

    python live_predict.py --port COM9 --fs 320

Run offline against MIT-BIH, which has real rhythm annotations (no hardware):

    python live_predict.py --replay mit-bih-arrhythmia-database/219 --fs 360 \
        --speed 0 --no-plot

    python live_predict.py --replay synth:nsr-af-nsr --speed 0 --no-plot

The serial port is READ-ONLY. The Arduino buzzer and LED beep once per detected
beat, which is the behaviour worth keeping, and tone() can only play one
frequency at a time -- a host-driven alarm tone would silence the beat beep for
as long as it sounded. Every alarm therefore surfaces on the host instead: the
status header (which flashes), the terminal bell, and --sound.
"""
import argparse
import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, lfilter, lfilter_zi

import features
from features import bandpass_filter

# --- Defaults ---------------------------------------------------------------
DEFAULT_PORT = 'COM9'
DEFAULT_BAUD = 115200
DEFAULT_FS = 300          # measure this; see the --fs note in check_sample_rate

# The shipped Arduino sketch loops: delay(3) + analogRead (~112us) + 2x
# digitalRead (~8us) + Serial.println. At 115200 baud a 3-6 byte line takes
# 260-520us to transmit, but println only blocks once the 64-byte TX buffer
# fills, and ~6 bytes per 3.2ms loop is 1.9 kB/s against 11.5 kB/s of capacity,
# so it never fills and queuing costs ~10us. That puts the loop period near
# 3.13ms -> fs ~= 320 Hz, not 300. Which is 6.7% out, just past FS_TOLERANCE, so
# the sample-rate warning is EXPECTED on a first hardware run. Measure it, then
# pass --fs with the measured value.

# --- Rhythm-path knobs ------------------------------------------------------
RHYTHM_HOP_SEC = 15       # a fresh verdict every 15s, each spanning a full 60s
AF_THRESHOLD = 0.90       # on P(AFib) + P(AFlutter)
AF_CONSEC = 3             # over-threshold windows required before alarming
AF_CLEAR = 2              # under-threshold windows required to clear it
MIN_RR_USED = 29          # training used n_beats >= 30, i.e. 29 RR intervals
MIN_COVERAGE = 0.5        # accepted-RR seconds / window seconds

# --- Beat-path alert -------------------------------------------------------
# The beat model flags ectopic beats, and PVCs occur in most healthy people, so
# a single Abnormal beat is not worth an alert. Hysteresis (on at 5, off at 2)
# rather than one threshold, or the alert chatters on and off at the boundary.
ECTOPY_N = 10             # look at the last N classified beats
ECTOPY_ON = 5             # >= this many Abnormal -> alert on
ECTOPY_OFF = 2            # <= this many Abnormal -> alert off

# --- Display ---------------------------------------------------------------
PLOT_SEC = 4              # seconds of ECG on screen
# 30 fps, wall-clock gated (see Dashboard). Chosen for SCROLL smoothness, which
# is a different thing from waveform fidelity: the trace advances
# axes_width/(PLOT_SEC * fps) pixels per frame, so 30 fps steps ~10px across a
# ~1000px axes while 15 fps would step ~21px and read as juddery. Measured cost
# is 8.9 ms/frame, so 30 fps is ~27% of wall clock -- verified to keep up with a
# real-time stream without dropping samples.
PLOT_FPS = 30
AF_TREND_MIN = 5          # minutes of AF-score history in the trend panel
RR_PLOT_LO, RR_PLOT_HI = 0.2, 1.8    # tachogram y range, seconds
# Display-only copies of the accept range, so the tachogram flags exactly the
# intervals rhythm_features_from_beats rejects. features.py stays the source of
# truth for the model itself.
RR_MIN_S_DISPLAY = features.RR_MIN_S
RR_MAX_S_DISPLAY = features.RR_MAX_S

# Held-out-patient performance, with precision rescaled to an assumed 2%
# real-world AF prevalence (the test set is 76.8% AF by construction):
#
#   threshold  AF recall  precision(test)  precision @2% AF
#      0.50      0.9599       0.9536            0.1124
#      0.70      0.9455       0.9679            0.1566
#      0.80      0.9310       0.9746            0.1912
#      0.90      0.8441       0.9833            0.2661   <- AF_THRESHOLD
#      0.95      0.5490       0.9877            0.3313
#
# 0.90 gives a 2.4x cut in false-alarm burden over the default 0.50 for a modest
# recall cost, and per-window recall understates episode detection because AF
# lasts minutes to hours, so one episode presents ~20 sliding windows. 0.95 is
# rejected: recall collapses to 0.55 for little precision gain. Independently
# confirmed on MIT-BIH, where 0.90 removes both of record 202's NSR false
# positives while keeping 24 of 26 AF windows.
#
# CAUTION on AF_CONSEC: consecutive 15s-hop windows share 75% of their beats, so
# they are NOT three independent tests. The tempting 0.0475**3 ~ 1-in-9000
# false-alarm estimate is WRONG. What the debounce actually buys is a
# requirement that AF-like RR statistics persist across a 90-second span. The
# real false-alarm rate per hour must be MEASURED on a quiet NSR recording from
# this hardware; it cannot be derived from the test-set FPR.

FS_CHECK_AFTER_SEC = 20   # first sample-rate check after this much stream
FS_RECHECK_SEC = 60
FS_TOLERANCE = 0.05       # warn past 5%
FS_FATAL = 0.15           # past 15%, stop scoring rhythm entirely

# DOMAIN SHIFT -- read this before trusting a verdict.
#
# Training beats came from Icentia's annotated beat detector with Q
# (unclassifiable) beats EXCLUDED. Those Q beats sat 100% outside the rhythm
# spans -- they were detections inside noise gaps -- and removing them turned
# each gap into one huge RR that the [0.24, 2.0] filter then rejected. So noise
# was removed from training windows, and the removal stayed VISIBLE as
# n_rr_rejected.
#
# Live beats come from find_peaks() on a bandpass-filtered 5-second chunk with
# no noise labelling at all. Motion artifact either ADDS peaks (spuriously short
# RRs -> higher rmssd, pnn50, rr_entropy) or MISSES them (long RRs, rejected by
# the range filter). The asymmetry matters: artifact pushes the score toward
# FALSE AF, and precision at a realistic 2% prevalence is only ~0.27 even on
# clean input.
#
# Measured bound on the algorithmic part of the shift (MIT-BIH, clean signal):
# find_peaks recovered 2272 of 2273 annotated beats on record 100, and af_score
# moved by <= 0.03 on records 100/202/219. That bounds the DETECTOR shift, not
# the noise shift -- MIT-BIH is clinical-grade signal and this sensor is not.
#
# Known structural gap, deliberately NOT fixed here: find_peaks(distance=fs*0.4)
# caps detection at 150 bpm while the model accepts RR down to 0.24s (250 bpm),
# so rapid AF will have merged beats and mean_rr biased high. Lowering it to
# 0.25*fs would fix the ceiling but admit T-wave double-detections, and it would
# change the beat-level path too. One variable at a time: measure it offline
# against records 219/202 as its own change.
#
# Mitigations in force: the quality gates in RhythmScorer.score(), lead-off
# exclusion, the AF_CONSEC debounce, and n_rr_rejected logged on every window so
# a bad session stays diagnosable after the fact.


class RhythmScorer:
    """Scores one 60-second window of beat positions. No I/O, no globals."""

    def __init__(self, bundle_path='rhythm_model.pkl'):
        bundle = joblib.load(bundle_path)
        self.model = bundle['model']
        # The feature ORDER travels with the pickle -- that is the whole point of
        # saving a dict rather than a bare pipeline. Use the bundle's copy as the
        # authority; features.RHYTHM_FEATURES is the second line of defence and
        # test_rhythm_features.py asserts the two agree.
        self.features = list(bundle['features'])
        self.window_sec = int(bundle['window_sec'])
        self.name = bundle.get('model_name', 'rhythm model')
        # model.classes_ is ALPHABETICAL ['AFib','AFlutter','NSR'], which is NOT
        # bundle['classes'] ['NSR','AFib','AFlutter']. predict_proba columns
        # follow classes_, so every lookup goes through this map. Positional
        # indexing here would silently swap NSR and AFib.
        self.class_col = {c: i for i, c in enumerate(self.model.classes_)}

    def score(self, beat_positions, fs, window_end, leadoff=0):
        """-> dict with 'status' in {ok, warmup, leadoff, low_quality}.

        proba / af_score / rhythm_top are None unless status == 'ok'.
        """
        out = {'status': 'ok', 'feats': None, 'coverage': float('nan'),
               'proba': None, 'af_score': None, 'rhythm_top': None}

        if window_end < self.window_sec * fs:
            out['status'] = 'warmup'
            return out
        if leadoff > 0:
            # A 60-second AF screen taken across a detached electrode is
            # worthless, however good the rest of the minute looked.
            out['status'] = 'leadoff'
            return out

        feats = features.rhythm_features_from_beats(list(beat_positions), fs)
        out['feats'] = feats

        if feats['n_rr_used'] < MIN_RR_USED:
            out['status'] = 'low_quality'
            return out
        if feats['n_drr_used'] < 1:
            # Verbatim from the training query; rmssd/pnn50 are nan below this.
            out['status'] = 'low_quality'
            return out

        # frac_covered and purity have NO live analogue -- they were properties
        # of the rhythm ANNOTATIONS, and live data has none. So every live window
        # is admitted under a weaker standard than any training window. This is
        # the honest substitute: the fraction of the minute genuinely spanned by
        # physiologically plausible cycles, which is the same mechanism (noise
        # gaps) that made low-coverage training windows untrustworthy.
        coverage = feats['n_rr_used'] * feats['mean_rr'] / self.window_sec
        out['coverage'] = coverage
        if coverage < MIN_COVERAGE:
            out['status'] = 'low_quality'
            return out

        vals = [feats[k] for k in self.features]
        if not all(np.isfinite(v) for v in vals):
            # Mandatory, not defensive: StandardScaler + LogisticRegression
            # RAISES on nan, and the caller's blanket except would swallow it
            # into an invisible per-sample spin.
            out['status'] = 'low_quality'
            return out

        # A named DataFrame, not a bare array: the pipeline was fitted on one, so
        # an ndarray triggers sklearn's "X does not have valid feature names"
        # warning on every call, and names turn a feature-order mistake into an
        # error rather than a silently wrong score.
        X = pd.DataFrame([vals], columns=self.features)
        proba = self.model.predict_proba(X)[0]
        out['proba'] = proba
        # Summing the two AF probabilities is exactly how the 0.9599 AUC was
        # measured, so the threshold table above transfers directly.
        out['af_score'] = float(proba[self.class_col['AFib']]
                                + proba[self.class_col['AFlutter']])
        out['rhythm_top'] = str(self.model.classes_[int(proba.argmax())])
        return out


class ReplaySerial:
    """Duck-types serial.Serial.readline() from an int array, for offline runs.

    Lets the exact production loop -- buffer, trim, find_peaks, de-duplication,
    gating, state machine, both CSVs -- run with no hardware attached.
    """

    def __init__(self, samples, fs, speed=0.0, leadoff_at=()):
        self.samples = np.asarray(samples)
        self.i = 0
        self.period = (1.0 / (fs * speed)) if speed else 0.0
        self.t0 = None
        self.leadoff_at = set(int(x) for x in leadoff_at)

    def readline(self):
        if self.i >= len(self.samples):
            return b''                      # EOF -> main() stops
        if self.period:
            # Sleep to an ABSOLUTE deadline, not for a fixed interval. Windows'
            # sleep granularity makes time.sleep(1/300) take ~3.6 ms, which
            # capped the old per-sample version at ~277 Hz and made --speed 1
            # trip the sample-rate warning all by itself -- the harness failing
            # its own test. Against a deadline the error self-corrects: after an
            # overshoot the next samples are already due and return instantly,
            # so the average rate is fs as long as the loop can keep up. Which
            # makes the measured rate a real measurement of whether it can.
            if self.t0 is None:
                self.t0 = time.monotonic()
            due = self.t0 + self.i * self.period
            slack = due - time.monotonic()
            if slack > 0:
                time.sleep(slack)
        i, self.i = self.i, self.i + 1
        if i in self.leadoff_at:
            return b'!\n'
        return f"{int(self.samples[i])}\n".encode()

    def close(self):
        pass


def load_replay_source(spec, fs):
    """-> (samples int array, leadoff indices, description)."""
    if spec.startswith('synth:'):
        return synth_stream(spec.split(':', 1)[1], fs)

    import wfdb
    rec = wfdb.rdrecord(spec)
    sig = rec.p_signal[:, 0]
    # The device emits unsigned ADC integers and the loop filters on
    # line.isdigit(), so map mV onto a positive integer range. Beat detection
    # uses height=mean+std on the filtered signal, which is relative, so the
    # exact scale does not matter -- only that it stays positive and integral.
    vals = np.clip(np.round(512 + 200.0 * sig), 0, 4095).astype(int)
    return vals, (), f"{spec} ({rec.fs} Hz, {len(vals)/rec.fs:.0f}s)"


def synth_stream(kind, fs, seed=0):
    """Render a spike train from generated beat times. Exercises the paths
    MIT-BIH cannot: exact known beat positions, alarm timing, and the gates."""
    rng = np.random.default_rng(seed)
    segments = {
        'nsr': [('nsr', 180)],
        'af': [('af', 180)],
        'nsr-af-nsr': [('nsr', 120), ('af', 180), ('nsr', 120)],
        # 'flat' is genuinely silent -- no beats AND no noise -- so find_peaks
        # finds nothing and the beat-count gate is what has to catch it.
        'flat': [('nsr', 90), ('flat', 60), ('nsr', 90)],
        # 'noise' is the more realistic and more dangerous case: no beats, but
        # baseline noise that survives the bandpass and clears
        # height=mean+std, so find_peaks invents beats. Irregular invented
        # beats look like AF. This is the artifact failure mode the debounce
        # exists to absorb.
        'noise': [('nsr', 90), ('noise', 60), ('nsr', 90)],
    }[kind]

    beat_times, spans, t = [], [], 0.0
    for mode, dur in segments:
        end = t + dur
        spans.append((mode, t, end))
        while t < end:
            if mode == 'nsr':
                rr = float(rng.normal(0.85, 0.02))
            elif mode == 'af':
                rr = float(rng.uniform(0.40, 1.10))
            else:                            # flat / noise: no beats at all
                t = end
                break
            t += max(rr, 0.25)
            if t < end:
                beat_times.append(t)
    total = int(sum(d for _, d in segments) * fs)

    vals = np.full(total, 512, dtype=float)
    vals += rng.normal(0, 1.5, total)        # mild baseline noise
    for mode, a, b in spans:                 # silence the truly-flat stretches
        if mode == 'flat':
            vals[int(a * fs):int(b * fs)] = 512.0
    qrs = np.array([-40, 120, 380, 120, -40], dtype=float)  # crude QRS shape
    positions = []
    for bt in beat_times:
        c = int(round(bt * fs))
        lo = c - len(qrs) // 2
        if lo < 0 or lo + len(qrs) > total:
            continue
        vals[lo:lo + len(qrs)] += qrs
        positions.append(c)
    out = np.clip(np.round(vals), 0, 4095).astype(int)
    synth_stream.last_positions = np.array(positions)
    return out, (), f"synth:{kind} ({total/fs:.0f}s, {len(positions)} beats)"


def check_sample_rate(total_samples, t_start, fs, already_warned):
    """-> (severity, measured) where severity is 'ok' | 'warn' | 'fatal'.

    RR is samples/fs, so an fs error of X% biases mean_rr, hr, hrv_sdnn and
    rmssd by X%. Worse, the corruption is NOT uniform: cv_rr is a ratio and
    survives, but pnn50's 50ms threshold and the 16 fixed entropy bins over
    [0.24, 2.0] are ABSOLUTE, so a wrong fs slides every RR across bin edges and
    across the accept/reject boundary -- distorting precisely the features that
    carry the irregularity signal. Separately, at a true 250 Hz with fs=300 the
    "60 second" window is really 72 seconds, breaking the bundle's window_sec
    contract.

    The measurement is a LOWER BOUND -- readline() blocking and plt.pause() both
    inflate elapsed time -- so it is never adopted automatically. Pass --fs with
    the measured value instead.
    """
    elapsed = max(time.monotonic() - t_start, 1e-9)
    measured = total_samples / elapsed
    dev = abs(measured / fs - 1.0)
    if dev > FS_FATAL:
        return 'fatal', measured
    if dev > FS_TOLERANCE and not already_warned:
        return 'warn', measured
    return 'ok', measured


THEMES = {
    'dark': dict(bg='#0b0f14', fg='#cfd8dc', dim='#78909c', grid='#1b2530',
                 axis='#3a4652', ecg='#00e676', af='#29b6f6', thresh='#ff5252',
                 ok='#00e676', warn='#ffb300', alarm='#ff1744', idle='#78909c',
                 detect='#546e7a', normal='#00e676', abnormal='#ff5252'),
    'light': dict(bg='#ffffff', fg='#212121', dim='#616161', grid='#e8e8e8',
                  axis='#9e9e9e', ecg='#000000', af='#1565c0', thresh='#c62828',
                  ok='#2e7d32', warn='#ef6c00', alarm='#c62828', idle='#757575',
                  detect='#bdbdbd', normal='#2e7d32', abnormal='#c62828'),
}


def fmt_hms(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def rhythm_label(af_score, af_run, af_alarm):
    """-> (text, theme colour key) for the headline rhythm verdict.

    Deliberately NOT the model's argmax, which is what the display used to
    show. The bundle's own note says to apply a threshold rather than argmax,
    because the training set is 76.8% AF by construction.

    Measured on the replay set: 122 scored windows (21 on 202, 33 on 219, 66 on
    222, 2 on synth) had an argmax of AFib or AFlutter while af_score sat below
    AF_THRESHOLD -- so the screen named a specific arrhythmia while the alarm
    beside it stayed silent. 'IRREGULAR' says the same thing without the
    contradiction, and the argmax is still available in the detail line.

    Note the honest limit of that measurement: on record 100, 30 minutes of
    clean NSR, argmax and threshold agreed on all 117 windows. So this is a
    fix for screen-vs-alarm disagreement on records that DO contain AF; the
    predicted failure on clean rhythm at a realistic ~2% prevalence is an
    inference from the training prevalence, not something observed here.
    """
    if af_alarm:
        return 'AF ALARM', 'alarm'
    if af_run:
        return f'AF? {af_run}/{AF_CONSEC}', 'warn'
    if af_score is None or not np.isfinite(af_score):
        return '--', 'idle'
    if af_score >= 0.50:
        return 'IRREGULAR', 'warn'
    return 'NSR', 'ok'


def alert_sound(enabled, freq, dur_ms, times=1):
    """Audible alarm on a daemon thread. Windows only, no dependency.

    OFF-THREAD IS MANDATORY. winsound.Beep blocks for its full duration, so a
    400ms alarm on the read loop would drop ~120 samples, and those gaps land
    straight in the RR intervals feeding rmssd, pnn50 and rr_entropy -- the
    alarm would corrupt the very data that raised it. Same trap the Arduino
    sketch already avoids by passing a duration to tone() instead of delay().
    """
    if not enabled or sys.platform != 'win32':
        return

    def _run():
        try:
            import winsound
            for _ in range(times):
                winsound.Beep(freq, dur_ms)
                time.sleep(0.08)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


class Dashboard:
    """Live monitor window: a status header plus ECG, AF trend and RR tachogram.

    WHY IT IS BUILT THIS WAY -- all three decisions came from measurement on
    this machine (TkAgg, matplotlib 3.10, 11x6.8in figure), not from taste.

    1. BLITTING, because the old code could not keep up. It did a FULL canvas
       redraw every 5 samples via plt.pause() -- ~60 fps at 300 Hz and 72 fps at
       360 Hz replay, so its cost scaled with the sample rate. Frames are now
       gated on the WALL CLOCK, which decouples redraw cost from fs entirely.

    2. dpi is PINNED to 100. Windows display scaling was silently returning 125,
       i.e. 1374x805 px instead of 1100x750 -- 37% more pixels to copy every
       frame for no visible benefit at this size.

    3. THE STATUS READOUTS ARE NATIVE Tk LABELS, NOT matplotlib TEXT. This is
       the big one. Measured cost of drawing ONE matplotlib Text artist into
       this figure, after restore_region:

           11x7.5in canvas, fontsize 24 ....... 25 ms
           11x7.5in canvas, fontsize 10 ....... 26 ms
            5x3.0in canvas, fontsize 24 ....... 7.6 ms

       The cost tracks CANVAS AREA, not font size or antialiasing -- Agg's
       draw_text_image is effectively O(canvas). Four readouts cost ~67 ms per
       frame, against 2.3 ms for the entire 1200-point ECG line. Native Tk
       labels render in microseconds, so the whole header became a tk.Frame
       packed above the canvas. Same frame with the header updating every
       redraw: 8.8 ms, versus 80+ ms with matplotlib text. It also looks better
       -- real font rendering and a frame background that can flash for alarms.

       Backends other than TkAgg fall back to the window title bar (see
       _build_header). Nothing else depends on the header existing.

    Net: ~7.5 ms/frame, ~19% of wall clock at 25 fps, which leaves ample
    headroom to keep up with a 300 Hz stream -- the thing the old plot could
    not do. At 25 fps the trace advances ~8.5 px per frame across an 852 px
    axes, which is what makes the SCROLL smooth. The waveform itself is drawn
    point-for-point: no decimation, no smoothing, no path simplification.
    """

    def __init__(self, fs, window_sec, theme='dark', fps=PLOT_FPS,
                 plot_sec=PLOT_SEC, trend_min=AF_TREND_MIN):
        import matplotlib.pyplot as plt

        self.fs = fs
        self.window_sec = window_sec
        self.plot_sec = plot_sec
        self.trend_min = trend_min
        self.frame_interval = 1.0 / max(fps, 1)
        self.slow_interval = 0.2          # header + tachogram, 5 Hz
        self.next_frame_t = 0.0
        self.next_slow_t = 0.0
        self.closed = False
        self.fail_count = 0
        self.autoscaled = False
        self.c = c = THEMES[theme]
        self._plt = plt

        plt.ion()
        # dpi pinned: see the class docstring.
        fig = plt.figure(figsize=(11, 6.8), dpi=100, facecolor=c['bg'])
        self.fig = fig
        try:
            fig.canvas.manager.set_window_title('ECG monitor - beat + rhythm')
        except Exception:
            pass
        gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 1.1, 1.1], hspace=0.38,
                              left=0.07, right=0.985, top=0.97, bottom=0.09)
        ax_ecg = fig.add_subplot(gs[0])
        ax_af = fig.add_subplot(gs[1])
        ax_rr = fig.add_subplot(gs[2])
        self.ax_ecg, self.ax_af, self.ax_rr = ax_ecg, ax_af, ax_rr

        for ax in (ax_ecg, ax_af, ax_rr):
            ax.set_facecolor(c['bg'])
            ax.grid(True, color=c['grid'], linewidth=0.6)
            ax.tick_params(colors=c['dim'], labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(c['axis'])

        # --- ECG ----------------------------------------------------------
        # xlim is FIXED so the background stays cacheable. The old code called
        # set_xlim(0, len(plot_buffer)) every frame, which re-lays-out the axes
        # and would invalidate the blit background on every single frame.
        ax_ecg.set_xlim(0, fs * plot_sec)
        ax_ecg.set_ylim(-150, 400)
        ax_ecg.set_xticks([k * fs for k in range(plot_sec + 1)])
        ax_ecg.set_xticklabels([str(k - plot_sec) for k in range(plot_sec + 1)])
        ax_ecg.set_ylabel('ECG', color=c['dim'], fontsize=9)
        # Drawn point-for-point and un-simplified: this is a diagnostic
        # waveform, so QRS height and width must be what the filter produced.
        # 1200 points across ~850 px costs 2.3 ms, which is not worth
        # optimising away at the price of clipping an R peak.
        self.line_ecg, = ax_ecg.plot([], [], color=c['ecg'], linewidth=1.0,
                                     solid_joinstyle='miter')
        # Line2D with markers, not scatter(): set_data is cheap, while a
        # PathCollection's set_offsets rebuilds transforms every frame.
        self.m_detect, = ax_ecg.plot([], [], linestyle='none', marker='|',
                                     markersize=9, color=c['detect'])
        self.m_norm, = ax_ecg.plot([], [], linestyle='none', marker='o',
                                   markersize=6, color=c['normal'])
        self.m_abn, = ax_ecg.plot([], [], linestyle='none', marker='o',
                                  markersize=7, color=c['abnormal'])

        # --- AF trend -------------------------------------------------------
        ax_af.set_xlim(-trend_min, 0)
        ax_af.set_ylim(0, 1)
        ax_af.set_ylabel('AF prob', color=c['dim'], fontsize=9)
        ax_af.axhspan(AF_THRESHOLD, 1.0, color=c['thresh'], alpha=0.10, zorder=0)
        ax_af.axhline(AF_THRESHOLD, color=c['thresh'], linestyle='--', linewidth=1.0)
        ax_af.text(-trend_min + 0.05, AF_THRESHOLD + 0.03,
                   f'alarm {AF_THRESHOLD:.2f}', color=c['thresh'], fontsize=8)
        self.line_af, = ax_af.plot([], [], color=c['af'], linewidth=1.4,
                                   marker='o', markersize=4, drawstyle='steps-post')
        # Gated windows are drawn, not omitted: a gap in the trend must not be
        # mistakable for a run of normal scores.
        self.m_af_gated, = ax_af.plot([], [], linestyle='none', marker='x',
                                      markersize=6, color=c['idle'])

        # --- RR tachogram ---------------------------------------------------
        # The rhythm model's input, made visible. NSR is a tight band, AF is a
        # scatter cloud -- the fastest way to tell a true positive from artifact.
        ax_rr.set_xlim(-window_sec, 0)
        ax_rr.set_ylim(RR_PLOT_LO, RR_PLOT_HI)
        ax_rr.set_ylabel('RR (s)', color=c['dim'], fontsize=9)
        ax_rr.set_xlabel('seconds ago', color=c['dim'], fontsize=9)
        ax_rr.axhline(RR_MIN_S_DISPLAY, color=c['thresh'], linestyle=':', linewidth=0.9)
        self.m_rr, = ax_rr.plot([], [], linestyle='none', marker='o',
                                markersize=3.5, color=c['af'])
        self.m_rr_bad, = ax_rr.plot([], [], linestyle='none', marker='o',
                                    markersize=5, color=c['thresh'])

        # Grouped per axes: each group blits on its own cadence, because a Tk
        # blit costs roughly its region area and there is no point repainting a
        # panel whose data has not changed.
        self.groups = [
            ('ecg', ax_ecg, [self.line_ecg, self.m_detect, self.m_norm, self.m_abn]),
            ('af', ax_af, [self.line_af, self.m_af_gated]),
            ('rr', ax_rr, [self.m_rr, self.m_rr_bad]),
        ]
        self.dynamic = [a for _, _, arts in self.groups for a in arts]
        self.bgs = {}

        fig.canvas.mpl_connect('close_event', self._on_close)
        fig.canvas.mpl_connect('resize_event', lambda e: self.recapture())
        plt.show(block=False)

        # THE SINGLE BIGGEST PERFORMANCE FIX IN THIS FILE.
        #
        # set_data() marks its artist stale, which propagates to the figure and
        # makes the Tk backend schedule a draw_idle. flush_events() -- which we
        # must call to keep the window responsive -- then runs it, so every
        # "blit" frame was silently followed by a FULL redraw of all three axes.
        # Measured: 85.8 ms/frame with the idle draw, 8.5 ms without, i.e. the
        # blitting was buying nothing at all. Profiling made it obvious: 113
        # Line2D draws per frame when only 4 artists were meant to be drawn.
        #
        # Setting fig.stale = False does NOT help, because the callback is
        # already queued in Tk (measured: 87.0 ms, unchanged). The fix is to
        # stop scheduling it. Safe here only because this class owns the entire
        # render loop: every frame blits the ECG axes explicitly, and the one
        # structural event that needs a real redraw -- resize -- is wired to
        # recapture(), which calls canvas.draw() directly.
        try:
            fig.canvas.draw_idle = lambda *a, **k: None
        except Exception:
            pass

        self._build_header()
        self.recapture()

    # -- header -------------------------------------------------------------
    def _build_header(self):
        """Native Tk readouts above the canvas; window title if not TkAgg."""
        self.header = None
        try:
            import tkinter as tk
            widget = self.fig.canvas.get_tk_widget()
            root = widget.master
            c = self.c
            bar = tk.Frame(root, bg=c['bg'])
            bar.pack(side='top', fill='x', before=widget)
            mk = lambda size, colour, bold=False: tk.Label(
                bar, text='', bg=c['bg'], fg=colour,
                font=('Consolas', size, 'bold' if bold else 'normal'))
            self.l_state = mk(26, c['idle'], bold=True)
            self.l_hr = mk(26, c['fg'])
            self.l_beat = mk(15, c['dim'])
            self.l_detail = mk(10, c['dim'])
            for w in (self.l_state, self.l_hr, self.l_beat):
                w.pack(side='left', padx=14)
            self.l_detail.pack(side='right', padx=14)
            self.header = bar
            self.l_state.config(text='WARMUP')
            # The pan/zoom toolbar is a light-grey strip across the bottom of a
            # black monitor, and nothing here is meant to be panned -- every
            # axis is pinned so the backgrounds stay cacheable.
            tb = getattr(self.fig.canvas.manager, 'toolbar', None)
            if tb is not None:
                tb.pack_forget()
        except Exception:
            # Any non-Tk backend, or a headless Tk: fall back to the title bar,
            # which costs nothing and keeps every number visible.
            self.header = None

    def _set_header(self, state, colour_key, hr, beat, detail, flash):
        c = self.c
        hr_txt = 'HR --' if hr is None else f'HR {hr:.0f}'
        if self.header is None:
            try:
                self.fig.canvas.manager.set_window_title(
                    f'{state}   {hr_txt}   beat {beat}   {detail}')
            except Exception:
                pass
            return
        bg = c['bg']
        if flash == 'alarm':
            # 1 Hz flash off the wall clock, so it is independent of the fps.
            bg = '#2a0000' if int(time.monotonic() * 2) % 2 else c['bg']
        elif flash == 'ectopy':
            bg = '#2a2000'
        self.header.config(bg=bg)
        for w, txt, fg in ((self.l_state, state, c[colour_key]),
                           (self.l_hr, hr_txt, c['fg']),
                           (self.l_beat, f'beat {beat}',
                            c['dim'] if _is_normal(beat) else c['abnormal']),
                           (self.l_detail, detail, c['dim'])):
            w.config(text=txt, fg=fg, bg=bg)

    # -- infrastructure ----------------------------------------------------
    def _on_close(self, _evt):
        # Without this, closing the window throws into main()'s blanket handler
        # and prints an error once PER SAMPLE. Now the session just keeps
        # logging headlessly.
        self.closed = True

    def recapture(self):
        """Re-cache the per-axes backgrounds. Rare: startup, resize, autoscale."""
        if self.closed:
            return
        # Hide the dynamic artists first, or they get baked into the cached
        # background and leave ghost trails behind the live ones.
        for a in self.dynamic:
            a.set_visible(False)
        self.fig.canvas.draw()
        self.bgs = {key: self.fig.canvas.copy_from_bbox(ax.bbox)
                    for key, ax, _ in self.groups}
        for a in self.dynamic:
            a.set_visible(True)
        # Repaint everything immediately. Without this the canvas is left
        # showing the bare background -- the draw above happened with the
        # dynamic artists hidden -- and the slow panels stay blank until their
        # own cadence comes round, which for the AF trend is up to 15 s. The
        # ECG hides the problem by blitting 30x a second; the trend does not.
        self._blit({key for key, _, _ in self.groups})

    def _blit(self, keys):
        """Blit just the named axes. One flush_events for the whole batch."""
        if self.closed or not self.bgs:
            return
        try:
            for key, ax, arts in self.groups:
                if key not in keys:
                    continue
                self.fig.canvas.restore_region(self.bgs[key])
                for a in arts:
                    ax.draw_artist(a)
                self.fig.canvas.blit(ax.bbox)
            self.fig.canvas.flush_events()
            self.fail_count = 0
        except Exception as e:
            self.fail_count += 1
            if self.fail_count >= 3:
                self.closed = True
                print(f"Plot disabled after repeated draw failures: "
                      f"{type(e).__name__}: {e}")

    def frame(self, now, plot_buffer, total_samples, beat_positions, classified,
              status):
        """Per-sample entry point. Rate-limits internally and returns fast.

        Called once per accepted sample, so the early return matters more than
        anything else in here.
        """
        if self.closed or now < self.next_frame_t or len(plot_buffer) < 10:
            return
        self.next_frame_t = now + self.frame_interval
        # First real look at the signal: scale to it immediately rather than
        # waiting for the 15 s rhythm tick. The default (-150, 400) is one
        # sensor's gain, and when the trace overflows it the R peaks clip AND
        # the beat markers -- which snap to the peak -- land off-screen, so the
        # window looks broken exactly when someone first plugs the device in.
        if not self.autoscaled and len(plot_buffer) >= 2 * self.fs:
            self.autoscaled = True
            self.autoscale_ecg(plot_buffer, force=True)
        keys = {'ecg'}
        self.update_ecg(plot_buffer, total_samples, beat_positions, classified)
        if now >= self.next_slow_t:
            # Numbers and the 60s tachogram do not need the ECG's frame rate.
            self.next_slow_t = now + self.slow_interval
            self.update_tacho(beat_positions, total_samples)
            self._set_header(*status)
            keys.add('rr')
        self._blit(keys)

    def close(self):
        if not self.closed:
            try:
                self._plt.close(self.fig)
            except Exception:
                pass
        self.closed = True

    # -- data updates -------------------------------------------------------
    def update_ecg(self, plot_buffer, total_samples, beat_positions, classified):
        if self.closed or not plot_buffer:
            return
        buf = list(plot_buffer)
        n = len(buf)
        self.line_ecg.set_data(np.arange(n), buf)
        origin = total_samples - n          # absolute position of buf[0]

        def project(positions):
            xs, ys = [], []
            snap = int(0.06 * self.fs)
            for pos in positions:
                i = pos - origin
                if not (0 <= i < n):
                    continue
                # Snap to the local max of the DISPLAYED trace. Peaks were found
                # on bandpass_filter (zero-phase filtfilt) but this trace is the
                # causal lfilter, which has group delay -- so an unsnapped
                # marker sits a few samples left of the visible peak. Do not
                # "fix" this by changing the detector.
                lo, hi = max(0, i - snap), min(n, i + snap + 1)
                j = lo + int(np.argmax(buf[lo:hi]))
                xs.append(j)
                ys.append(buf[j])
            return xs, ys

        self.m_detect.set_data(*project(beat_positions))
        norm = [p for p, lab in classified if _is_normal(lab)]
        abn = [p for p, lab in classified if not _is_normal(lab)]
        self.m_norm.set_data(*project(norm))
        self.m_abn.set_data(*project(abn))

    def autoscale_ecg(self, plot_buffer, force=False):
        """Rare: once at startup, then once per rhythm tick. The hardcoded
        (-150, 400) is one sensor's gain; on other hardware the trace leaves
        the axes and reads as a dead lead."""
        if self.closed or len(plot_buffer) < self.fs:
            return
        buf = np.fromiter(plot_buffer, float)
        lo, hi = np.percentile(buf, [1, 99])
        # Percentiles set the baseline, but the R peak IS the 99.9th percentile
        # of an ECG -- clipping it would hide the beat markers, which snap to
        # the peak. So take the true extremes for the top of the range.
        lo, hi = min(lo, buf.min()), max(hi, buf.max())
        pad = max(0.10 * (hi - lo), 20.0)
        new = (lo - pad, hi + pad)
        cur = self.ax_ecg.get_ylim()
        span = max(cur[1] - cur[0], 1e-9)
        if (force or abs(new[0] - cur[0]) / span > 0.15
                or abs(new[1] - cur[1]) / span > 0.15):
            self.ax_ecg.set_ylim(*new)
            self.recapture()

    def on_rhythm(self, af_hist, now_sec, plot_buffer):
        """Rhythm-tick entry point: the slow panel, 4x per minute."""
        if self.closed:
            return
        self.autoscale_ecg(plot_buffer)     # may recapture, so do it first
        self.update_trend(af_hist, now_sec)
        self._blit({'af'})

    def update_trend(self, af_hist, now_sec):
        if self.closed:
            return
        gx, gy, ox, oy = [], [], [], []
        for t_sec, score, status in af_hist:
            x = (t_sec - now_sec) / 60.0
            if score is None:
                gx.append(x)
                gy.append(0.03)
            else:
                ox.append(x)
                oy.append(score)
        self.line_af.set_data(ox, oy)
        self.m_af_gated.set_data(gx, gy)

    def update_tacho(self, beat_positions, total_samples):
        if self.closed:
            return
        pos = np.fromiter(beat_positions, dtype=np.int64)
        if pos.size < 2:
            self.m_rr.set_data([], [])
            self.m_rr_bad.set_data([], [])
            return
        rr = np.diff(pos) / self.fs
        t = (pos[1:] - total_samples) / self.fs
        ok = (rr >= RR_MIN_S_DISPLAY) & (rr <= RR_MAX_S_DISPLAY)
        self.m_rr.set_data(t[ok], rr[ok])
        # Rejected intervals are clipped into view rather than dropped, so a
        # window full of rejects looks full rather than empty.
        self.m_rr_bad.set_data(t[~ok], np.clip(rr[~ok], RR_PLOT_LO, RR_PLOT_HI))


def _is_normal(label):
    return str(label).strip().lower() in ('normal', '--', 'none')


RHYTHM_CSV_HEADER = [
    'timestamp', 'sample_pos', 'status', 'n_rr_used', 'n_rr_rejected',
    'n_drr_used', 'coverage', 'leadoff',
    'mean_rr', 'hr', 'hrv_sdnn', 'rmssd', 'pnn50', 'cv_rr', 'rr_entropy',
    'p_nsr', 'p_afib', 'p_aflutter', 'af_score', 'rhythm_top', 'af_run', 'alarm',
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', default=DEFAULT_PORT)
    ap.add_argument('--baud', type=int, default=DEFAULT_BAUD)
    ap.add_argument('--fs', type=int, default=DEFAULT_FS,
                    help='sampling rate in Hz. MEASURE THIS -- see check_sample_rate')
    ap.add_argument('--replay', default=None,
                    help='wfdb record path, or synth:{nsr,af,nsr-af-nsr,flat}')
    ap.add_argument('--speed', type=float, default=1.0,
                    help='replay speed; 0 = as fast as possible (skips the fs check)')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--limit-sec', type=float, default=None)
    ap.add_argument('--theme', choices=('dark', 'light'), default='dark')
    ap.add_argument('--plot-fps', type=float, default=PLOT_FPS)
    ap.add_argument('--quiet', action='store_true',
                    help='suppress the per-beat lines; keep the rhythm blocks')
    ap.add_argument('--sound', action='store_true',
                    help='audible host alarm (Windows). Never writes to the port.')
    args = ap.parse_args(argv)

    fs = args.fs
    window_size = fs * 5                    # 5-second window for the beat model

    scorer = RhythmScorer()
    model = joblib.load('ecg_model.pkl')
    rhythm_hop = RHYTHM_HOP_SEC * fs
    rhythm_history = scorer.window_sec * fs
    beat_refractory = int(0.20 * fs)

    # --- input -------------------------------------------------------------
    if args.replay:
        samples, leadoff_at, desc = load_replay_source(args.replay, fs)
        if args.limit_sec:
            samples = samples[:int(args.limit_sec * fs)]
        ser = ReplaySerial(samples, fs, args.speed, leadoff_at)
        print(f"REPLAY: {desc} at fs={fs}, speed={args.speed or 'max'}")
    else:
        import serial
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Serial: {args.port} @ {args.baud}, fs={fs}")

    # --- CSV logging -------------------------------------------------------
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = open(f'live_readings_{timestamp_str}.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'rr_interval', 'heart_rate', 'hrv',
                         'amplitude', 'qrs_width', 'prediction'])
    # A second file rather than extra columns: the beat path fires every ~3s and
    # the rhythm path every 15s, so merging them would repeat each verdict ~5x
    # and would change a header that existing recordings depend on. Same
    # timestamp string, so a session's two files pair by name.
    rhythm_file = open(f'live_rhythm_{timestamp_str}.csv', 'w', newline='')
    rhythm_writer = csv.writer(rhythm_file)
    rhythm_writer.writerow(RHYTHM_CSV_HEADER)

    # --- beat-model state (unchanged) --------------------------------------
    buffer = []

    # --- rhythm-path state -------------------------------------------------
    # total_samples is the absolute sample clock and the only new counter
    # needed: buffer is a suffix of the stream and the analysed chunk is
    # buffer[-window_size:], so chunk[i] sits at total_samples-window_size+i.
    # No separate start variable, so no drift. Python ints, not int32 -- at
    # 300 Hz an int32 sample counter overflows after ~83 days.
    total_samples = 0
    beat_positions = deque()                # absolute positions, pruned to 60s
    last_beat_pos = -10**18
    leadoff_marks = deque()
    last_rhythm_pos = 0
    last_good_pos = None
    af_run = clear_run = 0
    af_alarm = False
    rhythm_state = 'WARMUP'
    rhythm_colour = 'idle'
    rhythm_detail = ''
    last_af_score = float('nan')
    last_hr = None
    prediction = '--'
    t_stream_start = None
    fs_warned = False
    fs_fatal = False
    next_fs_check = FS_CHECK_AFTER_SEC * fs

    # --- display state -----------------------------------------------------
    # classified_beats holds only the beats the BEAT model actually scored --
    # one per chunk, not every detected peak -- so the plot can distinguish
    # "detected" from "classified" instead of implying the model saw them all.
    classified_beats = deque()
    beat_labels = deque(maxlen=ECTOPY_N)
    ectopy_alert = False
    af_hist = deque(maxlen=max(1, AF_TREND_MIN * 60 // RHYTHM_HOP_SEC))

    # --- live dashboard ----------------------------------------------------
    plot_buffer = deque(maxlen=fs * PLOT_SEC)
    dash = None
    if not args.no_plot:
        try:
            dash = Dashboard(fs, scorer.window_sec, theme=args.theme,
                             fps=args.plot_fps)
        except Exception as e:
            print(f"Plot unavailable ({type(e).__name__}: {e}); continuing headless.")
            dash = None

    # Causal filter for a stable plot, separate from the batch filtfilt used for
    # predictions.
    b_coef, a_coef = butter(2, [0.5 / (0.5 * fs), 40 / (0.5 * fs)], btype='band')
    zi = lfilter_zi(b_coef, a_coef) * 0

    print("Starting live ECG monitoring... (Ctrl+C to stop)")

    while True:
        try:
            raw = ser.readline()
            if raw == b'':
                print("Input exhausted.")
                break
            line = raw.decode('utf-8', errors='ignore').strip()

            if line == '!':
                print("Lead off - check electrode contact")
                leadoff_marks.append(total_samples)
                continue

            if line.isdigit():
                value = int(line)
                buffer.append(value)
                total_samples += 1
                if t_stream_start is None:
                    t_stream_start = time.monotonic()

                filtered_value, zi = lfilter(b_coef, a_coef, [value], zi=zi)
                plot_buffer.append(filtered_value[0])

                # Frames are gated on the WALL CLOCK, not on a sample count.
                # The old "every 5 samples" rule made redraw cost scale with fs
                # (60 fps at 300 Hz, 72 at 360) and made --speed 0 replays
                # crawl. A fixed cap decouples the two.
                if dash is not None and not dash.closed:
                    dash.frame(time.monotonic(), plot_buffer, total_samples,
                               beat_positions, classified_beats,
                               (rhythm_state, rhythm_colour, last_hr, prediction,
                                rhythm_detail,
                                'alarm' if af_alarm
                                else 'ectopy' if ectopy_alert else None))

                # --- sample-rate sanity, skipped when replaying flat out ----
                if (args.speed != 0 and not fs_fatal
                        and total_samples >= next_fs_check):
                    next_fs_check = total_samples + FS_RECHECK_SEC * fs
                    sev, measured = check_sample_rate(
                        total_samples, t_stream_start, fs, fs_warned)
                    if sev != 'ok':
                        fs_warned = True
                        print(f"\n*** SAMPLE RATE MISMATCH: configured fs={fs} Hz, "
                              f"measured {measured:.1f} Hz ***")
                        print("    Either the device is slower than assumed, OR "
                              "samples are being dropped in the serial buffer.")
                        print("    Both corrupt every RR-derived feature. Re-run "
                              f"with --fs {measured:.0f} once you trust the value.\n")
                    if sev == 'fatal':
                        fs_fatal = True
                        rhythm_state = 'FS MISMATCH'
                        rhythm_colour = 'alarm'
                        rhythm_detail = (f"configured {fs} Hz, measured "
                                         f"{measured:.0f} Hz - rhythm scoring off")
                        print("    Deviation over 15% - rhythm scoring disabled. "
                              "The beat model continues.\n")

            if len(buffer) >= window_size:
                signal_chunk = np.array(buffer[-window_size:], dtype=float)
                filtered = bandpass_filter(signal_chunk, fs=fs)

                peaks, _ = find_peaks(filtered, distance=fs*0.4,
                                      height=np.mean(filtered)+np.std(filtered))

                # --- beat bookkeeping for the 60-second rhythm history ------
                # Placed BEFORE the two early continues below on purpose: those
                # bail out on noisy stretches, which is exactly when a
                # low_quality row should be logged rather than silently skipped,
                # and a 1-2 peak window still has beats worth keeping.
                #
                # buffer is trimmed to 2s while the window is 5s, so consecutive
                # windows overlap and the same physical beat is detected two or
                # three times. De-duplicating on ABSOLUTE position with a
                # refractory gap is what makes the history possible. Verified on
                # MIT-BIH 100: 2272 of 2273 annotated beats recovered, zero
                # duplicates. Keying on absolute position also makes this robust
                # to either trim policy (1s on the bail-out path, 2s below).
                chunk_start = total_samples - window_size
                for p in peaks:
                    abs_pos = chunk_start + int(p)
                    if abs_pos - last_beat_pos >= beat_refractory:
                        beat_positions.append(abs_pos)
                        last_beat_pos = abs_pos
                # A fixed 60-second span ending at the newest sample -- the same
                # geometry as the training windows, not "the last N beats".
                cutoff = total_samples - rhythm_history
                while beat_positions and beat_positions[0] < cutoff:
                    beat_positions.popleft()
                while leadoff_marks and leadoff_marks[0] < cutoff:
                    leadoff_marks.popleft()
                # Only ever drawn, so prune to the visible window, not to 60s.
                plot_cutoff = total_samples - fs * PLOT_SEC
                while classified_beats and classified_beats[0][0] < plot_cutoff:
                    classified_beats.popleft()

                # --- rhythm evaluation, on its own cadence -----------------
                if not fs_fatal and total_samples - last_rhythm_pos >= rhythm_hop:
                    last_rhythm_pos = total_samples
                    try:
                        v = scorer.score(beat_positions, fs, total_samples,
                                         leadoff=len(leadoff_marks))

                        if v['status'] == 'ok':
                            # A quality gap must not make two distant windows
                            # look consecutive; a single bad window must not
                            # discard a building streak either.
                            if (last_good_pos is not None
                                    and total_samples - last_good_pos > 2 * rhythm_hop):
                                af_run = clear_run = 0
                            last_good_pos = total_samples
                            last_af_score = v['af_score']
                            if v['af_score'] >= AF_THRESHOLD:
                                af_run += 1
                                clear_run = 0
                            else:
                                clear_run += 1
                                af_run = 0
                            was_alarm = af_alarm
                            if af_run >= AF_CONSEC:
                                af_alarm = True
                            if clear_run >= AF_CLEAR:
                                af_alarm = False
                            rhythm_state, rhythm_colour = rhythm_label(
                                v['af_score'], af_run, af_alarm)
                            pr = v['proba']
                            cc = scorer.class_col
                            f = v['feats']
                            last_hr = f['hr']
                            rhythm_detail = (
                                f"AF {v['af_score']:.2f}   "
                                f"NSR {pr[cc['NSR']]:.2f} / AFib {pr[cc['AFib']]:.2f}"
                                f" / AFl {pr[cc['AFlutter']]:.2f}   "
                                f"cov {v['coverage']:.2f}   "
                                f"rr {f['n_rr_used']}/{f['n_rr_rejected']}rej   "
                                f"{fmt_hms(total_samples / fs)}")
                            af_hist.append((total_samples / fs, v['af_score'],
                                            'ok'))
                            print(f"  --- rhythm {scorer.window_sec}s @ "
                                  f"{fmt_hms(total_samples / fs)} "
                                  + '-' * 24)
                            print(f"      {rhythm_state:<12s} AF p={v['af_score']:.3f}"
                                  f"   (NSR {pr[cc['NSR']]:.2f}"
                                  f" / AFib {pr[cc['AFib']]:.2f}"
                                  f" / AFlutter {pr[cc['AFlutter']]:.2f})")
                            print(f"      HR {f['hr']:.0f}  SDNN {f['hrv_sdnn']:.3f}"
                                  f"  RMSSD {f['rmssd']:.3f}"
                                  f"  pNN50 {f['pnn50']:.2f}"
                                  f"  cov {v['coverage']:.2f}"
                                  f"  rr {f['n_rr_used']}/{f['n_rr_rejected']}rej")

                            # Transitions only, so a long episode does not
                            # reprint the banner four times a minute.
                            if af_alarm and not was_alarm:
                                print("\a  " + "*" * 58)
                                print("  ***  AF ALARM  -  "
                                      f"{AF_CONSEC} consecutive {scorer.window_sec}s "
                                      f"windows >= {AF_THRESHOLD:.2f}  ***")
                                print("  ***  Not a diagnosis. Check electrodes, "
                                      "then a doctor.  ***")
                                print("  " + "*" * 58)
                                alert_sound(args.sound, 1200, 350, times=3)
                            elif was_alarm and not af_alarm:
                                print("  --- AF alarm cleared "
                                      f"({AF_CLEAR} windows below threshold) ---")
                        else:
                            af_hist.append((total_samples / fs, None, v['status']))
                            if v['status'] == 'warmup':
                                rhythm_state, rhythm_colour = 'WARMUP', 'idle'
                                left = scorer.window_sec - total_samples / fs
                                rhythm_detail = (f"filling beat history, "
                                                 f"{max(left, 0):.0f}s to first verdict")
                                print(f"  [rhythm] warming up, "
                                      f"{total_samples/fs:.0f}/{scorer.window_sec}s "
                                      f"of beat history")
                            else:
                                # Streak is HELD, not reset; the staleness rule
                                # above handles genuine gaps.
                                rhythm_state = v['status'].upper().replace('_', ' ')
                                rhythm_colour = 'warn'
                                f = v['feats'] or {}
                                cov = v['coverage']
                                rhythm_detail = (
                                    f"rhythm not scored   rr {f.get('n_rr_used', 0)}"
                                    f"   cov {cov:.2f}   "
                                    f"{fmt_hms(total_samples / fs)}")
                                print(f"  [rhythm] {rhythm_state}"
                                      f"  rr={f.get('n_rr_used', 0)}"
                                      f"  cov={cov:.2f}")

                        # Log EVERY evaluation, including warmup / leadoff /
                        # low_quality. Silent skips would make the log unable to
                        # distinguish "no AF" from "never scored anything". One
                        # row per 15s is 240 rows/hour.
                        f = v['feats'] or {}
                        pr = v['proba']
                        rhythm_writer.writerow(
                            [datetime.now().isoformat(), total_samples, v['status'],
                             f.get('n_rr_used', ''), f.get('n_rr_rejected', ''),
                             f.get('n_drr_used', ''),
                             '' if np.isnan(v['coverage']) else f"{v['coverage']:.4f}",
                             len(leadoff_marks)]
                            + [f.get(k, '') for k in scorer.features]
                            + ([f"{pr[scorer.class_col['NSR']]:.6f}",
                                f"{pr[scorer.class_col['AFib']]:.6f}",
                                f"{pr[scorer.class_col['AFlutter']]:.6f}",
                                f"{v['af_score']:.6f}", v['rhythm_top']]
                               if pr is not None else ['', '', '', '', ''])
                            + [af_run, int(af_alarm)])
                        rhythm_file.flush()

                        # The two slow panels refresh on the rhythm cadence,
                        # not per frame -- they only change this often.
                        if dash is not None and not dash.closed:
                            dash.on_rhythm(af_hist, total_samples / fs, plot_buffer)
                    except Exception as e:
                        # Narrow, and typed: the loop's outer handler would make
                        # a rhythm bug indistinguishable from a serial glitch and
                        # print it once per sample.
                        print(f"Rhythm path error: {type(e).__name__}: {e}")

                # --- beat model (behaviour unchanged) ----------------------
                if len(peaks) < 3:
                    buffer = buffer[-int(fs*1):]
                    continue

                rr_intervals = []
                for i in range(1, len(peaks)):
                    rr = (peaks[i] - peaks[i-1]) / fs
                    if 0.3 < rr < 2.0:
                        rr_intervals.append(rr)

                if len(rr_intervals) < 2:
                    buffer = buffer[-int(fs*1):]
                    continue

                latest_rr = rr_intervals[-1]
                heart_rate = 60 / latest_rr
                hrv = np.std(rr_intervals[-5:]) if len(rr_intervals) > 1 else 0.0

                last_peak = peaks[-1]
                half_win = int(0.1 * fs)
                start = max(0, last_peak - half_win)
                end = min(len(filtered), last_peak + half_win)
                segment = filtered[start:end]
                amplitude = filtered[last_peak]

                if len(segment) > 0 and amplitude != 0:
                    threshold = amplitude * 0.5
                    above = np.where(np.abs(segment) > np.abs(threshold))[0]
                    qrs_width = (above[-1] - above[0]) / fs if len(above) > 1 else 0.0
                else:
                    qrs_width = 0.0

                row = pd.DataFrame([{
                    'rr_interval': latest_rr,
                    'heart_rate': heart_rate,
                    'hrv': hrv,
                    'amplitude': amplitude,
                    'qrs_width': qrs_width
                }])

                prediction = model.predict(row)[0]
                last_hr = heart_rate
                # Only the LAST peak of the chunk is classified, so this is the
                # one beat the beat model actually saw.
                classified_beats.append((chunk_start + int(last_peak), prediction))
                beat_labels.append(prediction)

                n_abn = sum(1 for lab in beat_labels if not _is_normal(lab))
                if not ectopy_alert and n_abn >= ECTOPY_ON:
                    ectopy_alert = True
                    print(f"\a  !! SUSTAINED ECTOPY: {n_abn} of the last "
                          f"{len(beat_labels)} classified beats abnormal")
                    alert_sound(args.sound, 2200, 120, times=2)
                elif ectopy_alert and n_abn <= ECTOPY_OFF:
                    ectopy_alert = False
                    print(f"  -- ectopy settled ({n_abn}/{len(beat_labels)} abnormal)")

                if not args.quiet:
                    print(f"t={fmt_hms(total_samples / fs)} | HR {heart_rate:3.0f} | "
                          f"beat {str(prediction):<9s} | {rhythm_state} "
                          f"AF {last_af_score:.2f}")

                csv_writer.writerow([
                    datetime.now().isoformat(),
                    latest_rr, heart_rate, hrv, amplitude, qrs_width, prediction
                ])
                csv_file.flush()

                buffer = buffer[-int(fs*2):]

        except KeyboardInterrupt:
            print("Stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    csv_file.close()
    rhythm_file.close()
    if dash is not None:
        dash.close()
    try:
        ser.close()
    except Exception:
        pass
    print(f"Wrote live_readings_{timestamp_str}.csv and "
          f"live_rhythm_{timestamp_str}.csv")
    return {'timestamp': timestamp_str, 'total_samples': total_samples,
            'beat_positions': list(beat_positions), 'af_alarm': af_alarm,
            'rhythm_state': rhythm_state}


if __name__ == '__main__':
    main()
