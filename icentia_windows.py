# %% [markdown]
# # Icentia11k 1-minute rhythm windows (Colab)
#
# Turns the beat annotations downloaded by `icentia_scan.py` into **one table of
# 1-minute windows, each labelled NSR / AFib / AFlutter**, with RR-derived
# features. This is the supervision the existing beat-level `ecg_model.pkl`
# cannot provide: AFib is a property of a *minute* of beats, not of any one beat.
#
# This stage reads **only** `beats/<pid>.npz` -- beat times and rhythm spans. It
# never opens a raw waveform, never imports `wfdb`, and makes no network
# requests, so the whole thing runs in minutes and is testable offline.
#
# **The table ships unfiltered.** The builder applies only structural floors (a
# window needs some rhythm coverage and at least 2 usable RR intervals); every
# quality signal -- coverage, purity, beat count, noise-beat fraction -- is kept
# as a column. So the coverage threshold, minimum beat count and purity cut are
# all tunable later with a `df.query`, with no re-run of this notebook.
#
# Run order: every cell top to bottom. The inventory and estimate cells print
# what you have *before* the build spends any time on it.

# %%
# --- Setup -------------------------------------------------------------------
IN_COLAB = False
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tqdm", "pyarrow"], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
    WORK_DIR = "/content/drive/MyDrive/icentia_scan"
else:
    WORK_DIR = "./icentia_scan"

import os, re, json, glob, gzip, time, shutil
from collections import Counter
import numpy as np
import pandas as pd

BEATS_DIR = os.path.join(WORK_DIR, "beats")           # input: per-patient .npz
INDEX_GZ = os.path.join(WORK_DIR, "label_index.json.gz")

# --- window knobs ------------------------------------------------------------
WINDOW_SEC = 60          # the analysis window; ~95 beats gives a stable SDNN
STRIDE_SEC = 60          # non-overlapping. See the note in windows_for_segment.
FS_EXPECTED = 250        # asserted against the npz; W is derived, never literal
SEG_LEN = 1_048_577      # constant in every .hea checked, including segment 49

# --- annotation codes --------------------------------------------------------
# WFDB label_store values. Icentia's beat labels are N/a/V/A/S plus Q
# (unclassifiable). Code 28 is '+', a rhythm-CHANGE marker, not a beat -- and it
# is emitted at the same sample as a real beat, so leaving it in injects a
# zero-length RR at every rhythm boundary. 14 '~', 16 '|', 22 '"' and 0 are also
# not beats.
BEAT_CODES = (1, 4, 5, 8, 9, 13)
Q_CODE = 13              # unclassifiable beat
V_CODE = 5               # PVC
SV_CODES = (4, 8, 9)     # aberrated APC / APC / SVPB
NONBEAT_CODES = (0, 14, 16, 22, 28)

# Q beats sit 100% outside the rhythm spans -- they are beat detections inside
# the noise gaps. Including them fabricates plausible-looking RR intervals out of
# artifact; excluding them instead bridges each gap into one huge RR, which the
# range filter below then removes. q_frac is kept as the quality signal.
EXCLUDE_Q_FROM_RR = True

# --- RR filtering ------------------------------------------------------------
RR_MIN_S, RR_MAX_S = 0.24, 2.0   # HR 250..30
# 0.24 is deliberately looser than live_predict.py's 0.3: RR irregularity IS the
# AFib signal and rapid AF genuinely produces sub-0.3s cycles, so clipping at 0.3
# would erase part of what we are trying to detect. n_rr_rejected keeps the cost
# of this filter auditable.
PNN_THRESH_S = 0.05      # pNN50 convention
ENTROPY_BINS = 16        # fixed bins over [RR_MIN_S, RR_MAX_S] -> comparable

# --- analysis-time thresholds (applied in the report, NOT in the builder) -----
MIN_BEATS = 30           # 60s at HR 30; fewer means missing annotations
MIN_COVERAGE = 0.5       # fraction of the minute that carries a rhythm label
MIN_PURITY = 0.7         # fraction of the ANNOTATED time in the winning rhythm

# --- run control -------------------------------------------------------------
PART_PATIENTS = 200      # patients per parquet part; part existence = resume
OVERWRITE = False        # a re-run resumes; it never silently rebuilds
LOCAL_CACHE = True       # in Colab, bulk-copy beats/ off Drive before the build

RHYTHM_ORDER = ("NSR", "AFib", "AFlutter")   # canonical label_id 0/1/2
PARTS_DIR = os.path.join(WORK_DIR, "windows")
TABLE_OUT = os.path.join(WORK_DIR, f"windows_{WINDOW_SEC}s.parquet")
SUMMARY_OUT = os.path.join(WORK_DIR, f"windows_summary_{WINDOW_SEC}s.json")
DONE_JSONL = os.path.join(WORK_DIR, f"windows_done_{WINDOW_SEC}s.jsonl")
FAILED_JSONL = os.path.join(WORK_DIR, f"windows_failed_{WINDOW_SEC}s.jsonl")

KEY = ["patient_id", "segment", "window_index"]

FRAC_COLS = [f"frac_{n.lower()}" for n in RHYTHM_ORDER]
FEATURE_COLS = ["mean_rr", "hr", "hrv_sdnn", "rmssd", "pnn50", "cv_rr", "rr_entropy"]
COLUMNS = (KEY + ["start_sample", "end_sample", "label_id"]
           + FRAC_COLS + ["frac_covered", "frac_winner", "purity"]
           + ["n_beats", "n_q", "n_v", "n_sv", "q_frac"]
           + ["n_rr_used", "n_rr_rejected", "n_drr_used"]
           + FEATURE_COLS)
DTYPES = dict(
    [("patient_id", "object"), ("segment", "uint8"), ("window_index", "uint16"),
     ("start_sample", "int32"), ("end_sample", "int32"), ("label_id", "int8")]
    + [(c, "float32") for c in FRAC_COLS]
    + [("frac_covered", "float32"), ("frac_winner", "float32"), ("purity", "float32")]
    + [("n_beats", "uint16"), ("n_q", "uint16"), ("n_v", "uint16"), ("n_sv", "uint16"),
       ("q_frac", "float32")]
    + [("n_rr_used", "uint16"), ("n_rr_rejected", "uint16"), ("n_drr_used", "uint16")]
    + [(c, "float32") for c in FEATURE_COLS]
)
# window_index is uint16 rather than uint8 on purpose: at WINDOW_SEC=10 there are
# 419 windows per segment and uint8 would silently wrap.

os.makedirs(WORK_DIR, exist_ok=True)
print(f"colab={IN_COLAB}  work_dir={WORK_DIR}")
print(f"window={WINDOW_SEC}s  stride={STRIDE_SEC}s  "
      f"rr=[{RR_MIN_S},{RR_MAX_S}]s  exclude_q={EXCLUDE_Q_FROM_RR}")

# %%
# --- Geometry, labelling and RR features: pure functions, no I/O -------------
_BEAT_ARR = np.array(BEAT_CODES, dtype=np.int64)
_SV_ARR = np.array(SV_CODES, dtype=np.int64)


def window_bounds(seg_len, window_samples, stride_samples):
    """Half-open [start, end) bounds of the WHOLE windows that fit in seg_len.

    n = 1 + (seg_len - W) // stride. At the defaults that is
    1 + (1048577 - 15000) // 15000 = 69, and the trailing 13,577 samples are
    dropped rather than forming a short 70th window.
    """
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window and stride must be positive")
    if seg_len < window_samples:
        z = np.zeros(0, dtype=np.int64)
        return z, z
    n = 1 + (seg_len - window_samples) // stride_samples
    if n >= 65535:
        raise ValueError(f"{n} windows exceeds the uint16 window_index range")
    w0 = np.arange(n, dtype=np.int64) * stride_samples
    return w0, w0 + window_samples


def label_windows(spans, w0, w1, n_rhythms):
    """-> frac (n_win, n_rhythms) float64: fraction of each window in each rhythm.

    spans is int64 (m, 3) of [start, end, rhythm_id] with rhythm_id already
    remapped to the canonical order. Overlaps are computed as one
    (n_win, n_spans) matrix, then summed into one column per rhythm.

    np.add.at would be the obvious way to accumulate by rhythm_id, but it takes
    numpy's unbuffered fancy-index path and is far slower; n_rhythms is fixed at
    3, so a 3-iteration column loop is both quicker and clearer.
    """
    n_win = len(w0)
    acc = np.zeros((n_win, n_rhythms), dtype=np.float64)
    if n_win == 0 or len(spans) == 0:
        return acc
    starts = spans[:, 0].astype(np.int64)
    ends = spans[:, 1].astype(np.int64)
    rid = spans[:, 2].astype(np.int64)
    ov = (np.minimum(w1[:, None], ends[None, :])
          - np.maximum(w0[:, None], starts[None, :]))
    np.clip(ov, 0, None, out=ov)
    for k in range(n_rhythms):
        sel = rid == k
        if sel.any():
            acc[:, k] = ov[:, sel].sum(axis=1)
    return acc / (w1 - w0)[:, None]


def beat_masks(code):
    """Boolean masks over the annotation array, by WFDB label_store."""
    beat = np.isin(code, _BEAT_ARR)
    q = code == Q_CODE
    return {
        "beat": beat,                                    # QRS-bearing, incl. Q
        "rr": (beat & ~q) if EXCLUDE_Q_FROM_RR else beat,
        "q": beat & q,
        "v": code == V_CODE,
        "sv": np.isin(code, _SV_ARR),
    }


def counts_per_window(samples, window_samples, n_win):
    """How many of `samples` fall in each window.

    Uses integer division, so it is only valid when stride == window_samples --
    with overlap a sample belongs to several windows and this silently
    under-counts. That invariant is asserted in windows_for_segment.
    """
    out = np.zeros(n_win, dtype=np.int64)
    if len(samples) == 0 or n_win == 0:
        return out
    wid = samples // window_samples
    wid = wid[(wid >= 0) & (wid < n_win)]
    if len(wid):
        out += np.bincount(wid, minlength=n_win)[:n_win]
    return out


def rr_window_features(beats, window_samples, n_win, fs,
                       rr_min_s=RR_MIN_S, rr_max_s=RR_MAX_S,
                       pnn_thresh_s=PNN_THRESH_S, n_bins=ENTROPY_BINS):
    """Every per-window RR statistic, in a handful of whole-array operations.

        wid  = beats // W
        rr   = diff(beats) / fs
        same = (wid[:-1] == wid[1:]) & (wid[:-1] < n_win)
        ok   = same & (rr >= rr_min) & (rr <= rr_max)

    `same` requires both endpoints of an RR to be in the same window, which makes
    this provably identical to np.diff of each window's beat slice -- and it is
    also what drops beats sitting in the discarded tail remainder.

    Sums accumulate in float64: a one-pass variance in float32 loses about four
    digits and can even go negative on the low-variance NSR windows.
    """
    nan = lambda: np.full(n_win, np.nan, dtype=np.float64)   # noqa: E731
    out = {
        "n_rr_used": np.zeros(n_win, dtype=np.int64),
        "n_rr_rejected": np.zeros(n_win, dtype=np.int64),
        "n_drr_used": np.zeros(n_win, dtype=np.int64),
        "mean_rr": nan(), "hr": nan(), "hrv_sdnn": nan(),
        "rmssd": nan(), "pnn50": nan(), "cv_rr": nan(), "rr_entropy": nan(),
    }
    if n_win == 0 or len(beats) < 2:
        return out

    beats = beats.astype(np.int64)
    wid = beats // window_samples
    rr = np.diff(beats).astype(np.float64) / fs
    wl = wid[:-1]                                   # window of the left endpoint
    same = (wl == wid[1:]) & (wl >= 0) & (wl < n_win)
    ok = same & (rr >= rr_min_s) & (rr <= rr_max_s)

    n_total = np.bincount(wl[same], minlength=n_win)[:n_win] if same.any() \
        else np.zeros(n_win, dtype=np.int64)
    w_ok, rr_ok = wl[ok], rr[ok]
    cnt = (np.bincount(w_ok, minlength=n_win)[:n_win] if ok.any()
           else np.zeros(n_win, dtype=np.int64))
    out["n_rr_used"] = cnt
    out["n_rr_rejected"] = n_total - cnt
    if not ok.any():
        return out

    s1 = np.bincount(w_ok, weights=rr_ok, minlength=n_win)[:n_win]
    s2 = np.bincount(w_ok, weights=rr_ok * rr_ok, minlength=n_win)[:n_win]

    h1 = cnt >= 1
    out["mean_rr"][h1] = s1[h1] / cnt[h1]
    out["hr"][h1] = 60.0 / out["mean_rr"][h1]

    h2 = cnt >= 2
    if h2.any():
        var = (s2[h2] - cnt[h2] * out["mean_rr"][h2] ** 2) / (cnt[h2] - 1)
        sd = np.sqrt(np.clip(var, 0.0, None))        # clip: float cancellation
        out["hrv_sdnn"][h2] = sd
        out["cv_rr"][h2] = sd / out["mean_rr"][h2]

    # Successive-difference features. dRR may only span two RR intervals that are
    # adjacent in the ORIGINAL index and both accepted -- otherwise a rejected
    # noise-bridge interval silently joins two non-consecutive cycles and
    # inflates RMSSD, which is exactly the feature AFib detection leans on.
    if len(rr) >= 2:
        drr = np.diff(rr)
        adj = ok[:-1] & ok[1:] & (wl[:-1] == wl[1:])
        if adj.any():
            w_d, d = wl[:-1][adj], drr[adj]
            nd = np.bincount(w_d, minlength=n_win)[:n_win]
            sd2 = np.bincount(w_d, weights=d * d, minlength=n_win)[:n_win]
            n50 = np.bincount(w_d, weights=(np.abs(d) > pnn_thresh_s).astype(np.float64),
                              minlength=n_win)[:n_win]
            out["n_drr_used"] = nd
            hd = nd >= 1
            out["rmssd"][hd] = np.sqrt(sd2[hd] / nd[hd])
            out["pnn50"][hd] = n50[hd] / nd[hd]

    # Shannon entropy of the RR histogram over FIXED bins, so values are
    # comparable across windows. Flatten (window, bin) into one bincount.
    b = ((rr_ok - rr_min_s) / (rr_max_s - rr_min_s) * n_bins).astype(np.int64)
    np.clip(b, 0, n_bins - 1, out=b)
    hist = np.bincount(w_ok * n_bins + b,
                       minlength=n_win * n_bins)[:n_win * n_bins].reshape(n_win, n_bins)
    tot = hist.sum(axis=1, keepdims=True)
    p = hist / np.maximum(tot, 1)
    ent = -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)
    out["rr_entropy"][h2] = ent[h2]
    return out


def rr_features_reference(beats, window_samples, n_win, fs,
                          rr_min_s=RR_MIN_S, rr_max_s=RR_MAX_S,
                          pnn_thresh_s=PNN_THRESH_S, n_bins=ENTROPY_BINS):
    """Naive per-window scalar implementation of rr_window_features.

    SELF-TEST ORACLE ONLY -- never called in the build path. It loops windows in
    Python and slices the beat array, which is obviously correct and obviously
    too slow. Exactly the role wfdb.rdann plays for the .atr parser in
    icentia_scan.py.
    """
    keys = ["n_rr_used", "n_rr_rejected", "n_drr_used", "mean_rr", "hr",
            "hrv_sdnn", "rmssd", "pnn50", "cv_rr", "rr_entropy"]
    out = {k: (np.zeros(n_win, dtype=np.int64) if k.startswith("n_")
               else np.full(n_win, np.nan)) for k in keys}
    edges = np.linspace(rr_min_s, rr_max_s, n_bins + 1)
    for w in range(n_win):
        lo, hi = w * window_samples, w * window_samples + window_samples
        b = np.asarray([x for x in beats if lo <= x < hi], dtype=np.int64)
        if len(b) < 2:
            continue
        rr_all = np.diff(b).astype(np.float64) / fs
        keep = (rr_all >= rr_min_s) & (rr_all <= rr_max_s)
        rr = rr_all[keep]
        out["n_rr_used"][w] = len(rr)
        out["n_rr_rejected"][w] = len(rr_all) - len(rr)
        if len(rr) >= 1:
            out["mean_rr"][w] = float(np.mean(rr))
            out["hr"][w] = 60.0 / out["mean_rr"][w]
        if len(rr) >= 2:
            out["hrv_sdnn"][w] = float(np.std(rr, ddof=1))
            out["cv_rr"][w] = out["hrv_sdnn"][w] / out["mean_rr"][w]
            idx = np.clip(np.digitize(rr, edges) - 1, 0, n_bins - 1)
            counts = np.bincount(idx, minlength=n_bins)
            p = counts[counts > 0] / len(rr)
            out["rr_entropy"][w] = float(-(p * np.log2(p)).sum())
        if len(rr_all) >= 2:
            d = [rr_all[i + 1] - rr_all[i] for i in range(len(rr_all) - 1)
                 if keep[i] and keep[i + 1]]
            out["n_drr_used"][w] = len(d)
            if d:
                d = np.asarray(d)
                out["rmssd"][w] = float(np.sqrt(np.mean(d * d)))
                out["pnn50"][w] = float(np.mean(np.abs(d) > pnn_thresh_s))
    return out

# %%
# --- Self-test: hand-computed cases + fast-path vs naive-oracle equality -----
# Do not skip or remove this. If the vectorized RR path drifts from the naive
# reference, or the overlap labelling is off by a window, every number
# downstream is quietly wrong. Same discipline as validate() in icentia_scan.py.


def _same(a, b, name, atol=1e-9):
    bad = []
    for k in b:
        x, y = np.asarray(a[k], dtype=np.float64), np.asarray(b[k], dtype=np.float64)
        if x.shape != y.shape:
            bad.append(f"{k}: shape {x.shape} vs {y.shape}")
            continue
        both_nan = np.isnan(x) & np.isnan(y)
        if not np.allclose(np.where(both_nan, 0, x), np.where(both_nan, 0, y),
                           rtol=0, atol=atol, equal_nan=True):
            worst = np.nanmax(np.abs(np.where(both_nan, 0, x) - np.where(both_nan, 0, y)))
            bad.append(f"{k}: max abs diff {worst:g}")
    if bad:
        raise AssertionError(f"{name} mismatch -- " + "; ".join(bad))


def selftest(seed=0, n_random=200, verbose=True):
    rng = np.random.default_rng(seed)
    W, fs = 15000, 250

    # -- geometry -------------------------------------------------------------
    w0, w1 = window_bounds(SEG_LEN, W, W)
    assert len(w0) == 69, len(w0)
    assert w0[0] == 0 and w1[-1] == 1_035_000
    assert np.all(w0[1:] == w1[:-1])                       # no gaps, no overlap
    assert SEG_LEN - int(w1[-1]) == 13_577
    assert len(window_bounds(W - 1, W, W)[0]) == 0          # shorter than a window

    # -- labelling, hand-computed (exact binary fractions) --------------------
    f = label_windows(np.array([[0, 9000, 0], [9000, 15000, 1]]), w0[:1], w1[:1], 3)
    assert f[0, 0] == 0.6 and f[0, 1] == 0.4 and f[0, 2] == 0.0
    assert f.sum() == 1.0 and f.argmax() == 0

    f = label_windows(np.array([[1000, 5000, 1]]), w0[:1], w1[:1], 3)
    assert f[0, 1] == 4000 / 15000 and f.sum() == f[0, 1] and f.argmax() == 1

    f = label_windows(np.zeros((0, 3), dtype=np.int64), w0[:2], w1[:2], 3)
    assert f.shape == (2, 3) and f.sum() == 0                # empty spans

    # exact tie must resolve to the lowest canonical id (NSR)
    f = label_windows(np.array([[0, 7500, 1], [7500, 15000, 0]]), w0[:1], w1[:1], 3)
    assert f[0, 0] == f[0, 1] == 0.5 and f.argmax() == 0

    # a span reaching past the last window must not leak into a phantom window
    f = label_windows(np.array([[0, SEG_LEN, 0]]), w0, w1, 3)
    assert np.all(f[:, 0] == 1.0)

    # -- fast RR path vs naive oracle, random beats ---------------------------
    for _ in range(n_random):
        n_win = int(rng.integers(1, 5))
        # irregular beats, occasional gaps and near-duplicates -> exercises the
        # range filter and the same-window condition
        step = rng.integers(40, 900, size=int(rng.integers(3, 80)))
        beats = np.cumsum(step) + int(rng.integers(0, W))
        beats = np.unique(beats[beats < n_win * W])
        if len(beats) < 2:
            continue
        _same(rr_window_features(beats, W, n_win, fs),
              rr_features_reference(beats, W, n_win, fs), "random RR")

    # a window whose beats are all outside the range filter
    beats = np.array([0, 10, 20, 30])
    _same(rr_window_features(beats, W, 1, fs),
          rr_features_reference(beats, W, 1, fs), "all-rejected RR")
    assert rr_window_features(beats, W, 1, fs)["n_rr_used"][0] == 0
    # fewer than two beats -> everything undefined, nothing raised
    assert np.isnan(rr_window_features(np.array([5]), W, 1, fs)["mean_rr"][0])
    assert np.isnan(rr_window_features(np.zeros(0, np.int64), W, 1, fs)["mean_rr"][0])

    # -- code masks -----------------------------------------------------------
    code = np.array([1, 28, 13, 5, 8, 22, 1])
    m = beat_masks(code)
    assert m["beat"].sum() == 5 and m["q"].sum() == 1
    assert m["rr"].sum() == 4 and m["v"].sum() == 1 and m["sv"].sum() == 1
    assert not m["beat"][1]                                  # '+' is not a beat

    if verbose:
        print(f"selftest PASSED  (69-window geometry, hand-computed labels, "
              f"{n_random} random fast-vs-reference RR comparisons)")
    return True


selftest()

# %%
# --- Reading beats/<pid>.npz -------------------------------------------------
_SEG_KEY = re.compile(r"^s(\d+)_samp$")


def rhythm_names(names):
    """Reconcile a patient's rhythm order against the canonical one, by NAME.

    Every npz carries its own `rhythms` array, so nothing here depends on
    icentia_scan.RHYTHMS. Returns remap with remap[local_id] == canonical_id.
    """
    names = tuple(str(x) for x in names)
    if sorted(names) != sorted(RHYTHM_ORDER):
        raise ValueError(f"unexpected rhythm set {names}")
    return np.array([RHYTHM_ORDER.index(n) for n in names], dtype=np.int64)


def segments_in(keys):
    """Segment numbers actually present -- discovered, never assumed.

    Only <=10 of the 50 segments were downloaded per patient, so iterating
    range(50) would be wrong. A segment missing its _code or _spans companion is
    skipped rather than half-read.
    """
    segs = []
    for k in keys:
        m = _SEG_KEY.match(k)
        if not m:
            continue
        s = int(m.group(1))
        if f"s{s:02d}_code" in keys and f"s{s:02d}_spans" in keys:
            segs.append(s)
    return sorted(segs)


def load_beats(path):
    """-> (payload, None) or (None, error string).

    np.load on a compressed npz holds an open zip handle and returns lazy
    members, so every array is copied out before the handle closes -- 2,000
    un-closed loads would leak descriptors, and a lazy reference is dead after
    the `with` block.

    download_patient() in icentia_scan.py writes the npz straight to its final
    path (no .part + os.replace, unlike the record files), so a Colab disconnect
    mid-write leaves a truncated file that looks valid. That reads as BadZipFile
    here; it is reported per patient rather than aborting the run.
    """
    try:
        with np.load(path) as z:
            keys = set(z.files)
            remap = rhythm_names(z["rhythms"])
            fs = int(z["fs"])
            if fs != FS_EXPECTED:
                return None, f"fs={fs}, expected {FS_EXPECTED}"
            data = {}
            for s in segments_in(keys):
                data[s] = (
                    np.asarray(z[f"s{s:02d}_samp"], dtype=np.int64),
                    np.asarray(z[f"s{s:02d}_code"], dtype=np.int64),
                    np.asarray(z[f"s{s:02d}_spans"], dtype=np.int64).reshape(-1, 3),
                )
        return (fs, remap, data), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def windows_for_segment(pid, seg, samp, code, spans, remap, fs):
    """-> (columns dict of equal-length arrays, stats dict) for one segment.

    Only STRUCTURAL floors are applied: a kept window must carry some rhythm
    label and yield at least 2 usable RR intervals. Coverage, purity and beat
    count are stored as columns so every judgement threshold stays a downstream
    decision -- see apply_quality_filter.
    """
    W, stride = WINDOW_SEC * fs, STRIDE_SEC * fs
    # counts_per_window and rr_window_features both assign a sample to a window
    # by integer division, which is only correct without overlap.
    if stride != W:
        raise NotImplementedError(
            "overlapping windows need the tiling decomposed into W/stride "
            "shifted non-overlapping passes; see the plan")

    # Tolerate a segment longer than expected rather than silently truncating.
    hi = max(int(samp.max()) if len(samp) else 0,
             int(spans[:, 1].max()) if len(spans) else 0)
    seg_len = max(SEG_LEN, hi + 1)

    w0, w1 = window_bounds(seg_len, W, stride)
    n_win = len(w0)
    stats = {"n_candidate": n_win, "n_unknown": 0, "n_lowrr": 0, "n_kept": 0,
             "n_coverage_alarm": 0}
    if n_win == 0:
        return None, stats

    sp = spans.copy()
    if len(sp):
        sp[:, 2] = remap[sp[:, 2]]
    frac = label_windows(sp, w0, w1, len(RHYTHM_ORDER))

    cov_raw = frac.sum(axis=1)
    # parse_atr_full closes an open span before opening the next, so spans are
    # non-overlapping by construction -- but that is an invariant of the parser,
    # not of the data. Clip and count rather than assume.
    stats["n_coverage_alarm"] = int((cov_raw > 1 + 1e-6).sum())
    cov = np.clip(cov_raw, 0.0, 1.0)
    winner = frac.max(axis=1)
    label_id = frac.argmax(axis=1)          # ties -> lowest id -> NSR
    unknown = cov_raw <= 0
    purity = np.divide(winner, cov, out=np.zeros(n_win), where=cov > 0)

    m = beat_masks(code)
    n_beats = counts_per_window(samp[m["beat"]], W, n_win)
    n_q = counts_per_window(samp[m["q"]], W, n_win)
    n_v = counts_per_window(samp[m["v"]], W, n_win)
    n_sv = counts_per_window(samp[m["sv"]], W, n_win)
    rrf = rr_window_features(samp[m["rr"]], W, n_win, fs)

    keep = (~unknown) & (rrf["n_rr_used"] >= 2)
    stats["n_unknown"] = int(unknown.sum())
    stats["n_lowrr"] = int(((~unknown) & (rrf["n_rr_used"] < 2)).sum())
    stats["n_kept"] = int(keep.sum())
    if not keep.any():
        return None, stats

    n = int(keep.sum())
    cols = {
        "patient_id": np.full(n, pid, dtype=object),
        "segment": np.full(n, seg, dtype=np.uint8),
        "window_index": np.nonzero(keep)[0].astype(np.uint16),
        "start_sample": w0[keep].astype(np.int32),
        "end_sample": w1[keep].astype(np.int32),
        "label_id": label_id[keep].astype(np.int8),
        "frac_covered": cov[keep],
        "frac_winner": winner[keep],
        "purity": purity[keep],
        "n_beats": n_beats[keep],
        "n_q": n_q[keep],
        "n_v": n_v[keep],
        "n_sv": n_sv[keep],
        "q_frac": np.divide(n_q[keep], n_beats[keep], out=np.zeros(n),
                            where=n_beats[keep] > 0),
    }
    for k, name in enumerate(FRAC_COLS):
        cols[name] = frac[keep, k]
    for k in ["n_rr_used", "n_rr_rejected", "n_drr_used"] + FEATURE_COLS:
        cols[k] = rrf[k][keep]
    return cols, stats


def windows_for_patient(pid, path):
    """-> (columns dict over all this patient's segments, stats dict)."""
    payload, err = load_beats(path)
    if payload is None:
        return None, {"error": err}
    fs, remap, data = payload

    per_seg, stats = [], {"pid": pid, "n_seg": len(data), "n_candidate": 0,
                          "n_unknown": 0, "n_lowrr": 0, "n_kept": 0,
                          "n_coverage_alarm": 0, "codes": Counter()}
    for seg in sorted(data):
        samp, code, spans = data[seg]
        stats["codes"].update(Counter(code.tolist()))
        cols, st = windows_for_segment(pid, seg, samp, code, spans, remap, fs)
        for k in ("n_candidate", "n_unknown", "n_lowrr", "n_kept", "n_coverage_alarm"):
            stats[k] += st[k]
        if cols is not None:
            per_seg.append(cols)
    if not per_seg:
        return None, stats
    merged = {k: np.concatenate([c[k] for c in per_seg]) for k in per_seg[0]}
    return merged, stats

# %%
# --- Inventory: what actually landed in Drive --------------------------------
# Run this before the build. If the subset download was interrupted, the patient
# count here is short and the run_download cell in icentia_scan_colab.ipynb is
# re-runnable (it resumes from the tar archives already written).


def inventory(beats_dir=BEATS_DIR, sample=40):
    paths = sorted(glob.glob(os.path.join(beats_dir, "*.npz")))
    if not paths:
        print(f"NO npz FILES under {beats_dir}")
        print("  In Colab, check that Drive is mounted and the subset download "
              "finished. Locally, only the 5-patient smoke-test residue exists.")
        return []
    total = sum(os.path.getsize(p) for p in paths)
    print(f"{len(paths)} patient npz files, {total/1e6:.1f} MB total")

    n_seg, bad = 0, []
    for p in paths[:sample]:
        payload, err = load_beats(p)
        if payload is None:
            bad.append((os.path.basename(p), err))
        else:
            n_seg += len(payload[2])
    checked = min(sample, len(paths))
    if checked:
        print(f"sampled {checked}: {n_seg/max(checked-len(bad),1):.1f} segments "
              f"per patient on average -> ~{len(paths)*n_seg/max(checked-len(bad),1):.0f} "
              f"segments, ~{len(paths)*n_seg/max(checked-len(bad),1)*69:.0f} candidate windows")
    for name, err in bad:
        print(f"  UNREADABLE {name}: {err}")
    return paths


PATHS = inventory()

# %%
# --- Cheap class-balance pre-estimate, straight from the label index ---------
# label_index.json.gz already holds per-segment sample counts per rhythm, so
# dividing by the window length approximates the window yield for the WHOLE
# 11,000-patient scan in seconds, with no npz opened. Two uses: a sanity bound on
# what the build should report, and a measure of how much is left behind by
# having downloaded only <=10 of each patient's 50 segments.


def estimate_from_index(path=INDEX_GZ, window_sec=WINDOW_SEC):
    if not os.path.exists(path):
        print(f"no {path} -- skipping the pre-estimate (it is optional)")
        return None
    with gzip.open(path, "rt") as f:
        idx = json.load(f)
    fs = idx.get("fs", FS_EXPECTED)
    w = window_sec * fs
    # schema: [seg, nsr_n, afib_n, afl_n, nsr_samp, afib_samp, afl_samp]
    tot = np.zeros(3, dtype=np.float64)
    pat = np.zeros(3, dtype=np.int64)
    for _pid, segs in idx["patients"].items():
        per = np.zeros(3, dtype=np.float64)
        for r in segs:
            per += (r[4], r[5], r[6])
        tot += per
        pat += per > 0
    print(f"--- pre-estimate over the full scan ({len(idx['patients'])} patients) ---")
    print(f"{'rhythm':10s} {'est. windows':>14s} {'hours':>9s} {'patients':>9s}")
    for i, name in enumerate(RHYTHM_ORDER):
        print(f"{name:10s} {tot[i]/w:14.0f} {tot[i]/fs/3600:9.1f} {pat[i]:9d}")
    print("Upper bound: this counts all 50 segments per patient, but only the "
          "downloaded ones can actually be turned into windows.")
    return {"windows": (tot / w).tolist(), "patients": pat.tolist()}


ESTIMATE = estimate_from_index()

# %%
# --- Build the table ---------------------------------------------------------
# Columnar throughout: windows_for_segment returns dict[str, ndarray] and a whole
# part becomes one DataFrame. Pushing 1.4M row dicts through pandas would cost
# gigabytes and minutes; this costs ~130 MB and seconds.
#
# Batches are cut from the full sorted patient list, so part indices are stable
# across runs and a part file's existence is an exact resume marker -- the same
# scheme as the tar archives in icentia_scan.py.


def _tqdm(*a, **k):
    try:
        from tqdm.auto import tqdm
        return tqdm(*a, **k)
    except ImportError:
        class _Null:
            def __init__(self, **kw): pass
            def update(self, n=1): pass
            def set_postfix_str(self, s): pass
            def __enter__(self): return self
            def __exit__(self, *e): return False
        return _Null()


def cache_beats_locally(src=BEATS_DIR, dst="/content/beats_cache"):
    """Bulk-copy beats/ off mounted Drive once.

    2,000 individual Drive reads at 50-100 ms of latency each is minutes of pure
    waiting, and a single bulk copy also survives a mid-run Drive hiccup.
    """
    if not (IN_COLAB and LOCAL_CACHE):
        return src
    if not os.path.isdir(dst) or len(os.listdir(dst)) < len(os.listdir(src)):
        t0 = time.time()
        os.makedirs(dst, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"cached beats/ to {dst} in {time.time()-t0:.0f}s")
    return dst


def build_windows(beats_dir=None, part_size=PART_PATIENTS):
    beats_dir = beats_dir or cache_beats_locally()
    os.makedirs(PARTS_DIR, exist_ok=True)

    pids = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(beats_dir, "*.npz")))
    if not pids:
        raise SystemExit(f"no npz files under {beats_dir}")
    batches = [pids[i:i + part_size] for i in range(0, len(pids), part_size)]
    print(f"{len(pids)} patients in {len(batches)} parts of <={part_size}")

    codes = Counter()
    totals = Counter()
    parts = []
    for bi, batch in enumerate(batches):
        part = os.path.join(
            PARTS_DIR, f"win_{WINDOW_SEC}s_{bi:03d}_{batch[0]}_{batch[-1]}.parquet")
        parts.append(part)
        if os.path.exists(part) and not OVERWRITE:
            print(f"part {bi:03d}: already built, skipping")
            continue

        chunks, done, failed = [], [], []
        with _tqdm(total=len(batch), desc=f"part {bi:03d}", unit="pt") as bar:
            for pid in batch:
                cols, st = windows_for_patient(
                    pid, os.path.join(beats_dir, f"{pid}.npz"))
                if "error" in st:
                    failed.append({"pid": pid, "error": st["error"]})
                else:
                    codes.update(st.pop("codes"))
                    for k in ("n_candidate", "n_unknown", "n_lowrr", "n_kept",
                              "n_coverage_alarm"):
                        totals[k] += st[k]
                    done.append(st)
                    if cols is not None:
                        chunks.append(cols)
                bar.update(1)
                bar.set_postfix_str(f"{totals['n_kept']} windows, "
                                    f"{len(failed)} failed")

        if chunks:
            merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
            df = pd.DataFrame({c: merged[c] for c in COLUMNS})
            for c, t in DTYPES.items():
                if t == "float32":
                    df[c] = np.round(df[c].astype("float64"), 5).astype("float32")
                elif c != "patient_id":
                    df[c] = df[c].astype(t)
            tmp = part + ".part"
            df.to_parquet(tmp, compression="zstd", index=False)
            os.replace(tmp, part)
            print(f"part {bi:03d}: {len(df)} windows -> {part} "
                  f"({os.path.getsize(part)/1e6:.1f} MB)")
            del df, merged, chunks

        for path, rows in ((DONE_JSONL, done), (FAILED_JSONL, failed)):
            if rows:
                with open(path, "a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

    print(f"\ncandidates={totals['n_candidate']} unknown={totals['n_unknown']} "
          f"low-RR={totals['n_lowrr']} kept={totals['n_kept']}")
    if totals["n_coverage_alarm"]:
        print(f"INTEGRITY ALARM: {totals['n_coverage_alarm']} windows had "
              f"overlapping rhythm spans (coverage > 1)")
    unknown_codes = {c: n for c, n in codes.items()
                     if c not in BEAT_CODES and c not in NONBEAT_CODES}
    if unknown_codes:
        print(f"UNRECOGNISED ANNOTATION CODES {unknown_codes} -- these were "
              f"treated as non-beats; check the WFDB label table")
    return [p for p in parts if os.path.exists(p)], dict(totals), codes


PARTS, BUILD_TOTALS, CODE_HIST = build_windows()

# %%
# --- Consolidate the parts into the one file you download --------------------


def consolidate(parts=None, out_path=TABLE_OUT):
    parts = parts or sorted(glob.glob(
        os.path.join(PARTS_DIR, f"win_{WINDOW_SEC}s_*.parquet")))
    if not parts:
        raise SystemExit("no part files to consolidate")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    # keep='last' so a later atr-only expansion pass can supersede a row rather
    # than collide with it
    before = len(df)
    df = df.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)
    if before != len(df):
        print(f"dropped {before-len(df)} duplicate keys (later rows won)")
    assert not df.duplicated(subset=KEY).any(), "key is not unique"

    # label is derived here, not stored in the parts: concatenating columns with
    # per-part category sets silently degrades them to object.
    df["label"] = pd.Categorical.from_codes(df["label_id"].to_numpy(),
                                            categories=list(RHYTHM_ORDER))
    df["patient_id"] = df["patient_id"].astype("category")

    df.to_parquet(out_path, compression="zstd", index=False)
    print(f"{len(df)} windows, {df['patient_id'].nunique()} patients "
          f"-> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return df


DF = consolidate(PARTS)

# %%
# --- Step 5: class balance ---------------------------------------------------


def apply_quality_filter(df, min_beats=MIN_BEATS, min_coverage=MIN_COVERAGE,
                         min_purity=MIN_PURITY):
    """The documented downstream filter. -> (kept, waterfall)."""
    stages = [
        ("in table", np.ones(len(df), dtype=bool)),
        (f"n_beats >= {min_beats}", df["n_beats"] >= min_beats),
        (f"frac_covered >= {min_coverage}", df["frac_covered"] >= min_coverage),
        (f"purity >= {min_purity}", df["purity"] >= min_purity),
    ]
    mask = np.ones(len(df), dtype=bool)
    waterfall = []
    for name, cond in stages:
        mask = mask & np.asarray(cond)
        waterfall.append((name, int(mask.sum())))
    return df[mask], waterfall


def class_table(df):
    g = df.groupby("label", observed=False)
    out = pd.DataFrame({
        "windows": g.size(),
        "hours": g.size() * WINDOW_SEC / 3600.0,
        "patients": g["patient_id"].nunique(),
    })
    out["prevalence"] = out["windows"] / max(len(df), 1)
    return out


def report(df, build_totals=None, code_hist=None):
    line = "=" * 68
    print(line + f"\nSTEP 5 -- CLASS BALANCE, {WINDOW_SEC}s WINDOWS\n" + line)

    print("\n1. DROP WATERFALL")
    if not build_totals or not build_totals.get("n_candidate"):
        # Every part was already on disk, so no patient was walked this run and
        # the pre-table drop counts are not available. The table-level rows
        # below are still exact.
        print("   (build resumed from existing parts -- per-window drop counts "
              "for this run are in windows_done_60s.jsonl)")
    else:
        c = build_totals.get("n_candidate", 0)
        for name, n in (("candidate windows", c),
                        ("  - no rhythm label (noise/gap)", -build_totals.get("n_unknown", 0)),
                        ("  - fewer than 2 usable RR", -build_totals.get("n_lowrr", 0)),
                        ("= written to table", build_totals.get("n_kept", 0))):
            print(f"   {name:34s} {n:>10,}" + (f"  ({100*abs(n)/max(c,1):5.1f}%)"
                                               if c else ""))
    kept, waterfall = apply_quality_filter(df)
    for name, n in waterfall:
        print(f"   {name:34s} {n:>10,}  ({100*n/max(len(df),1):5.1f}%)")

    print(f"\n2. WINDOWS PER CLASS  (after the default filter: "
          f"n_beats>={MIN_BEATS}, coverage>={MIN_COVERAGE}, purity>={MIN_PURITY})")
    ct = class_table(kept)
    print(ct.to_string(float_format=lambda v: f"{v:,.3f}"))

    print("\n3. PATIENT CONCENTRATION")
    for name in RHYTHM_ORDER:
        sub = kept[kept["label"] == name]
        if not len(sub):
            print(f"   {name:10s} no windows")
            continue
        per = sub.groupby("patient_id", observed=True).size().sort_values(ascending=False)
        top = per.head(10).sum() / len(sub)
        print(f"   {name:10s} {len(sub):>9,} windows over {len(per):>5} patients; "
              f"top-10 patients hold {100*top:4.1f}%")

    print("\n4. WINDOWS PER PATIENT (deciles)")
    for name in RHYTHM_ORDER:
        sub = kept[kept["label"] == name]
        if not len(sub):
            continue
        per = sub.groupby("patient_id", observed=True).size()
        q = np.percentile(per, [10, 50, 90])
        print(f"   {name:10s} median {q[1]:6.0f}   p10 {q[0]:6.0f}   p90 {q[2]:6.0f}")

    print("\n5. SENSITIVITY TO THE QUALITY THRESHOLDS")
    settings = [("lenient", 2, 0.0, 0.0), ("default", MIN_BEATS, MIN_COVERAGE, MIN_PURITY),
                ("strict", 40, 0.9, 0.95)]
    rows = {}
    for tag, mb, mc, mp in settings:
        k, _ = apply_quality_filter(df, mb, mc, mp)
        rows[f"{tag} ({mb}/{mc}/{mp})"] = class_table(k)["windows"]
    print(pd.DataFrame(rows).to_string())

    print("\n6. EFFECTIVE SAMPLE SIZE")
    for name in RHYTHM_ORDER[1:]:
        sub = kept[kept["label"] == name]
        n_pat = sub["patient_id"].nunique()
        print(f"   {name}: {len(sub):,} windows from {n_pat} patients -> a grouped "
              f"5-fold puts ~{n_pat//5} {name} patients in each held-out fold")

    if code_hist:
        print("\n7. ANNOTATION CODES SEEN")
        for c, n in sorted(code_hist.items()):
            kind = ("beat" if c in BEAT_CODES else
                    "non-beat" if c in NONBEAT_CODES else "UNRECOGNISED")
            print(f"   code {c:3d}  {n:>12,}  {kind}")

    print("\n8. VERDICT")
    ct_w = class_table(kept)["windows"]
    ct_p = class_table(kept)["patients"]
    afl_w, afl_p = int(ct_w.get("AFlutter", 0)), int(ct_p.get("AFlutter", 0))
    afib_w, afib_p = int(ct_w.get("AFib", 0)), int(ct_p.get("AFib", 0))
    if afl_w < 500 or afl_p < 15:
        print(f"   AFlutter is thin ({afl_w} windows, {afl_p} patients): collapse "
              f"AFib+AFlutter into one 'AF' class. label_id is kept, so that is a "
              f"one-line map, not a re-run.")
    else:
        print(f"   AFlutter is viable ({afl_w} windows, {afl_p} patients): keep 3 classes.")
    if afib_w < 20000 or afib_p < 100:
        print(f"   AFib is thin ({afib_w} windows, {afib_p} patients): run the "
              f"atr-only expansion (fetch .atr for more segments of the AFib "
              f"patients -- 43 KB each, no waveform) before training.")
    else:
        print(f"   AFib is workable ({afib_w} windows, {afib_p} patients).")
    print("   Prevalence here is MANUFACTURED -- the subset was selected "
          "AFib-burden-first, so it is not population prevalence. Evaluate with "
          "precision/recall, not accuracy.")
    print(line)

    with open(SUMMARY_OUT, "w") as f:
        json.dump({
            "window_sec": WINDOW_SEC, "stride_sec": STRIDE_SEC,
            "rr_range_s": [RR_MIN_S, RR_MAX_S],
            "exclude_q_from_rr": EXCLUDE_Q_FROM_RR,
            "thresholds": {"min_beats": MIN_BEATS, "min_coverage": MIN_COVERAGE,
                           "min_purity": MIN_PURITY},
            "rhythm_order": list(RHYTHM_ORDER),
            "build_totals": build_totals or {},
            "code_histogram": {str(k): int(v) for k, v in (code_hist or {}).items()},
            "class_table": class_table(kept).to_dict(),
            "n_rows_table": int(len(df)), "n_rows_filtered": int(len(kept)),
        }, f, indent=1, default=float)
    print(f"wrote {SUMMARY_OUT}")
    return kept


KEPT = report(DF, BUILD_TOTALS, CODE_HIST)

# %%
# --- Independent cross-checks ------------------------------------------------


def cross_check_index(beats_dir=None, index_path=INDEX_GZ, limit=None):
    """Recompute per-rhythm span totals from the npz and compare to the index.

    Exact equality is expected: label_index.json.gz was produced by scan_atr()
    while the npz spans came from parse_atr_full() -- separate code paths over
    the same bytes, so this is a genuine second opinion.

    Compares SEGMENT totals, never window sums: windows discard the 13,577-sample
    tail and every uncovered region, so a window-based comparison would disagree
    for entirely correct reasons.
    """
    if not os.path.exists(index_path):
        print(f"no {index_path} -- skipping cross-check")
        return None
    beats_dir = beats_dir or BEATS_DIR
    with gzip.open(index_path, "rt") as f:
        idx = json.load(f)["patients"]

    checked = mismatched = 0
    for path in sorted(glob.glob(os.path.join(beats_dir, "*.npz")))[:limit]:
        pid = os.path.splitext(os.path.basename(path))[0]
        if pid not in idx:
            continue
        payload, err = load_beats(path)
        if payload is None:
            continue
        _fs, remap, data = payload
        ref = {r[0]: (r[4], r[5], r[6]) for r in idx[pid]}
        for seg, (_samp, _code, spans) in data.items():
            if seg not in ref:
                continue
            mine = [0, 0, 0]
            for a, b, rid in spans:
                mine[int(remap[rid])] += int(b) - int(a)
            if tuple(mine) != tuple(int(x) for x in ref[seg]):
                mismatched += 1
                if mismatched <= 5:
                    print(f"  MISMATCH {pid}_s{seg:02d}: npz={mine} index={ref[seg]}")
            checked += 1
    if mismatched:
        print(f"CROSS-CHECK FAILED on {mismatched}/{checked} segments")
    else:
        print(f"cross-check PASSED: span totals match label_index.json.gz "
              f"exactly on {checked} segments")
    return checked, mismatched


CROSS = cross_check_index()

# %%
# --- Grouped-split preview ---------------------------------------------------
# Windows from one patient share an electrode, a body and an annotator, so they
# are nowhere near independent. Splitting on windows leaks; splits must be
# grouped by patient_id. main.ipynb's plain stratified split over beats has
# exactly this leak (beats of one record land on both sides) -- do not repeat it.


def grouped_split_preview(df, n_splits=5, seed=42):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:
        print("sklearn unavailable -- skipping")
        return None
    if df["patient_id"].nunique() < n_splits:
        print(f"only {df['patient_id'].nunique()} patients -- too few for "
              f"{n_splits} folds, skipping")
        return None

    y = df["label_id"].to_numpy()
    groups = df["patient_id"].astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for i, (tr, te) in enumerate(cv.split(df, y, groups)):
        assert not (set(groups[tr]) & set(groups[te])), "patient leaked across folds"
        counts = {RHYTHM_ORDER[k]: int((y[te] == k).sum()) for k in range(len(RHYTHM_ORDER))}
        rows.append({"fold": i, "train_windows": len(tr), "test_windows": len(te),
                     "test_patients": len(set(groups[te])), **counts})
    print(pd.DataFrame(rows).to_string(index=False))
    print("no patient appears in two folds")
    return rows


grouped_split_preview(KEPT)

# %% [markdown]
# ## What you end up with
#
# | Path (under `MyDrive/icentia_scan/`) | Contents |
# |---|---|
# | `windows_60s.parquet` | **the one file to download** -- every window, unfiltered |
# | `windows_summary_60s.json` | knobs, code histogram, class table |
# | `windows/win_60s_NNN_*.parquet` | part files; each one's existence is a resume point |
# | `windows_done_60s.jsonl` | per-patient drop counts |
# | `windows_failed_60s.jsonl` | patients whose npz was unreadable |
#
# The parquet is ~30 MB for the full subset, so it downloads in minutes even on a
# slow link. Read it back locally with no extra dependency (`pyarrow` is already
# installed):
#
# ```python
# import pandas as pd
# df = pd.read_parquet("windows_60s.parquet")
# df.groupby("label", observed=True).size()
# ```
#
# Because the table is unfiltered, re-tuning costs a query rather than a re-run:
#
# ```python
# train = df.query("n_beats >= 30 and frac_covered >= 0.5 and purity >= 0.7")
# ```
#
# ### Mapping onto the existing 5 features
#
# `mean_rr`, `hr` and `hrv_sdnn` are this project's `rr_interval`, `heart_rate`
# and `hrv` -- but measured over a minute rather than one beat, so they are *not*
# interchangeable with the beat-level model's inputs. `amplitude` and `qrs_width`
# need the raw waveform and arrive in step 6, joined on
# `(patient_id, segment, window_index)`.
#
# ### Before this trains anything
#
# `hrv` in `ecg_model.pkl` is the standard deviation of the last 5 RR intervals
# inside a 5-second buffer. `hrv_sdnn` here is the SD of ~95 intervals over 60
# seconds. Those are different quantities, so a rhythm model trained on this
# table must be served a **60-second** beat history -- and `live_predict.py`
# currently keeps 5 seconds and recycles its buffer down to 2. The sampling-rate
# difference (250 vs 300 Hz) is harmless because RR is in seconds; the window
# length is not.
