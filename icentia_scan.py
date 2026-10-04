# %% [markdown]
# # Icentia11k full annotation scan (Colab)
#
# Scans the rhythm annotations (`.atr`) of **all 11,000 patients x 50 segments**
# and writes a small label index you can download and use locally.
#
# Why this runs here and not on your laptop:
#
# * The scan reads ~550,000 `.atr` files, ~43 KB each = **~24 GB of transfer**.
# * On a home connection that measured ~0.15 MB/s that is ~40 hours. On Colab it
#   is minutes, because Colab sits next to the data.
# * The *output* is tiny (~10 MB), so only the output comes back to you.
#
# It reads from the **public S3 mirror** (`physionet-open`), not `physionet.org`.
# The mirror is anonymous, has no web-server rate limiting, and serves the same
# bytes. Progress is committed per patient to Google Drive, so a Colab
# disconnect costs you nothing -- just re-run and it resumes.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wfdb", "tqdm"], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
    WORK_DIR = "/content/drive/MyDrive/icentia_scan"
else:
    WORK_DIR = "./icentia_scan"

import os, io, json, time, random, gzip, tarfile, threading, shutil, sys
import concurrent.futures as cf
import numpy as np
import requests
from requests.adapters import HTTPAdapter

os.makedirs(WORK_DIR, exist_ok=True)

S3 = "https://physionet-open.s3.amazonaws.com/icentia11k-continuous-ecg/1.0"

PROGRESS_JSONL = os.path.join(WORK_DIR, "scan_records.jsonl")
FAILED_JSONL = os.path.join(WORK_DIR, "scan_failed.jsonl")
INDEX_OUT = os.path.join(WORK_DIR, "label_index.json.gz")
SELECTED_TXT = os.path.join(WORK_DIR, "selected_patients.txt")
SUBSET_DIR = os.path.join(WORK_DIR, "subset")       # tar archives land here
BEATS_DIR = os.path.join(WORK_DIR, "beats")         # per-patient .npz
STAGING = "/content/subset_staging" if IN_COLAB else "./subset_staging"

N_WORKERS = 64 if IN_COLAB else 12   # patients scanned in parallel
MAX_RETRIES = 6                      # per-segment, exponential backoff
FLUSH_EVERY = 50                     # patients per Drive write

# --- subset selection knobs --------------------------------------------------
TARGET_PATIENTS = 2000
MAX_NSR_PER_PATIENT = 5
# The original download_selected() took *every* abnormal segment, and a
# heavy-AFib patient can have all 50 -- unbounded blow-up. Cap it.
MAX_SEGMENTS_PER_PATIENT_DOWNLOAD = 10
PATIENTS_PER_TAR = 100               # one Drive write per batch, not per file

FS = 250  # Hz, icentia11k sampling rate

print(f"colab={IN_COLAB}  work_dir={WORK_DIR}  workers={N_WORKERS}")

# %%
# --- HTTP with retry ---------------------------------------------------------
# One Session per thread: TLS handshakes and connections are reused across the
# 50 sequential segment fetches a worker does for its patient. That reuse is
# most of the speed.
_tls = threading.local()


def session():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "icentia-scan/1.0"
        s.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=4))
        _tls.s = s
    return s


class Missing(Exception):
    """Object genuinely absent (404) -- not an error worth retrying."""


def fetch(path, retries=MAX_RETRIES):
    """GET {S3}/{path}, returning bytes.

    Retries transient failures with exponential backoff + jitter. A 404 raises
    Missing immediately. Only gives up after `retries` attempts -- a single
    timeout never aborts anything, which is what killed the old scanner.
    """
    url = f"{S3}/{path}"
    last = None
    for attempt in range(retries):
        try:
            r = session().get(url, timeout=(15, 90))
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                raise Missing(path)
            last = f"HTTP {r.status_code}"
        except Missing:
            raise
        except Exception as e:  # timeouts, connection resets, TLS errors
            last = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(min(30.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"{path} failed after {retries} attempts: {last}")

# %%
# --- Fast .atr parser --------------------------------------------------------
# wfdb.rdann is correct but pure-Python and slow; at ~6,900 annotations per
# segment it would cost more CPU-hours than the download costs network. This
# walks the WFDB annotation byte format directly and pulls out only what we
# need. It is validated against wfdb.rdann in the next cell.
#
# Format: stream of little-endian 16-bit words. code = word >> 10, and for
# ordinary annotations the low 10 bits are the sample interval since the
# previous annotation. Special codes: 59 SKIP, 60 NUM, 61 SUB, 62 CHN, 63 AUX
# (length in the low byte, payload follows, padded to an even byte count).
#
# SKIP is the subtle one, and getting it wrong cost a debugging round here.
# Mirroring wfdb's proc_core_fields: a SKIP word is followed by 4 bytes holding
# a signed 32-bit interval, and then by an ordinary word whose label is the
# annotation's label AND whose own 10-bit interval is *added* to the skip
# interval. SKIPs can also repeat, so it is a loop, not an if. Discarding that
# trailing 10-bit interval shifts every subsequent sample index by a constant —
# invisible in durations (it cancels), obvious in absolute beat times.

RHYTHMS = ("NSR", "AFib", "AFlutter")


def classify(note):
    if note.startswith(b"(AFIB"):
        return "AFib"
    if note.startswith(b"(AFL"):
        return "AFlutter"
    if note.startswith(b"(N"):
        return "NSR"
    return None


def walk_atr(buf):
    """Yield (sample, label_store, aux_notes) per annotation.

    Mirrors wfdb.io.annotation.proc_core_fields + proc_extra_field. Both
    parsers below are built on this so they cannot drift apart.
    """
    n = len(buf)
    i = 0
    t = 0

    while i + 1 < n:
        dt = 0
        # Consecutive SKIPs accumulate; each is 3 word-pairs wide.
        while i + 1 < n and (buf[i + 1] >> 2) == 59:
            if i + 6 > n:
                return
            sd = ((buf[i + 2] << 16) + (buf[i + 3] << 24)
                  + buf[i + 4] + (buf[i + 5] << 8))
            if sd > 2147483647:
                sd -= 4294967296        # signed 32-bit, two's complement
            dt += sd
            i += 6
        if i + 1 >= n:
            return

        lo, hi = buf[i], buf[i + 1]
        code = hi >> 2
        dt += lo + 256 * (hi & 3)       # this word's own interval, skip or not
        i += 2
        t += dt

        notes = []
        while i + 1 < n:                # codes > 59 are extra fields
            c2 = buf[i + 1] >> 2
            if c2 == 63:                # AUX
                length = buf[i]
                notes.append(buf[i + 2: i + 2 + length].rstrip(b"\x00").strip())
                i += 2 + 2 * ((length + 1) // 2)
            elif c2 in (60, 61, 62):    # NUM / SUB / CHN
                i += 2
            else:
                break

        yield t, code, notes


def scan_atr(buf):
    """Return (onsets, samples) dicts keyed by rhythm name.

    onsets  -- how many times each rhythm starts
    samples -- how many samples are spent in each rhythm (onset to the
               matching ')' , or to the last annotation if unterminated)
    """
    t = 0
    cur = None       # rhythm currently open
    cur_start = 0
    onsets = {k: 0 for k in RHYTHMS}
    samples = {k: 0 for k in RHYTHMS}

    for t, _code, notes in walk_atr(buf):
        for note in notes:
            if note == b")":
                if cur is not None:
                    samples[cur] += t - cur_start
                    cur = None
            else:
                r = classify(note)
                if r is not None:
                    if cur is not None:
                        samples[cur] += t - cur_start
                    onsets[r] += 1
                    cur, cur_start = r, t

    if cur is not None:
        samples[cur] += t - cur_start
    return onsets, samples


def parse_atr_full(buf):
    """Same walk, but keep everything: beat times, beat codes, rhythm spans.

    Returns (samp, code, spans):
      samp  -- int32 sample index of each annotation
      code  -- uint8 WFDB label_store of each annotation. Map to symbols with
               wfdb.io.annotation.ann_label_table (1=N, 5=V, 8=A/S, ...); stored
               as the raw store value so no mapping is baked in here.
      spans -- int64 (n, 3) array of [start_sample, end_sample, rhythm_id]
               with rhythm_id indexing RHYTHMS.

    samp/code match wfdb.rdann's ann.sample / ann.label_store exactly, including
    rdann's removal of non-annotations (label_store 0, which covers the trailing
    EOF word) and of the sample-0 NOTE carrying '## time resolution: ...'.
    Spans are built from every annotation, before that removal.
    """
    t = 0
    cur = None
    cur_start = 0
    samp, code, spans = [], [], []

    for t, c, notes in walk_atr(buf):
        if c != 0 and not (c == 22 and t == 0):
            samp.append(t)
            code.append(c)

        for note in notes:
            if note == b")":
                if cur is not None:
                    spans.append([cur_start, t, RHYTHMS.index(cur)])
                    cur = None
            else:
                r = classify(note)
                if r is not None:
                    if cur is not None:
                        spans.append([cur_start, t, RHYTHMS.index(cur)])
                    cur, cur_start = r, t

    if cur is not None:
        spans.append([cur_start, t, RHYTHMS.index(cur)])

    return (np.asarray(samp, dtype=np.int32),
            np.asarray(code, dtype=np.uint8),
            np.asarray(spans, dtype=np.int64).reshape(-1, 3))

# %%
# --- Validate the fast parser against wfdb.rdann -----------------------------
# Do not skip this. If the byte walker drifts from wfdb's own reader, every
# number downstream is quietly wrong.
import tempfile
import wfdb


def validate(n_samples=25, seed=0):
    rng = random.Random(seed)
    tmp = tempfile.mkdtemp()
    mismatches = []
    checked = with_af = 0
    for _ in range(n_samples):
        pid = f"p{rng.randrange(11000):05d}"
        seg = rng.randrange(50)
        rec = f"{pid}_s{seg:02d}"
        try:
            buf = fetch(f"{pid[:3]}/{pid}/{rec}.atr")
        except Missing:
            continue

        mine_onsets, mine_samples = scan_atr(buf)

        p = os.path.join(tmp, f"{rec}.atr")
        with open(p, "wb") as f:
            f.write(buf)
        # label_store is None unless asked for; rdann returns symbols by default.
        ann = wfdb.rdann(p[: -len(".atr")], "atr",
                         return_label_elements=["symbol", "label_store"])
        os.remove(p)

        # Rebuild both onsets and durations from wfdb's own sample numbers, so
        # the durations are checked against an independent source too.
        ref_onsets = {k: 0 for k in RHYTHMS}
        ref_samples = {k: 0 for k in RHYTHMS}
        cur, cur_start = None, 0
        for samp, note in zip(ann.sample, ann.aux_note):
            note = note.strip().encode() if isinstance(note, str) else (note or b"")
            note = note.rstrip(b"\x00").strip()
            if note == b")":
                if cur is not None:
                    ref_samples[cur] += int(samp) - cur_start
                    cur = None
                continue
            r = classify(note)
            if r is not None:
                if cur is not None:
                    ref_samples[cur] += int(samp) - cur_start
                ref_onsets[r] += 1
                cur, cur_start = r, int(samp)
        if cur is not None:
            ref_samples[cur] += int(ann.sample[-1]) - cur_start

        if mine_onsets != ref_onsets or mine_samples != ref_samples:
            mismatches.append((rec, (mine_onsets, mine_samples),
                               (ref_onsets, ref_samples)))

        # parse_atr_full must agree with wfdb on every beat time and code, and
        # its spans must reproduce the same onsets/durations as scan_atr.
        f_samp, f_code, f_spans = parse_atr_full(buf)
        problems = []
        if not np.array_equal(f_samp, np.asarray(ann.sample, dtype=np.int32)):
            problems.append("sample")
        if not np.array_equal(f_code, np.asarray(ann.label_store, dtype=np.uint8)):
            problems.append("label_store")
        span_onsets = {k: 0 for k in RHYTHMS}
        span_samples = {k: 0 for k in RHYTHMS}
        for a, b, rid in f_spans:
            span_onsets[RHYTHMS[rid]] += 1
            span_samples[RHYTHMS[rid]] += int(b - a)
        if span_onsets != ref_onsets or span_samples != ref_samples:
            problems.append("spans")
        if problems:
            mismatches.append((rec + " [full:" + ",".join(problems) + "]",
                               (span_onsets, span_samples),
                               (ref_onsets, ref_samples)))

        checked += 1
        if any(ref_onsets[r] for r in ("AFib", "AFlutter")):
            with_af += 1

    if mismatches:
        for rec, mine, ref in mismatches:
            print(f"  MISMATCH {rec}:\n    fast={mine}\n    wfdb={ref}")
        raise SystemExit(f"parser validation FAILED on {len(mismatches)}/{checked}")
    print(f"parser validation PASSED on {checked} random segments "
          f"(onsets + durations); {with_af} of them contained AFib/AFlutter")


validate()

# %%
# --- Master patient list -----------------------------------------------------
records_path = os.path.join(WORK_DIR, "RECORDS")
if not os.path.exists(records_path):
    with open(records_path, "wb") as f:
        f.write(fetch("RECORDS"))

with open(records_path) as f:
    PATIENTS = [ln.strip().rstrip("/").split("/")[-1] for ln in f if ln.strip()]

print(f"{len(PATIENTS)} patients in master list; first={PATIENTS[0]} last={PATIENTS[-1]}")

# %%
# --- The scan ----------------------------------------------------------------
# Unit of work is a whole patient: one worker fetches that patient's RECORDS,
# then each listed .atr sequentially on a warm connection. A patient is
# committed as one JSONL line only when all its segments succeeded, so resume
# is exact and a partial patient is never half-recorded.


def scan_patient(pid):
    recs = fetch(f"{pid[:3]}/{pid}/RECORDS").decode().split()
    segs = []
    nbytes = 0
    for rec in recs:
        seg = int(rec.rsplit("_s", 1)[1])
        try:
            buf = fetch(f"{pid[:3]}/{pid}/{rec}.atr")
        except Missing:
            continue
        nbytes += len(buf)
        onsets, samples = scan_atr(buf)
        segs.append([seg,
                     onsets["NSR"], onsets["AFib"], onsets["AFlutter"],
                     samples["NSR"], samples["AFib"], samples["AFlutter"]])
    return {"pid": pid, "segs": segs, "bytes": nbytes}


def load_done(path):
    """Read the append-only log, tolerating a truncated final line."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue  # interrupted mid-write; the patient just gets rescanned
            done[rec["pid"]] = rec
    return done


def run_scan(patients=None, workers=N_WORKERS):
    from tqdm.auto import tqdm

    patients = patients or PATIENTS
    done = load_done(PROGRESS_JSONL)
    todo = [p for p in patients if p not in done]
    print(f"{len(done)} patients already scanned, {len(todo)} to go")
    if not todo:
        return done

    buf_out, failed = [], []
    lock = threading.Lock()
    t0 = time.time()
    total_bytes = 0

    def flush():
        if not buf_out:
            return
        with open(PROGRESS_JSONL, "a") as f:
            for rec in buf_out:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        buf_out.clear()

    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(scan_patient, p): p for p in todo}
        with tqdm(total=len(todo), unit="pt", smoothing=0.05) as bar:
            for fut in cf.as_completed(futs):
                pid = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    # A patient that exhausted its retries. Logged, not fatal;
                    # re-running the cell picks it up again.
                    with lock:
                        failed.append({"pid": pid, "error": f"{type(e).__name__}: {e}"})
                    bar.update(1)
                    continue
                with lock:
                    done[pid] = rec
                    buf_out.append(rec)
                    total_bytes += rec["bytes"]
                    if len(buf_out) >= FLUSH_EVERY:
                        flush()
                    mb = total_bytes / 1e6
                    el = time.time() - t0
                    bar.set_postfix_str(f"{mb/1e3:.2f} GB, {mb/max(el,1e-9):.1f} MB/s, "
                                        f"{len(failed)} failed")
                bar.update(1)

    with lock:
        flush()
        if failed:
            with open(FAILED_JSONL, "a") as f:
                for rec in failed:
                    f.write(json.dumps(rec) + "\n")

    print(f"\nscanned {len(done)}/{len(patients)} patients, "
          f"{total_bytes/1e9:.2f} GB this run, {len(failed)} failed")
    if failed:
        print("Re-run this cell to retry the failures (already-done patients are skipped).")
    return done


DONE = run_scan()

# %%
# --- Write the label index ---------------------------------------------------
# This is the only artifact you need locally. Rows are
# [segment, nsr_onsets, afib_onsets, afl_onsets, nsr_samples, afib_samples, afl_samples]
index = {pid: rec["segs"] for pid, rec in sorted(DONE.items())}

with gzip.open(INDEX_OUT, "wt") as f:
    json.dump({"fs": FS, "schema": ["seg", "nsr_n", "afib_n", "afl_n",
                                    "nsr_samp", "afib_samp", "afl_samp"],
               "patients": index}, f, separators=(",", ":"))

n_seg = sum(len(v) for v in index.values())
n_afib = sum(1 for v in index.values() if any(r[2] or r[3] for r in v))
print(f"{len(index)} patients, {n_seg} segments indexed")
print(f"{n_afib} patients have AFib or AFlutter somewhere "
      f"({100*n_afib/max(len(index),1):.1f}%)")
print(f"wrote {INDEX_OUT}  ({os.path.getsize(INDEX_OUT)/1e6:.1f} MB)")

if IN_COLAB:
    from google.colab import files
    files.download(INDEX_OUT)

# %% [markdown]
# ## Subset selection and signal download
#
# Everything above only touched annotations. The cells below pull the actual ECG
# waveforms (`.dat`) for the selected patients, plus the `.hea` and `.atr` that
# `features.py` needs alongside them.
#
# Two things differ from the old `download_selected`:
#
# * Segments per patient are **capped** (`MAX_SEGMENTS_PER_PATIENT_DOWNLOAD`).
#   The old code took every abnormal segment, and a heavy-AFib patient can have
#   all 50 -- so the download size was unbounded.
# * Records are staged on Colab's local disk and shipped to Drive as **tar
#   archives**, ~100 patients each. Writing ~12,000 small files straight through
#   mounted Drive hits Drive's API operation limits and stalls; one tar per batch
#   is ~100x fewer operations, and each finished archive is a resume point.

# %%
# --- Select patients and segments -------------------------------------------
# Same selection semantics as the original scanner, so the subset is the one it
# would have produced, just bounded.
IDX = {pid: rec["segs"] for pid, rec in DONE.items()}


def totals(segs, i):
    return sum(r[i] for r in segs)


def select_patients(index, target_n=TARGET_PATIENTS):
    with_ab, without_ab = [], []
    for pid, segs in index.items():
        if totals(segs, 2) or totals(segs, 3):
            with_ab.append(pid)
        else:
            without_ab.append(pid)

    with_ab.sort(key=lambda p: totals(index[p], 2) + totals(index[p], 3),
                 reverse=True)
    without_ab.sort()

    selected = with_ab[:target_n]
    if len(selected) < target_n:
        selected += without_ab[:target_n - len(selected)]
    return selected


def choose_segments(segs):
    """Abnormal segments first (most AFib/AFlutter burden), then some NSR."""
    abnormal = [r for r in segs if r[2] or r[3]]
    abnormal.sort(key=lambda r: r[5] + r[6], reverse=True)   # by duration
    abnormal = abnormal[:MAX_SEGMENTS_PER_PATIENT_DOWNLOAD]

    ab_ids = {r[0] for r in abnormal}
    nsr = sorted((r for r in segs if r[0] not in ab_ids), key=lambda r: r[0])

    # Original rule: as many NSR segments as abnormal ones (at least 1), capped
    # at MAX_NSR_PER_PATIENT. Then bounded by whatever the segment cap leaves.
    want = min(max(len(abnormal), 1), MAX_NSR_PER_PATIENT, len(nsr))
    budget = max(MAX_SEGMENTS_PER_PATIENT_DOWNLOAD - len(abnormal), 0)
    n_nsr = min(want, budget)
    return sorted(ab_ids | {r[0] for r in nsr[:n_nsr]})


SELECTED = select_patients(IDX)
PLAN = {pid: choose_segments(IDX[pid]) for pid in SELECTED}
PLAN = {pid: segs for pid, segs in PLAN.items() if segs}

n_records = sum(len(v) for v in PLAN.values())
n_ab = sum(1 for pid in PLAN if totals(IDX[pid], 2) or totals(IDX[pid], 3))
# .dat is a fixed 2,097,154 bytes per segment (1,048,577 int16 samples at 250 Hz,
# about 70 minutes); .atr averages ~43 KB; .hea ~86 B.
projected = n_records * (2_097_154 + 44_000)

os.makedirs(os.path.dirname(SELECTED_TXT) or ".", exist_ok=True)
with open(SELECTED_TXT, "w") as f:
    f.write("\n".join(SELECTED))

print(f"selected {len(PLAN)} patients ({n_ab} with AFib/AFlutter)")
print(f"{n_records} records to download")
print(f"projected download: {projected/1e9:.2f} GB")
print(f"wrote {SELECTED_TXT}")
print("\nSet CONFIRM_DOWNLOAD = True in the next cell to proceed.")

# %%
# --- Download the subset -----------------------------------------------------
CONFIRM_DOWNLOAD = False   # <-- flip to True once the projection above looks right

SUBSET_WORKERS = 32 if IN_COLAB else 8


def download_patient(pid, segs):
    """Fetch .dat/.hea/.atr for one patient into staging; write its beats .npz.

    The .atr is parsed for beat times while it is already in memory, so the
    beats artifact costs no extra requests.
    """
    pdir = os.path.join(STAGING, pid)
    os.makedirs(pdir, exist_ok=True)
    nbytes = 0
    beats = {}

    for seg in segs:
        rec = f"{pid}_s{seg:02d}"
        for ext in ("hea", "dat", "atr"):
            dest = os.path.join(pdir, f"{rec}.{ext}")
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                if ext == "atr":
                    with open(dest, "rb") as fh:
                        buf = fh.read()
                nbytes += os.path.getsize(dest)
                continue
            buf = fetch(f"{pid[:3]}/{pid}/{rec}.{ext}")
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                fh.write(buf)
            os.replace(tmp, dest)     # never leave a half-written record behind
            nbytes += len(buf)

        samp, code, spans = parse_atr_full(buf)
        beats[f"s{seg:02d}_samp"] = samp
        beats[f"s{seg:02d}_code"] = code
        beats[f"s{seg:02d}_spans"] = spans

    os.makedirs(BEATS_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(BEATS_DIR, f"{pid}.npz"),
                        rhythms=np.array(RHYTHMS), fs=FS, **beats)
    return nbytes


def run_download(plan, workers=SUBSET_WORKERS):
    from tqdm.auto import tqdm

    os.makedirs(STAGING, exist_ok=True)
    os.makedirs(SUBSET_DIR, exist_ok=True)

    pids = sorted(plan)
    batches = [pids[i:i + PATIENTS_PER_TAR]
               for i in range(0, len(pids), PATIENTS_PER_TAR)]
    print(f"{len(pids)} patients in {len(batches)} batches of {PATIENTS_PER_TAR}")

    for bi, batch in enumerate(batches):
        tar_path = os.path.join(SUBSET_DIR, f"subset_batch_{bi:03d}.tar")
        if os.path.exists(tar_path):
            print(f"batch {bi:03d}: already in Drive, skipping")
            continue

        failed = []
        total = 0
        lock = threading.Lock()
        with cf.ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(download_patient, p, plan[p]): p for p in batch}
            with tqdm(total=len(batch), desc=f"batch {bi:03d}", unit="pt") as bar:
                for fut in cf.as_completed(futs):
                    pid = futs[fut]
                    try:
                        nb = fut.result()
                    except Exception as e:
                        with lock:
                            failed.append((pid, f"{type(e).__name__}: {e}"))
                        bar.update(1)
                        continue
                    with lock:
                        total += nb
                    bar.set_postfix_str(f"{total/1e9:.2f} GB, {len(failed)} failed")
                    bar.update(1)

        if failed:
            # Do not seal a partial batch into Drive -- the archive's presence is
            # the resume marker, so an incomplete one would be skipped forever.
            print(f"batch {bi:03d}: {len(failed)} patients failed, NOT archiving. "
                  f"Re-run this cell to retry (finished records are reused).")
            for pid, err in failed[:5]:
                print(f"    {pid}: {err}")
            continue

        tmp_tar = tar_path + ".part"
        with tarfile.open(tmp_tar, "w") as tf:
            for pid in batch:
                tf.add(os.path.join(STAGING, pid), arcname=pid)
        os.replace(tmp_tar, tar_path)
        for pid in batch:
            shutil.rmtree(os.path.join(STAGING, pid), ignore_errors=True)
        print(f"batch {bi:03d}: wrote {tar_path} "
              f"({os.path.getsize(tar_path)/1e9:.2f} GB)")

    print("\nsubset download complete")


if CONFIRM_DOWNLOAD:
    run_download(PLAN)
else:
    print(f"CONFIRM_DOWNLOAD is False -- nothing downloaded.\n"
          f"This would fetch ~{projected/1e9:.1f} GB into {SUBSET_DIR}.")

# %% [markdown]
# ## What you end up with, in Drive under `MyDrive/icentia_scan/`
#
# | Path | Contents |
# |---|---|
# | `label_index.json.gz` | all 550,000 segments: AFib/AFlutter/NSR onsets + durations |
# | `selected_patients.txt` | the chosen patient IDs, best-first |
# | `scan_records.jsonl` | resume log, one line per scanned patient |
# | `scan_failed.jsonl` | patients that exhausted retries |
# | `subset/subset_batch_NNN.tar` | `.dat` + `.hea` + `.atr`, ~100 patients each |
# | `beats/<pid>.npz` | beat times, beat codes, rhythm spans (~70x smaller than `.dat`) |
#
# To read a record back after extracting a tar:
#
# ```python
# import wfdb
# rec = wfdb.rdrecord("p00123/p00123_s07")     # 1,048,577 samples @ 250 Hz (~70 min)
# ann = wfdb.rdann("p00123/p00123_s07", "atr")
# ```
#
# And the beats file, if you want RR intervals without touching the waveform:
#
# ```python
# import numpy as np
# z = np.load("beats/p00123.npz")
# samp = z["s07_samp"]                          # beat sample indices
# rr = np.diff(samp) / z["fs"]                  # RR intervals in seconds
# spans = z["s07_spans"]                        # [start, end, rhythm_id]
# rhythms = z["rhythms"]                        # rhythm_id -> name
# ```
