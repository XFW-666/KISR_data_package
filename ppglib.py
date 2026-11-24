"""Shared signal-processing utilities for the wrist-PPG motion-artifact study.

Every function operates on real recorded signals. Nothing here synthesises data.
"""
from __future__ import annotations

import os
import numpy as np
import wfdb
from scipy import signal as sps

PPG = "wrist_ppg"
ECG = "chest_ecg"
GYRO = ["wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z"]
ACC_W = ["wrist_wide_range_accelerometer_x", "wrist_wide_range_accelerometer_y",
         "wrist_wide_range_accelerometer_z"]
ACC_L = ["wrist_low_noise_accelerometer_x", "wrist_low_noise_accelerometer_y",
         "wrist_low_noise_accelerometer_z"]
MAG = ["wrist_mag_x", "wrist_mag_y", "wrist_mag_z"]
TIMES = "sample_times_for_all_signals_apart_from_ecg"


def load_wrist(path):
    """Load one PhysioNet 'wrist' record into a plain dict of numpy arrays."""
    rec = wfdb.rdrecord(str(path))
    fs = float(rec.fs)
    idx = {n: i for i, n in enumerate(rec.sig_name)}
    sig = rec.p_signal

    def col(n):
        return np.asarray(sig[:, idx[n]], dtype=float)

    out = {
        "fs": fs,
        "record": os.path.basename(str(path).replace("\\", "/")),
        "ecg": col(ECG),
        "ppg": col(PPG),
        "gyro": np.stack([col(n) for n in GYRO], axis=1),
        "acc": np.stack([col(n) for n in ACC_W], axis=1),
        "acc_ln": np.stack([col(n) for n in ACC_L], axis=1),
        "mag": np.stack([col(n) for n in MAG], axis=1),
    }
    t = col(TIMES) if TIMES in idx else None
    if t is not None and np.all(np.isfinite(t)):
        out["t_nonec"] = t
        dt = np.diff(t)
        out["times_uniform"] = bool(np.max(np.abs(dt - np.median(dt))) < 1.0 / fs)
    else:
        out["t_nonec"] = np.arange(len(out["ppg"])) / fs
        out["times_uniform"] = True

    for k in ("ecg", "ppg"):
        v = out[k]
        bad = ~np.isfinite(v)
        if bad.any():
            good = np.flatnonzero(~bad)
            v[bad] = np.interp(np.flatnonzero(bad), good, v[good])
    for k in ("gyro", "acc", "acc_ln", "mag"):
        v = out[k]
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = 0.0
    return out


def bandpass(x, fs, lo=0.4, hi=8.0, order=3):
    x = np.asarray(x, dtype=float)
    ny = fs / 2.0
    hi = min(hi, 0.95 * ny)
    b, a = sps.butter(order, [lo / ny, hi / ny], btype="band")
    pad = min(3 * max(len(a), len(b)), max(len(x) - 1, 0))
    if len(x) <= pad + 1:
        return x.copy()
    return sps.filtfilt(b, a, x, padlen=pad)


def lowpass(x, fs, fc=4.0, order=3):
    x = np.asarray(x, dtype=float)
    ny = fs / 2.0
    b, a = sps.butter(order, min(fc, 0.95 * ny) / ny, btype="low")
    pad = min(3 * max(len(a), len(b)), max(len(x) - 1, 0))
    if len(x) <= pad + 1:
        return x.copy()
    return sps.filtfilt(b, a, x, padlen=pad)


def welch(x, fs, nperseg_s=8.0, detrend="linear"):
    x = np.asarray(x, dtype=float)
    n = int(min(len(x), max(64, nperseg_s * fs)))
    f, P = sps.welch(x, fs=fs, nperseg=n, noverlap=n // 2, detrend=detrend)
    return f, P


def ppg_sqi(x, fs, seg_s=1.2):
    """Template-correlation signal quality index for a PPG window.

    Returns the mean Pearson correlation of each beat-length segment against the
    window template, in [-1, 1]. This is a standard, reproducible SQI definition.
    """
    x = np.asarray(x, dtype=float)
    if x.size < fs * 2:
        return 0.0
    n = int(seg_s * fs)
    if n < 8 or x.size < 2 * n:
        return 0.0
    segs = np.lib.stride_tricks.sliding_window_view(x, n)[:: max(n // 2, 1)]
    if segs.shape[0] < 2:
        return 0.0
    tmpl = np.median(segs, axis=0)
    tmpl = tmpl - tmpl.mean()
    nt = np.linalg.norm(tmpl)
    if nt < 1e-12:
        return 0.0
    cors = []
    for s in segs:
        s = s - s.mean()
        ns = np.linalg.norm(s)
        cors.append(float(s @ tmpl / (ns * nt)) if ns > 1e-12 else 0.0)
    return float(np.mean(cors))


def ecg_rpeaks(ecg, fs):
    """Detect R peaks on the chest ECG reference channel."""
    ecg = np.asarray(ecg, dtype=float)
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(ecg, sampling_rate=int(round(fs)), method="neurokit")
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        x = bandpass(ecg, fs, 5.0, 20.0, order=3)
        thr = 0.4 * np.percentile(np.abs(x), 99)
        pk, _ = sps.find_peaks(x, height=thr, distance=int(0.25 * fs))
        return pk.astype(int)


def ref_hr(rpeaks, fs, t0, t1):
    """Reference heart rate (bpm) from R peaks inside [t0, t1] seconds."""
    if rpeaks is None or len(rpeaks) < 2:
        return np.nan
    tt = np.asarray(rpeaks, dtype=float) / fs
    sel = tt[(tt >= t0) & (tt <= t1)]
    if sel.size < 2:
        return np.nan
    rr = np.diff(sel)
    rr = rr[(rr > 0.3) & (rr < 2.5)]
    if rr.size == 0:
        return np.nan
    return 60.0 / float(np.median(rr))


def hr_spectral(x, fs, fmin=0.6, fmax=3.4, hw=0.4, nperseg_s=8.0):
    """Heart rate from the PPG spectrum, with a harmonic-consistency bonus.

    A candidate fundamental f0 is scored as P(f0) + hw * P(2 f0), which biases the
    choice toward the true cardiac fundamental rather than a motion harmonic.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 4 * fs:
        return np.nan
    f, P = welch(x, fs, nperseg_s=nperseg_s)
    band = (f >= fmin) & (f <= fmax)
    if not band.any():
        return np.nan
    fb, Pb = f[band], P[band]
    if np.all(Pb <= 0):
        return np.nan
    second = np.interp(np.minimum(2.0 * fb, f[-1]), f, P, left=0.0, right=0.0)
    score = Pb + hw * second
    return float(fb[int(np.argmax(score))] * 60.0)


def hr_harmonic(x, fs, fmin=0.6, fmax=3.4, prev=None, cont_w=0.35, nperseg_s=8.0,
                n_harm=3, lowband=1.1):
    """Heart rate by harmonic-comb scoring with a low-band evidence gate.

    For each candidate f0 the score sums spectral power at f0, 2f0... (normalised,
    each harmonic weighted 1/h, requiring harmonics to *decay*: a pure motion
    sinusoid whose "harmonics" are independent motion peaks gains little). Two
    further guards against the dominant failure mode - low-frequency motion
    harmonics masquerading as the pulse:
      * candidates below ``lowband`` Hz need harmonic evidence of at least 40%
        of their fundamental power (a true sub-1.1-Hz pulse has clear 2f content);
      * a Gaussian continuity prior around the previous estimate when available.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 4 * fs:
        return np.nan, np.nan
    f, P = welch(x, fs, nperseg_s=nperseg_s)
    f = f[f > 0]
    P = P[: len(f)]
    if f.size < 8 or P.max() <= 0:
        return np.nan, np.nan
    Pn = P / P.max()

    cand = f[(f >= fmin) & (f <= fmax)]
    if cand.size == 0:
        return np.nan, np.nan

    def comb(f0):
        s = 0.0
        ev = []
        for h in range(1, n_harm + 1):
            fh = f0 * h
            if fh >= f[-1]:
                break
            v = np.interp(fh, f, Pn) / h
            if h > 1:
                ev.append(v)
            s += v
        return s, ev

    best, best_f = -np.inf, np.nan
    for f0 in cand:
        s, ev = comb(f0)
        if f0 < lowband and ev:
            # require the second harmonic to carry real power
            if max(ev) < 0.4 * np.interp(f0, f, Pn):
                s *= 0.25
        if prev is not None and np.isfinite(prev) and prev > 0:
            pf = prev / 60.0
            s *= np.exp(-0.5 * ((f0 - pf) / (cont_w * max(pf, 0.5))) ** 2)
        if s > best:
            best, best_f = s, f0
    if not np.isfinite(best_f):
        return np.nan, np.nan
    # parabolic refinement on the local peak
    i = int(np.argmin(np.abs(f - best_f)))
    if 0 < i < len(f) - 1:
        y0, y1, y2 = Pn[i - 1], Pn[i], Pn[i + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) < 1:
                best_f = f[i] + delta * (f[1] - f[0])
    return float(best_f * 60.0), float(best)


def hr_peaks(x, fs, fmin=0.6, fmax=3.4):
    """Heart rate from inter-pulse intervals of a band-limited waveform."""
    x = np.asarray(x, dtype=float)
    if x.size < 4 * fs:
        return np.nan
    y = bandpass(x, fs, fmin, fmax, order=3)
    thr = 0.3 * np.percentile(np.abs(y), 98)
    pk, _ = sps.find_peaks(y, height=thr, distance=int(0.3 * fs))
    if pk.size < 3:
        return np.nan
    rr = np.diff(pk) / fs
    rr = rr[(rr > 1.0 / fmax) & (rr < 1.0 / fmin)]
    if rr.size < 2:
        return np.nan
    return 60.0 / float(np.median(rr))


def accel_magnitude(acc):
    acc = np.asarray(acc, dtype=float)
    return np.linalg.norm(acc, axis=1)
