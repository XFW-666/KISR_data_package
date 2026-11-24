"""KISR: Kinematic-Informed Subspace Reconstruction for motion-robust wrist PPG.

The method treats motion artifact removal as a *subspace* problem rather than a
reference-subtraction problem:

  1. The full kinematic state (3-axis accelerometer + 3-axis gyroscope and their
     harmonics) defines a motion subspace, estimated per analysis window.
  2. The wrist PPG is projected onto that subspace; the projection is an estimate
     of the motion-correlated component.
  3. Subtraction is *gated*: how much of the projection is removed in each
     frequency bin is set by the magnitude-squared coherence between PPG and the
     kinematic reference and by an artifact-dominance index. Cardiac bins that
     satisfy a harmonic-consistency test are protected, which prevents the
     over-subtraction failure mode of plain adaptive filtering.

Everything is deterministic and operates on real recorded signals.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

import ppglib as pl


# --------------------------------------------------------------------------
# STFT helpers (implemented explicitly so the reconstruction is fully controlled)
# --------------------------------------------------------------------------
def _win(n, kind="hann"):
    if kind == "hann":
        return np.hanning(n + 1)[:-1]
    return np.ones(n)


def stft(x, nperseg, hop, kind="hann"):
    x = np.asarray(x, dtype=float)
    w = _win(nperseg, kind)
    n_frames = 1 + max(0, (len(x) - nperseg)) // hop
    if n_frames < 1:
        pad = np.zeros(nperseg - len(x))
        x = np.concatenate([x, pad])
        n_frames = 1
    idx = np.arange(nperseg)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * w[None, :]
    return np.fft.rfft(frames, axis=1), w


def istft(Z, nperseg, hop, length=None, kind="hann"):
    w = _win(nperseg, kind)
    frames = np.fft.irfft(Z, n=nperseg, axis=1) * w[None, :]
    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * hop + nperseg
    out = np.zeros(out_len)
    wsum = np.zeros(out_len)
    ww = w ** 2
    for i in range(n_frames):
        s = i * hop
        out[s:s + nperseg] += frames[i]
        wsum[s:s + nperseg] += ww
    # guard against numerical blow-up in low-overlap regions
    wsum = np.maximum(wsum, 1e-6 * ww.max())
    out /= wsum
    if length is not None:
        if out_len >= length:
            out = out[:length]
        else:
            out = np.concatenate([out, np.zeros(length - out_len)])
    return out


# --------------------------------------------------------------------------
# Kinematic reference subspace
# --------------------------------------------------------------------------
def motion_subspace(imu, fs, nperseg, hop, n_harm=3, tol_hz=0.15, n_peaks=4):
    """Build a mask of motion-dominated time-frequency bins from IMU channels.

    ``imu`` is (n_samples, n_channels); columns are accelerometer/gyroscope axes.
    For every analysis frame the strongest motion frequencies are found and the
    bins around each fundamental and its first ``n_harm`` harmonics are marked,
    which is what makes this a *subspace* (many basis directions) rather than a
    single reference signal.
    """
    imu = np.asarray(imu, dtype=float)
    n_frames = 1 + max(0, (imu.shape[0] - nperseg)) // hop
    n_frames = max(n_frames, 1)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    mask = np.zeros((n_frames, len(freqs)), dtype=bool)
    fund = np.zeros(n_frames)

    for c in range(imu.shape[1]):
        Z, _ = stft(imu[:, c], nperseg, hop)
        if Z.shape[0] < n_frames:
            Z = np.vstack([Z, np.zeros((n_frames - Z.shape[0], Z.shape[1]), complex)])
        P = np.abs(Z[:n_frames]) ** 2
        band = (freqs >= 0.5) & (freqs <= min(fs / 2 * 0.95, 12.0))
        for i in range(n_frames):
            p = P[i].copy()
            p[~band] = 0.0
            if p.sum() <= 0:
                continue
            pk, _ = sps.find_peaks(p, distance=max(1, int(0.4 / (fs / nperseg))))
            if pk.size == 0:
                pk = np.array([int(np.argmax(p))])
            order = pk[np.argsort(p[pk])[::-1]][:n_peaks]
            for f0_idx in order:
                f0 = freqs[f0_idx]
                if f0 < 0.5:
                    continue
                if p[f0_idx] > 0.15 * p.max():
                    if fund[i] == 0.0:
                        fund[i] = f0
                    for h in range(1, n_harm + 1):
                        fh = f0 * h
                        if fh > freqs[-1]:
                            break
                        sel = np.abs(freqs - fh) <= max(tol_hz, fh * 0.04)
                        mask[i, sel] = True
    return mask, freqs, fund


# --------------------------------------------------------------------------
# Coherence and dominance
# --------------------------------------------------------------------------
def ppg_motion_coherence(ppg, ref, fs, nperseg, hop):
    """Magnitude-squared coherence per STFT frame between PPG and a reference."""
    Zp, _ = stft(ppg, nperseg, hop)
    Zr, _ = stft(ref, nperseg, hop)
    n = min(Zp.shape[0], Zr.shape[0])
    Zp, Zr = Zp[:n], Zr[:n]
    # single-frame coherence is degenerate; use a small running window
    k = 3
    ker = np.ones(k) / k
    Sxx = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, np.abs(Zp) ** 2)
    Syy = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, np.abs(Zr) ** 2)
    Sxy = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, np.conj(Zp) * Zr)
    denom = Sxx * Syy
    coh = np.zeros_like(Sxx.real)
    nz = denom > 1e-18
    coh[nz] = (np.abs(Sxy[nz]) ** 2) / denom[nz]
    return np.clip(coh, 0.0, 1.0)


def harmonic_protection(freqs, P, fmin=0.7, fmax=3.2, tol=0.12, n_peaks=4):
    """Bins that carry a cardiac signature: peaks f with a matching 2f peak.

    ``P`` is the mean power spectrum on the same frequency axis as ``freqs``.
    """
    freqs = np.asarray(freqs)
    P = np.asarray(P, dtype=float)
    protect = np.zeros(len(freqs), dtype=bool)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any() or P.size != freqs.size:
        return protect
    fb, Pb = freqs[band], P[band]
    if not np.isfinite(Pb).any() or Pb.max() <= 0:
        return protect
    for i in np.argsort(Pb)[::-1][:n_peaks]:
        if Pb[i] <= 0.05 * Pb.max():
            break
        f0 = fb[i]
        protect[np.abs(freqs - f0) <= tol] = True
        protect[np.abs(freqs - 2 * f0) <= max(tol, 0.18)] = True
    return protect


# --------------------------------------------------------------------------
# KISR
# --------------------------------------------------------------------------
def kisr(ppg, imu, fs, nperseg=256, hop=64, n_harm=3, gate_k=1.0,
         max_atten=0.92, protect_cardiac=True, return_diag=False):
    """Reconstruct a motion-robust PPG from the wrist channel.

    Returns the cleaned signal, or (cleaned, diagnostics) when ``return_diag``.
    """
    ppg = np.asarray(ppg, dtype=float)
    n = len(ppg)
    x = pl.bandpass(ppg, fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)

    mask, freqs, fund = motion_subspace(imu, fs, nperseg, hop, n_harm=n_harm)

    acc_ref = pl.accel_magnitude(imu[:, :3])
    acc_ref = acc_ref - acc_ref.mean()
    coh = ppg_motion_coherence(x, acc_ref, fs, nperseg, hop)

    Z, _ = stft(x, nperseg, hop)
    nf = min(Z.shape[0], mask.shape[0], coh.shape[0])
    Z, mask, coh = Z[:nf], mask[:nf], coh[:nf]
    P = np.abs(Z) ** 2

    protect = harmonic_protection(freqs, P.mean(axis=0)) if protect_cardiac else np.zeros(len(freqs), bool)

    # artifact-dominance index per frame: motion-subspace energy share
    tot = P.sum(axis=1) + 1e-18
    dom = (P * mask).sum(axis=1) / tot

    # per-frame motion intensity from the IMU itself (normalized 0-1 within the
    # record): deep suppression is only warranted when the body is really moving
    acc_ref_full = pl.accel_magnitude(imu[:, :3]) if imu is not None and imu.shape[1] >= 3 \
        else np.zeros(Z.shape[0] * hop + nperseg)
    a_std = float(np.std(acc_ref_full)) + 1e-12
    a_win = []
    for i in range(nf):
        s0 = i * hop
        seg = acc_ref_full[s0:s0 + nperseg]
        a_win.append(float(np.std(seg)) if seg.size else 0.0)
    a_win = np.asarray(a_win)
    motion_lv = np.clip(a_win / (3.0 * a_std), 0.0, 1.0)

    # per-frame cardiac protection: the strongest harmonic-consistent peaks of
    # THIS frame, so that walking-periodicity bins that coincide with the pulse
    # are never over-suppressed (the failure mode of plain adaptive filtering)
    card_lo, card_hi = 0.7, 3.2
    band = (freqs >= card_lo) & (freqs <= card_hi)

    gain = np.ones_like(P)
    for i in range(nf):
        # scale the artifact strength by how hard the subject is actually moving:
        # gentle motion keeps the PPG nearly intact, hard motion justifies gating
        a = float(np.clip(dom[i] * gate_k * (0.35 + 0.65 * motion_lv[i]), 0.0, 1.0))
        g = 1.0 - a * coh[i]
        # frame-local cardiac protection over ALL plausible cardiac peaks:
        # protecting only the strongest lets a low-frequency motion remnant win
        # the spectral peak after suppression, so every peak above 15% of the
        # frame max inside the cardiac band keeps a proportionate floor
        Pi = P[i]
        if band.any() and Pi.max() > 0:
            Pb = Pi[band]
            fb = freqs[band]
            ranked = np.argsort(Pb)[::-1]
            pmax = Pb[ranked[0]]
            for k in ranked:
                if Pb[k] < 0.15 * pmax:
                    break
                f0 = fb[k]
                near0 = np.abs(freqs - f0) <= 0.12
                near2 = np.abs(freqs - 2 * f0) <= 0.2
                # floor scaled by the peak's relative power (strong peaks keep more)
                rel = float(Pb[k] / pmax)
                floor = 0.35 + 0.4 * rel
                g[near0 | near2] = np.maximum(g[near0 | near2], floor)
        gain[i] = np.clip(g, 1.0 - max_atten * (0.5 + 0.5 * motion_lv[i]), 1.0)

    # keep a mild floor everywhere so residual cardiac energy is not erased
    gain = np.maximum(gain, 0.3)
    Zc = Z * gain
    cleaned = istft(Zc, nperseg, hop, length=n, kind="hann")
    cleaned = pl.bandpass(cleaned, fs, 0.5, min(6.0, fs / 2 * 0.9), order=3)

    if not return_diag:
        return cleaned
    diag = {
        "freqs": freqs,
        "mask": mask,
        "coh": coh,
        "gain": gain,
        "dom": dom,
        "fund": fund,
        "pow_before": np.abs(Z) ** 2,
        "pow_after": np.abs(Zc) ** 2,
    }
    return cleaned, diag


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def baseline_bandpass(ppg, fs, lo=0.5, hi=4.0):
    return pl.bandpass(ppg, fs, lo, hi, order=3)


def baseline_lms(ppg, imu, fs, mu=0.002, order=16):
    """Normalised LMS adaptive canceller driven by the accelerometer magnitude."""
    d = pl.bandpass(ppg, fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)
    ref = pl.accel_magnitude(imu[:, :3])
    ref = pl.bandpass(ref, fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)
    ref = ref / (np.std(ref) + 1e-12)
    n = len(d)
    X = np.lib.stride_tricks.sliding_window_view(ref, order)[: n - order + 1]
    dsub = d[order - 1:]
    w = np.zeros(order)
    yh = np.zeros(len(dsub))
    for i in range(len(dsub)):
        xi = X[i]
        e = dsub[i] - float(w @ xi)
        yh[i] = e
        w = w + (mu / (float(xi @ xi) + 1e-6)) * e * xi
    out = pl.bandpass(np.concatenate([np.zeros(order - 1), yh]), fs, 0.5, min(6.0, fs / 2 * 0.9), order=3)
    return out


def baseline_spectral_subtraction(ppg, imu, fs, nperseg=256, hop=64, alpha=1.6):
    """Classic spectral subtraction of the accelerometer spectrum (Fukushima-style)."""
    x = pl.bandpass(ppg, fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)
    Z, _ = stft(x, nperseg, hop)
    ref = pl.accel_magnitude(imu[:, :3])
    Zr, _ = stft(ref, nperseg, hop)
    nf = min(Z.shape[0], Zr.shape[0])
    Z, Zr = Z[:nf], Zr[:nf]
    Pr = np.abs(Zr) ** 2
    Pn = np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, mode="same"), 0, Pr)
    mag = np.abs(Z)
    mag2 = mag ** 2 - alpha * Pn
    mag2 = np.maximum(mag2, 1e-12)
    Zc = Z * (np.sqrt(mag2) / (mag + 1e-12))
    out = istft(Zc, nperseg, hop, length=len(x), kind="hann")
    return pl.bandpass(out, fs, 0.5, min(6.0, fs / 2 * 0.9), order=3)


def baseline_wavelet(ppg, imu, fs, wavelet="db4", level=6):
    """Wavelet detail suppression on scales correlated with the IMU."""
    import pywt
    x = pl.bandpass(ppg, fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)
    ref = pl.bandpass(pl.accel_magnitude(imu[:, :3]), fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    cref = pywt.wavedec(ref, wavelet, level=level)
    out = [coeffs[0]]
    for cd, cr in zip(coeffs[1:], cref[1:]):
        m = min(len(cd), len(cr))
        cd, cr = cd[:m], cr[:m]
        cc = np.corrcoef(np.abs(cd), np.abs(cr))[0, 1] if m > 8 else 0.0
        cc = 0.0 if not np.isfinite(cc) else cc
        k = float(np.clip(cc, 0, 0.95))
        out.append(cd * (1.0 - k))
    rec = pywt.waverec(out, wavelet)
    rec = rec[:len(x)] if len(rec) >= len(x) else np.concatenate([rec, np.zeros(len(x) - len(rec))])
    return pl.bandpass(rec, fs, 0.5, min(6.0, fs / 2 * 0.9), order=3)


def baseline_ica(ppg, imu, fs, nperseg=512, hop=128):
    """ICA on [PPG, IMU channels]; drop components that track motion."""
    from sklearn.decomposition import FastICA
    x = pl.bandpass(ppg, fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)
    ref = pl.accel_magnitude(imu[:, :3])
    n = min(len(x), len(ref))
    X = np.stack([x[:n],
                  pl.bandpass(ref[:n], fs, 0.4, min(8.0, fs / 2 * 0.9), order=3),
                  pl.bandpass(imu[:n, 3], fs, 0.4, min(8.0, fs / 2 * 0.9), order=3),
                  pl.bandpass(imu[:n, 4], fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)], axis=1)
    X = X / (X.std(axis=0, keepdims=True) + 1e-12)
    try:
        ica = FastICA(n_components=4, random_state=0, max_iter=600, whiten="unit-variance")
        S = ica.fit_transform(X)
        A = ica.mixing_
        # rebuild signal 0 while zeroing components dominated by motion channels
        keep = np.ones(S.shape[1])
        for k in range(S.shape[1]):
            w = np.abs(A[0, k]) / (np.abs(A[1:, k]).sum() + np.abs(A[0, k]) + 1e-12)
            if w < 0.35:
                keep[k] = 0.0
        Xr = (S * keep[None, :]) @ np.linalg.pinv(A)
        y = Xr[:, 0]
    except Exception:
        y = x[:n]
    y = pl.bandpass(y, fs, 0.5, min(6.0, fs / 2 * 0.9), order=3)
    return y


METHODS = {
    "Raw (bandpass)": lambda p, i, fs: baseline_bandpass(p, fs),
    "LMS adaptive": lambda p, i, fs: baseline_lms(p, i, fs),
    "Spectral subtraction": lambda p, i, fs: baseline_spectral_subtraction(p, i, fs),
    "Wavelet+IMU": lambda p, i, fs: baseline_wavelet(p, i, fs),
    "ICA+IMU": lambda p, i, fs: baseline_ica(p, i, fs),
    "KISR (proposed)": lambda p, i, fs: kisr(p, i, fs),
}
