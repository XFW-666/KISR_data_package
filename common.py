"""Shared helpers for the _novelty experiments (A: wrist expanded + LOSO,
B: BIDMC replication, C: transient gate).

Conventions reused from src/run_experiment.py / src/run_ablation.py /
src/run_nature_supp.py:
  * 8-s windows stepped by 4 s;
  * ECG R-peak reference via ppglib.ecg_rpeaks (neurokit2 detector, bandpass
    fallback) + ppglib.ref_hr (median RR inside the window);
  * windows with reference HR outside 40-180 bpm excluded (same guard as the
    original run_experiment.py);
  * harmonic-comb HR estimator ppglib.hr_harmonic (fmin 0.6, fmax 3.4 Hz,
    3 harmonics, low-band evidence gate, Gaussian continuity prior cont_w=0.35);
  * ADC quantisation: signed mid-tread levels 2^(bits-1)-1 after peak
    normalisation (identical to run_ablation.quantize).
Statistics: subject-level Wilcoxon signed-rank on per-subject mean errors and
subject-clustered bootstrap 95% CIs (resample subjects with replacement).
"""
import os
import sys

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import ppglib as pl          # noqa: E402
import kisr as K             # noqa: E402
from fetch import expected_dat_size  # noqa: E402

WRIST = os.path.join(ROOT, "data", "wrist")
BIDMC = os.path.join(ROOT, "data", "bidmc")
WIN_S, STEP_S = 8.0, 4.0
REF_LO, REF_HI = 40.0, 180.0   # exclusion guard, as in run_experiment.py


def subject_of(rec_name):
    return rec_name.split("_")[0]


def activity_of(record):
    r = record.split("_", 1)[1] if "_" in record else record
    if "run" in r:
        return "run"
    if "walk" in r:
        return "walk"
    return "bike"


def complete_wrist_records():
    """All wrist records whose .dat byte length matches the .hea sample count."""
    out, dropped = [], []
    for f in sorted(os.listdir(WRIST)):
        if not f.endswith(".dat"):
            continue
        hea = os.path.join(WRIST, f[:-4] + ".hea")
        if not os.path.exists(hea):
            dropped.append((f[:-4], "no .hea"))
            continue
        exp = expected_dat_size(hea)
        act = os.path.getsize(os.path.join(WRIST, f))
        if exp is not None and act == exp:
            out.append(f[:-4])
        else:
            dropped.append((f[:-4], f"dat {act} B != expected {exp} B"))
    return out, dropped


def window_starts(n, fs):
    wl, sl = int(WIN_S * fs), int(STEP_S * fs)
    if n < wl + sl:
        return []
    return list(range(0, n - wl, sl))


def quantize_bits(p, bits):
    """Signed quantisation identical to run_ablation.quantize."""
    pmax = np.max(np.abs(p)) + 1e-12
    levels = 2 ** (bits - 1) - 1
    return np.round(p / pmax * levels) / levels * pmax


def decimate(x, factor):
    """Plain decimation, as in run_ablation.decimate (wrist, integer factors)."""
    return x[::factor]


def cluster_bootstrap_ci(errs_by_subject, n_boot=4000, seed=0):
    """95% CI of the pooled MAE, resampling SUBJECTS with replacement."""
    subjects = sorted(errs_by_subject)
    arrs = [np.asarray(errs_by_subject[s], float) for s in subjects]
    arrs = [a[np.isfinite(a)] for a in arrs]
    keep = [a for a in arrs if a.size]
    if not keep:
        return [None, None]
    rng = np.random.default_rng(seed)
    k = len(keep)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, size=k)
        tot, cnt = 0.0, 0
        for i in idx:
            tot += keep[i].sum()
            cnt += keep[i].size
        means[b] = tot / cnt
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def subj_mean_errs(errs_by_subject):
    return {s: float(np.mean(v)) for s, v in errs_by_subject.items() if len(v)}


def paired_subject_wilcoxon(e_a, e_b):
    """Two-sided Wilcoxon signed-rank on per-subject mean errors (paired)."""
    sa, sb = subj_mean_errs(e_a), subj_mean_errs(e_b)
    common = sorted(set(sa) & set(sb))
    if len(common) < 3:
        return None, len(common)
    a = np.array([sa[s] for s in common])
    b = np.array([sb[s] for s in common])
    if np.allclose(a, b):
        return 1.0, len(common)
    try:
        p = float(sps.wilcoxon(a, b, alternative="two-sided").pvalue)
    except Exception:
        p = None
    return p, len(common)


def mae_summary(errs_by_subject, n_boot=4000, seed=0):
    all_e = np.concatenate([np.asarray(v, float) for v in errs_by_subject.values()]) \
        if errs_by_subject else np.array([])
    all_e = all_e[np.isfinite(all_e)]
    return {
        "mae": float(all_e.mean()) if all_e.size else None,
        "ci95_subject_cluster": cluster_bootstrap_ci(errs_by_subject, n_boot, seed),
        "n_windows": int(all_e.size),
        "n_subjects": int(len([s for s, v in errs_by_subject.items() if len(v)])),
    }


def delta_summary(e_gate, e_base, n_boot=4000, seed=1):
    """delta = MAE(gate) - MAE(base); negative = improvement.
    CI by subject-clustered bootstrap of the paired difference of subject means
    weighted by window counts (pooled-window delta resampled over subjects)."""
    subjects = sorted(set(e_gate) & set(e_base))
    rng = np.random.default_rng(seed)
    diffs = []
    pairs = []
    for s in subjects:
        a = np.asarray(e_gate[s], float)
        b = np.asarray(e_base[s], float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size and b.size:
            pairs.append((a, b))
    if not pairs:
        return {"delta": None, "ci95": [None, None], "p_subject_wilcoxon": None,
                "n_subjects": 0}
    k = len(pairs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        ta = tb = 0.0
        na = nb = 0
        for j in idx:
            a, b = pairs[j]
            ta += a.sum(); na += a.size
            tb += b.sum(); nb += b.size
        boots[i] = ta / na - tb / nb
    p, ns = paired_subject_wilcoxon(e_gate, e_base)
    mae_a = sum(a.sum() for a, _ in pairs) / sum(a.size for a, _ in pairs)
    mae_b = sum(b.sum() for _, b in pairs) / sum(b.size for _, b in pairs)
    return {
        "delta_mae": float(mae_a - mae_b),
        "ci95_subject_cluster": [float(np.percentile(boots, 2.5)),
                                 float(np.percentile(boots, 97.5))],
        "p_subject_wilcoxon": p,
        "n_subjects": ns,
    }