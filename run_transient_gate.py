"""Experiment C: transient-aware gating on top of KISR (wrist, hard activities).

Motivation: the interference-index analysis (out/mechanism.json) was negative,
pointing at NON-harmonic transients (impacts, stride shocks) rather than
harmonic motion overlap as the residual failure mode. Two gate variants are
added to KISR, both driven by a per-STFT-frame transient flag:

  flag = (robust jerk z-score > 2.5) AND (excess kurtosis > 2.0 OR
          spectral flatness in 0.5-8 Hz > 0.30)

  * IMU jerk magnitude: ||d(acc)/dt|| per frame (RMS), robust z within window;
  * PPG excess kurtosis per frame (impulsiveness);
  * PPG spectral flatness per frame (broadband = non-harmonic content).

Variants
  gate_soft : on flagged frames reduce subspace-subtraction strength to 30%
              (gain <- 1 - 0.3*(1-gain)); everything else identical to KISR.
  gate_bp   : on flagged frames replace the KISR spectrum with a conservative
              0.7-3.0 Hz bandpass of the same input (cardiac-only band);
              non-flagged frames keep the standard KISR gain.

kisr_gated(variant=None) reproduces src/kisr.py kisr() exactly (verified
numerically in --check mode). Evaluation: hard activities only
(high_resistance_bike + run), 8-s/4-s windows, ECG R-peak reference, 40-180 bpm
guard, harmonic comb estimator with prev=None for ALL arms (isolates the gate).
Subject-level paired Wilcoxon + subject-clustered bootstrap CIs.

Output: out/_novelty/transient_gate.json (+ transient_windows_cache.json).
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
pl, K = C.pl, C.K

OUT = C.HERE
N_BOOT = 4000

# transient-flag thresholds (fixed from signal characteristics, not tuned on MAE)
JERK_Z_THR = 2.5
KURT_THR = 2.0
FLAT_THR = 0.30
SOFT_KEEP = 0.30          # fraction of subtraction retained on flagged frames
BP_LO, BP_HI = 0.7, 3.0   # conservative cardiac band for gate_bp


def frame_transient_flags(x, P, freqs, acc, fs, nperseg, hop, nf):
    """Per-frame transient flags + diagnostics. x: bandpassed PPG; P: |STFT|^2."""
    n = len(x)
    # IMU jerk magnitude (differential acceleration)
    jerk = np.linalg.norm(np.diff(acc, axis=0) * fs, axis=1)
    jr = np.zeros(nf)
    ku = np.zeros(nf)
    fl = np.zeros(nf)
    band = (freqs >= 0.5) & (freqs <= min(8.0, freqs[-1]))
    eps = 1e-12
    for i in range(nf):
        s0 = i * hop
        seg = jerk[s0:s0 + nperseg - 1]
        jr[i] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
        xs = x[s0:s0 + nperseg]
        ku[i] = float(sps.kurtosis(xs, fisher=True)) if xs.size > 3 else 0.0
        Pb = P[i, band] + eps
        fl[i] = float(np.exp(np.mean(np.log(Pb))) / np.mean(Pb))
    med = np.median(jr)
    mad = np.median(np.abs(jr - med)) * 1.4826 + 1e-9
    jz = (jr - med) / mad
    flags = (jz > JERK_Z_THR) & ((ku > KURT_THR) | (fl > FLAT_THR))
    return flags, {"jerk_z": jz, "kurt": ku, "flat": fl}


def kisr_gated(ppg, imu, fs, variant=None, nperseg=256, hop=64, n_harm=3,
               gate_k=1.0, max_atten=0.92, protect_cardiac=True,
               return_diag=False):
    """KISR with optional transient gate. variant=None -> identical to K.kisr.

    Body mirrors src/kisr.py kisr() line-for-line; the only additions are the
    transient-flag computation and the two gate interventions.
    """
    ppg = np.asarray(ppg, dtype=float)
    n = len(ppg)
    x = pl.bandpass(ppg, fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)

    mask, freqs, fund = K.motion_subspace(imu, fs, nperseg, hop, n_harm=n_harm)

    acc_ref = pl.accel_magnitude(imu[:, :3])
    acc_ref = acc_ref - acc_ref.mean()
    coh = K.ppg_motion_coherence(x, acc_ref, fs, nperseg, hop)

    Z, _ = K.stft(x, nperseg, hop)
    nf = min(Z.shape[0], mask.shape[0], coh.shape[0])
    Z, mask, coh = Z[:nf], mask[:nf], coh[:nf]
    P = np.abs(Z) ** 2

    protect = K.harmonic_protection(freqs, P.mean(axis=0)) if protect_cardiac \
        else np.zeros(len(freqs), bool)

    tot = P.sum(axis=1) + 1e-18
    dom = (P * mask).sum(axis=1) / tot

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

    card_lo, card_hi = 0.7, 3.2
    band = (freqs >= card_lo) & (freqs <= card_hi)

    gain = np.ones_like(P)
    for i in range(nf):
        a = float(np.clip(dom[i] * gate_k * (0.35 + 0.65 * motion_lv[i]), 0.0, 1.0))
        g = 1.0 - a * coh[i]
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
                rel = float(Pb[k] / pmax)
                floor = 0.35 + 0.4 * rel
                g[near0 | near2] = np.maximum(g[near0 | near2], floor)
        gain[i] = np.clip(g, 1.0 - max_atten * (0.5 + 0.5 * motion_lv[i]), 1.0)

    # ---- transient gate (the only addition over KISR) ----
    flags, tdiag = frame_transient_flags(x, P, freqs, imu[:, :3], fs,
                                         nperseg, hop, nf)
    if variant == "soft":
        gain[flags] = 1.0 - SOFT_KEEP * (1.0 - gain[flags])
    gain = np.maximum(gain, 0.3)
    Zc = Z * gain
    if variant == "bp":
        xbp = pl.bandpass(x, fs, BP_LO, min(BP_HI, fs / 2 * 0.9), order=3)
        Zbp, _ = K.stft(xbp, nperseg, hop)
        Zc[flags] = Zbp[:nf][flags]
    cleaned = K.istft(Zc, nperseg, hop, length=n, kind="hann")
    cleaned = pl.bandpass(cleaned, fs, 0.5, min(6.0, fs / 2 * 0.9), order=3)

    if not return_diag:
        return cleaned
    diag = {"flags": flags, "flag_frac": float(flags.mean()) if nf else 0.0,
            "jerk_z": tdiag["jerk_z"], "kurt": tdiag["kurt"], "flat": tdiag["flat"],
            "gain": gain, "dom": dom}
    return cleaned, diag


def hard_records():
    recs, _ = C.complete_wrist_records()
    return [r for r in recs
            if "high_resistance_bike" in r or r.endswith("_run")]


def check_equivalence():
    rec = pl.load_wrist(os.path.join(C.WRIST, "s3_run"))
    fs = rec["fs"]
    p = rec["ppg"][:2048]
    imu = np.concatenate([rec["acc"], rec["gyro"]], axis=1)[:2048]
    a = K.kisr(p, imu, fs)
    b = kisr_gated(p, imu, fs, variant=None)
    d = float(np.max(np.abs(a - b)))
    print(f"gate=None vs K.kisr max|diff| = {d:.3e}")
    assert d < 1e-9, "gated implementation diverges from KISR baseline"
    for v in ("soft", "bp"):
        y, dg = kisr_gated(p, imu, fs, variant=v, return_diag=True)
        print(f"  variant {v}: flag_frac={dg['flag_frac']:.3f} "
              f"jerk_z p90={np.percentile(dg['jerk_z'],90):.2f} "
              f"kurt p90={np.percentile(dg['kurt'],90):.2f} "
              f"flat p90={np.percentile(dg['flat'],90):.2f}")


def diag_thresholds():
    """Distribution of transient statistics over hard records (no MAE)."""
    recs = hard_records()
    allj, allk, allf, allflag = [], [], [], []
    for rn in recs[:4]:
        rec = pl.load_wrist(os.path.join(C.WRIST, rn))
        fs = rec["fs"]
        imu = np.concatenate([rec["acc"], rec["gyro"]], axis=1)
        for s in C.window_starts(len(rec["ppg"]), fs)[::4]:
            e = s + int(C.WIN_S * fs)
            p = rec["ppg"][s:e]
            x = pl.bandpass(p, fs, 0.4, min(8.0, fs / 2 * 0.9), order=3)
            Z, _ = K.stft(x, 256, 64)
            nf = Z.shape[0]
            P = np.abs(Z) ** 2
            freqs = np.fft.rfftfreq(256, d=1.0 / fs)
            flags, td = frame_transient_flags(x, P, freqs, imu[s:e, :3], fs,
                                              256, 64, nf)
            allj.extend(td["jerk_z"]); allk.extend(td["kurt"])
            allf.extend(td["flat"]); allflag.extend(flags)
    q = [50, 75, 90, 95, 99]
    print("jerk_z pct", dict(zip(q, np.round(np.percentile(allj, q), 2))))
    print("kurt   pct", dict(zip(q, np.round(np.percentile(allk, q), 2))))
    print("flat   pct", dict(zip(q, np.round(np.percentile(allf, q), 2))))
    print(f"flagged frame fraction = {np.mean(allflag):.3f} (n={len(allflag)})")


def run():
    t0 = time.time()
    recs = hard_records()
    subjects = sorted({C.subject_of(r) for r in recs})
    print(f"hard records: {recs}\nsubjects: {subjects}", flush=True)
    arms = ["kisr", "gate_soft", "gate_bp"]
    rows = []
    flag_stats = {a: [] for a in arms}
    for rn in recs:
        rec = pl.load_wrist(os.path.join(C.WRIST, rn))
        fs = rec["fs"]
        ppg = rec["ppg"]
        imu = np.concatenate([rec["acc"], rec["gyro"]], axis=1)
        rp = pl.ecg_rpeaks(rec["ecg"], fs)
        n_used = 0
        for s in C.window_starts(len(ppg), fs):
            e = s + int(C.WIN_S * fs)
            ref = pl.ref_hr(rp, fs, s / fs, e / fs)
            if not np.isfinite(ref) or ref < C.REF_LO or ref > C.REF_HI:
                continue
            n_used += 1
            p, i = ppg[s:e], imu[s:e]
            row = {"rec": rn, "subject": C.subject_of(rn),
                   "act": "bike_high" if "high_resistance_bike" in rn else "run",
                   "t": round(s / fs, 1), "ref": round(float(ref), 2), "err": {}}
            for arm in arms:
                try:
                    if arm == "kisr":
                        y = K.kisr(p, i, fs)
                        ff = None
                    else:
                        v = "soft" if arm == "gate_soft" else "bp"
                        y, dg = kisr_gated(p, i, fs, variant=v, return_diag=True)
                        ff = dg["flag_frac"]
                        flag_stats[arm].append(ff)
                    hr, _ = pl.hr_harmonic(y, fs, prev=None)
                    row["err"][arm] = round(abs(float(hr) - ref), 4) \
                        if np.isfinite(hr) else None
                except Exception:
                    row["err"][arm] = None
            rows.append(row)
        print(f"  {rn}: {n_used} windows ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "transient_windows_cache.json"), "w") as f:
        json.dump(rows, f)
    return rows, recs, subjects, flag_stats


def errs_of(rows, arm, act=None):
    out = {}
    for r in rows:
        if act is not None and r["act"] != act:
            continue
        v = r["err"].get(arm)
        if v is not None:
            out.setdefault(r["subject"], []).append(v)
    return out


def summarize(rows, recs, subjects, flag_stats):
    res = {
        "protocol": {
            "win_s": C.WIN_S, "step_s": C.STEP_S,
            "ref_exclusion_bpm": [C.REF_LO, C.REF_HI],
            "estimator": "ppglib.hr_harmonic comb, prev=None for all arms",
            "activities": "high_resistance_bike + run (hard)",
            "gate": {"jerk_z_thr": JERK_Z_THR, "kurt_thr": KURT_THR,
                     "flat_thr": FLAT_THR, "soft_keep": SOFT_KEEP,
                     "bp_band_hz": [BP_LO, BP_HI],
                     "rule": "flag = jerk_z>2.5 AND (kurt>2.0 OR flat>0.30); "
                             "jerk z is robust (median/MAD) within each 8-s window"},
        },
        "cohort": {"n_subjects": len(subjects), "n_records": len(recs),
                   "n_windows": len(rows), "subjects": subjects, "records": recs},
        "flagged_frame_fraction": {
            a: {"mean": float(np.mean(v)) if v else None,
                "p90": float(np.percentile(v, 90)) if v else None}
            for a, v in flag_stats.items() if v},
        "overall": {}, "by_activity": {}, "per_subject_mae": {},
    }
    base = errs_of(rows, "kisr")
    for arm in ["kisr", "gate_soft", "gate_bp"]:
        e = errs_of(rows, arm)
        s = C.mae_summary(e, N_BOOT)
        if arm != "kisr":
            s["vs_kisr"] = C.delta_summary(e, base, N_BOOT)
        res["overall"][arm] = s
        res["per_subject_mae"][arm] = C.subj_mean_errs(e)
    for act in ["bike_high", "run"]:
        res["by_activity"][act] = {}
        b = errs_of(rows, "kisr", act)
        for arm in ["kisr", "gate_soft", "gate_bp"]:
            e = errs_of(rows, arm, act)
            s = C.mae_summary(e, N_BOOT)
            if arm != "kisr":
                s["vs_kisr"] = C.delta_summary(e, b, N_BOOT)
            res["by_activity"][act][arm] = s
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "check":
        check_equivalence()
        return
    if mode == "diag":
        diag_thresholds()
        return
    t0 = time.time()
    check_equivalence()
    rows, recs, subjects, flag_stats = run()
    res = summarize(rows, recs, subjects, flag_stats)
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "transient_gate.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\n== overall ==", flush=True)
    for arm, s in res["overall"].items():
        line = f"  {arm:10s} MAE={s['mae']:.2f} CI={s['ci95_subject_cluster']}"
        if "vs_kisr" in s:
            d = s["vs_kisr"]
            line += f" delta={d['delta_mae']:+.2f} p={d['p_subject_wilcoxon']}"
        print(line, flush=True)
    print("flagged fractions:", res["flagged_frame_fraction"], flush=True)
    print(f"elapsed {res['elapsed_s']}s -> transient_gate.json", flush=True)


if __name__ == "__main__":
    main()