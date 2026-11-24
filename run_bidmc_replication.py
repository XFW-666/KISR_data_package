"""Experiment B: cross-dataset replication on PhysioNet BIDMC (53 subjects).

BIDMC has no IMU, so the front-end is baseline_bandpass (0.5-4 Hz) followed by
the SAME harmonic-comb estimator used in run_experiment.py (ppglib.hr_harmonic).
Reference: ECG lead II R-peaks with the manuscript detector (ppglib.ecg_rpeaks,
neurokit2 with bandpass fallback), median-RR HR per 8-s window, 40-180 bpm
exclusion guard - identical protocol to the wrist experiment.

Configs
  (i)  sizing: native 125 Hz vs 64/32/16 Hz (scipy resample_poly, rational
       64/125, 32/125, 16/125, FIR anti-alias) and bits 8/10/12/16 (signed
       peak-normalised quantisation, run_ablation convention);
  (ii) continuity prior on/off at native rate (BIDMC is resting/ICU: the prior
       is expected to HELP here, opposite to wrist motion segments -> supports
       activity-gated continuity).

All 53 records are used (no cherry-picking); records/windows without a valid
ECG reference are excluded by the same guard as the wrist protocol.
Statistics: subject-clustered bootstrap 95% CIs; subject-level Wilcoxon for
prior on/off and each sizing config vs native.

Output: out/_novelty/bidmc_replication.json (+ bidmc_windows_cache.json).
"""
import json
import os
import sys
import time

import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
pl, K = C.pl, C.K

OUT = C.HERE
N_BOOT = 4000
RATES = {"fs64": (64, 125), "fs32": (32, 125), "fs16": (16, 125)}


def bidmc_records():
    recs = []
    for f in sorted(os.listdir(C.BIDMC)):
        if f.endswith(".hea"):
            recs.append(f[:-4])
    return recs


def load_bidmc(rec_name):
    import wfdb
    rec = wfdb.rdrecord(os.path.join(C.BIDMC, rec_name))
    fs = float(rec.fs)
    idx = {n.strip().upper().rstrip(","): i for i, n in enumerate(rec.sig_name)}
    sig = rec.p_signal
    ppg = np.asarray(sig[:, idx["PLETH"]], float)
    ecg = np.asarray(sig[:, idx["II"]], float)   # lead II for R-peaks
    for v in (ppg, ecg):
        bad = ~np.isfinite(v)
        if bad.any():
            good = np.flatnonzero(~bad)
            if good.size:
                v[bad] = np.interp(np.flatnonzero(bad), good, v[good])
            else:
                v[:] = 0.0
    return fs, ppg, ecg


def run():
    t0 = time.time()
    recs = bidmc_records()
    if len(recs) != 53:
        raise SystemExit(f"expected 53 BIDMC records, found {len(recs)} - "
                         "download incomplete; refusing to cherry-pick")
    rows = []
    per_rec = {}
    for rn in recs:
        fs, ppg, ecg = load_bidmc(rn)
        rp = pl.ecg_rpeaks(ecg, fs)
        # pre-decimated PPG for the three reduced rates
        dec = {name: sps.resample_poly(ppg, up, down) for name, (up, down) in RATES.items()}
        n_used = 0
        prev_on = None
        wl, sl = int(C.WIN_S * fs), int(C.STEP_S * fs)
        for s in range(0, len(ppg) - wl, sl):
            e = s + wl
            ref = pl.ref_hr(rp, fs, s / fs, e / fs)
            if not np.isfinite(ref) or ref < C.REF_LO or ref > C.REF_HI:
                continue
            n_used += 1
            p = ppg[s:e]
            row = {"rec": rn, "subject": rn, "t": round(s / fs, 1),
                   "ref": round(float(ref), 2), "err": {}}

            def est(y, fsu, prev):
                hr, _ = pl.hr_harmonic(y, fsu, prev=prev)
                return hr

            try:
                # native 125 Hz, 16-bit, prior off  (== prior_off alias)
                y = K.baseline_bandpass(p, fs)
                hr = est(y, fs, None)
                row["err"]["native"] = round(abs(hr - ref), 4) if np.isfinite(hr) else None
                # prior on (same cleaned signal, continuity tracking)
                hr_on = est(y, fs, prev_on)
                if np.isfinite(hr_on):
                    prev_on = hr_on
                row["err"]["prior_on"] = round(abs(hr_on - ref), 4) if np.isfinite(hr_on) else None
                # reduced rates
                for name, (up, down) in RATES.items():
                    fsd = fs * up / down
                    sd = int(round(s * up / down))
                    wld = int(C.WIN_S * fsd)
                    pd = dec[name][sd:sd + wld]
                    if pd.size < int(4 * fsd):
                        row["err"][name] = None
                        continue
                    yd = K.baseline_bandpass(pd, fsd)
                    hrd = est(yd, fsd, None)
                    row["err"][name] = round(abs(hrd - ref), 4) if np.isfinite(hrd) else None
                # quantisation at native rate
                for bits in (8, 10, 12):
                    pq = C.quantize_bits(p, bits)
                    yq = K.baseline_bandpass(pq, fs)
                    hrq = est(yq, fs, None)
                    row["err"][f"b{bits}"] = round(abs(hrq - ref), 4) if np.isfinite(hrq) else None
            except Exception:
                for k in ["native", "prior_on", *RATES, "b8", "b10", "b12"]:
                    row["err"].setdefault(k, None)
            rows.append(row)
        per_rec[rn] = n_used
        print(f"  {rn}: {n_used} windows ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "bidmc_windows_cache.json"), "w") as f:
        json.dump(rows, f)
    return rows, recs, per_rec


def errs_of(rows, cfg):
    out = {}
    for r in rows:
        v = r["err"].get(cfg)
        if v is not None:
            out.setdefault(r["subject"], []).append(v)
    return out


def summarize(rows, recs, per_rec):
    subjects = sorted({r["subject"] for r in rows})
    res = {
        "protocol": {
            "dataset": "PhysioNet BIDMC PPG and Respiration v1.0.0 (legacy archive)",
            "source": "https://physionet.org/physiobank/database/bidmc/",
            "n_records_downloaded": len(recs),
            "fs_native_hz": 125, "ppg_channel": "PLETH", "ecg_reference": "lead II",
            "win_s": C.WIN_S, "step_s": C.STEP_S,
            "ref_exclusion_bpm": [C.REF_LO, C.REF_HI],
            "frontend": "baseline_bandpass 0.5-4 Hz (no IMU available)",
            "estimator": "ppglib.hr_harmonic comb, identical to run_experiment.py",
            "rate_conversion": "scipy resample_poly (64/125, 32/125, 16/125)",
            "quantisation": "signed 2^(bits-1)-1 levels after peak normalisation",
            "continuity_prior": "sizing configs prev=None; prior_on tracks the "
                                "previous estimate per record at native rate",
        },
        "cohort": {"n_subjects": len(subjects), "n_records": len(recs),
                   "n_windows": len(rows),
                   "records_zero_windows": [r for r, n in per_rec.items() if n == 0]},
        "sizing": {}, "continuity": {}, "per_subject_mae": {},
    }
    native = errs_of(rows, "native")
    res["sizing"]["native_125Hz_16bit"] = C.mae_summary(native, N_BOOT)
    for cfg in ["fs64", "fs32", "fs16", "b8", "b10", "b12"]:
        e = errs_of(rows, cfg)
        s = C.mae_summary(e, N_BOOT)
        s["vs_native"] = C.delta_summary(e, native, N_BOOT)
        res["sizing"][cfg] = s
        res["per_subject_mae"][cfg] = C.subj_mean_errs(e)
    e_on = errs_of(rows, "prior_on")
    sm_on, sm_off = C.subj_mean_errs(e_on), C.subj_mean_errs(native)
    common = sorted(set(sm_on) & set(sm_off))
    res["continuity"] = {
        "prior_on": C.mae_summary(e_on, N_BOOT),
        "prior_off": C.mae_summary(native, N_BOOT),
        "on_minus_off": C.delta_summary(e_on, native, N_BOOT),
        "subjects_prior_better": int(sum(1 for s in common
                                         if sm_on[s] < sm_off[s])),
        "subjects_prior_worse": int(sum(1 for s in common
                                        if sm_on[s] > sm_off[s])),
        "n_subjects_paired": len(common),
        "interpretation": "resting/ICU contrast to wrist motion: a beneficial "
                          "prior here would support activity-gated continuity; "
                          "the observed sign is reported as-is",
    }
    res["per_subject_mae"]["prior_on"] = C.subj_mean_errs(e_on)
    res["per_subject_mae"]["native"] = C.subj_mean_errs(native)
    return res


def main():
    t0 = time.time()
    rows, recs, per_rec = run()
    res = summarize(rows, recs, per_rec)
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "bidmc_replication.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\n== sizing ==", flush=True)
    for k, v in res["sizing"].items():
        print(f"  {k:20s} MAE={v['mae']:.2f} CI={v['ci95_subject_cluster']} n={v['n_windows']}",
              flush=True)
    print("== continuity ==", flush=True)
    ct = res["continuity"]
    print(f"  on={ct['prior_on']['mae']:.2f} off={ct['prior_off']['mae']:.2f} "
          f"delta={ct['on_minus_off']['delta_mae']:+.2f} "
          f"p={ct['on_minus_off']['p_subject_wilcoxon']}", flush=True)
    print(f"elapsed {res['elapsed_s']}s -> bidmc_replication.json", flush=True)


if __name__ == "__main__":
    main()