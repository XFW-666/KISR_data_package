"""PPG-DaLiA replication of the Wrist channel law ("one accelerometer is enough").

Independent exercise cohort (15 subjects, Reiss et al., Sensors 2019) with
wrist PPG + inertial channels + chest ECG.  SAME cleaner (kisr.kisr) and SAME
estimator (ppglib.hr_harmonic, prev=None) and SAME statistics (subject-level
Wilcoxon + subject-clustered bootstrap, 4000 resamples) as
run_wrist_expanded.py, plus a pre-specified TOST equivalence test (margin
+/-2 bpm) on the per-subject paired MAE differences vs the full inertial
union.

ACTUAL distributed data (verified by inspection + PPG_FieldStudy_readme.pdf):
  * SX.pkl = {'signal': {'chest': {ACC, ECG, Resp, EDA/EMG/Temp dummy},
                       'wrist': {ACC, BVP, EDA, TEMP}},
              'activity' (4 Hz, IDs 0-8), 'rpeaks', 'label', 'questionnaire',
              'subject'};  synchronised, common t=0.
  * rates: wrist BVP (PPG) 64 Hz, wrist ACC 32 Hz, chest ACC/ECG/Resp 700 Hz,
    EDA/TEMP 4 Hz, activity 4 Hz.
  * NO gyroscope and NO magnetometer anywhere in the public release
    (readme III.1: RespiBAN = ECG, respiration, 3-axis ACC; III.2: E4 = ACC,
    BVP, EDA, TEMP; the raw S*_RespiBAN.h5 carries only CH1-CH8 = those
    modalities).  The requested gyro/mag ablations are therefore impossible
    without fabricating data; the inertial channel law is replicated with
    the TWO available accelerometers instead:
        ch_acc        wrist ACC (32 Hz, same site as the PPG)
        ch_acc_chest  chest ACC (700 Hz)
        ch_acc_both   wrist+chest ACC union (6 ch)  <- reference "full union"
  * NO running in the protocol (readme II: sitting 1, stairs 2, table soccer
    3, cycling 4, driving 5, lunch 6, walking 7, working 8; 0 = transient).
    Exercise activities used: walking (7), cycling (4) and stairs (2) as the
    third vigorous activity in place of the absent running (documented
    deviation, same fallback spirit as the spec's stairs provision).
  * S6.pkl contains only the first ~1.5 h (dataset-level hardware issue,
    readme III.3); the subject is kept with whatever exercise windows exist.

Protocol differences vs the Wrist run (all in the JSON protocol block):
  * windows 8 s, NON-overlapping (step 8 s); if the pilot subject
    extrapolates to > 40 min for all 15, the step is raised to 16 s for
    EVERY subject and logged;
  * a window needs >= 75% purity of the modal 4-Hz activity label
    (transition windows excluded);
  * IMU resampled onto the 64-Hz PPG grid with scipy.signal.resample_poly
    (zero-phase polyphase FIR, rational up/down); effective fs = 64 Hz;
  * KISR STFT at 64 Hz uses nperseg=64, hop=16 (1-s frame, 75% overlap) -
    the SAME time-frequency geometry as the Wrist native run (256/64 at
    256 Hz) and the 64-Hz sizing config of run_wrist_expanded.py;
  * reference HR from chest ECG R-peaks detected with ppglib.ecg_rpeaks at
    the native 700 Hz + ppglib.ref_hr median-RR (the distributed 'rpeaks'
    and 'label' fields are NOT used, to keep the Wrist protocol identical);
    40-180 bpm guard.

kisr treats the first 3 IMU columns as the magnitude reference - identical
convention to the Wrist single-channel ablations.

Outputs: out/_novelty/dalia_channel_law.json (+ dalia_windows_cache.json).
Usage:  python run_dalia_channel_law.py [resummarize]
"""
import fractions
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy import signal as sps
from scipy import stats as spst

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
pl, K = C.pl, C.K

OUT = C.HERE
DATA = os.path.join(OUT, "dalia", "data")
ACQ_PATH = os.path.join(OUT, "dalia_acquisition.json")
N_BOOT = 4000

WIN_S = 8.0
STEP_S = 8.0                    # non-overlapping; auto-escalates to 16 if needed
BUDGET_S = 40 * 60
FS_PPG = 64.0                   # effective fs (PPG grid)
# native sampling rates of the distributed pickles (readme III + verified
# length ratios at load time)
RATES = {"ppg": 64, "acc_wrist": 32, "acc_chest": 700, "ecg": 700,
         "resp": 700, "eda": 4, "temp": 4, "activity": 4}
# readme II activity IDs; keys match the Wrist convention where possible
ACT_MAP = {2: "stairs", 4: "bike", 7: "walk"}
ACT_LABEL = {"walk": "Walking", "bike": "Cycling", "stairs": "Stairs"}
PURITY = 0.75                   # min fraction of window with the modal label

CHANNEL_SETS = ["ch_acc", "ch_acc_chest", "ch_acc_both"]
FULL = "ch_acc_both"
TOST_MARGIN = 2.0               # bpm, PRE-SPECIFIED


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_pickles():
    pkls = []
    for root, _, files in os.walk(DATA):
        for fn in files:
            if fn.lower().endswith(".pkl"):
                pkls.append(os.path.join(root, fn))
    return sorted(pkls, key=lambda p: int(subject_of(p)[1:]))


def subject_of(pkl_path):
    return os.path.splitext(os.path.basename(pkl_path))[0]


def _resample_to_ppg(x, rate):
    """Zero-phase polyphase FIR resampling onto the 64-Hz PPG grid."""
    f = fractions.Fraction(int(FS_PPG), int(rate))
    if f.numerator == 1 and f.denominator == 1:
        return np.asarray(x, dtype=float)
    return sps.resample_poly(np.asarray(x, dtype=float), f.numerator,
                             f.denominator, axis=0)


def _fit_len(x, n):
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - len(x),) + x.shape[1:])], axis=0)


def load_dalia(path):
    """Load one DaLiA subject pickle; sanitise exactly like ppglib.load_wrist
    (interpolate non-finite PPG/ECG, zero-fill IMU) and resample IMU to 64 Hz."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    try:
        wrist, chest = d["signal"]["wrist"], d["signal"]["chest"]
        ppg = np.asarray(wrist["BVP"], dtype=float).reshape(-1)
        ecg = np.asarray(chest["ECG"], dtype=float).reshape(-1)
        aw = np.asarray(wrist["ACC"], dtype=float)
        ac = np.asarray(chest["ACC"], dtype=float)
        act = np.asarray(d["activity"], dtype=float).reshape(-1)
    except KeyError as e:
        raise KeyError(f"{path}: unexpected pickle structure ({e})")
    n = len(ppg)
    dur = n / RATES["ppg"]
    ratios = {"acc_wrist": round(len(aw) / RATES["acc_wrist"] / dur, 4),
              "acc_chest": round(len(ac) / RATES["acc_chest"] / dur, 4),
              "ecg": round(len(ecg) / RATES["ecg"] / dur, 4),
              "activity": round(len(act) / RATES["activity"] / dur, 4)}
    for v in (ppg, ecg):
        bad = ~np.isfinite(v)
        if bad.any():
            good = np.flatnonzero(~bad)
            if good.size == 0:
                raise ValueError(f"{path}: signal entirely non-finite")
            v[bad] = np.interp(np.flatnonzero(bad), good, v[good])
    imu = {}
    for name, v, rate in (("acc", aw, RATES["acc_wrist"]),
                          ("acc_chest", ac, RATES["acc_chest"])):
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"{path}: {name} shape {v.shape} != (n,3)")
        v = v.copy()
        v[~np.isfinite(v)] = 0.0
        imu[name] = _fit_len(_resample_to_ppg(v, rate), n)
    act[~np.isfinite(act)] = 0
    return {"ppg": ppg, "ecg": ecg, "activity": act.astype(int),
            "fs": FS_PPG, "fs_ecg": float(RATES["ecg"]),
            "duration_s": dur, "length_ratios": ratios, **imu}


def imu_for(rec, name):
    aw, ac = rec["acc"], rec["acc_chest"]
    return {
        "ch_acc": aw,
        "ch_acc_chest": ac,
        "ch_acc_both": np.concatenate([aw, ac], axis=1),
    }[name]


def window_activity(act, s, e, fs):
    """Modal dataset label inside [s,e) samples; (label, purity)."""
    a0 = int(np.floor(s / fs * RATES["activity"]))
    a1 = int(np.ceil(e / fs * RATES["activity"]))
    seg = act[a0:a1]
    if seg.size == 0:
        return None, 0.0
    vals, cnts = np.unique(seg, return_counts=True)
    lab = int(vals[int(np.argmax(cnts))])
    purity = float(cnts.max()) / seg.size
    return lab, purity


def process_subject(path, step_s):
    """Return (rows, stats) for one subject.  rows mirror the Wrist cache."""
    t0 = time.time()
    subj = subject_of(path)
    rec = load_dalia(path)
    fs = rec["fs"]
    ppg, ecg = rec["ppg"], rec["ecg"]
    rp = pl.ecg_rpeaks(ecg, rec["fs_ecg"])      # same detector, native 700 Hz
    imus = {name: imu_for(rec, name) for name in CHANNEL_SETS}
    wl = int(WIN_S * fs)
    sl = int(step_s * fs)
    # KISR STFT geometry identical to the Wrist native run (1-s frame, 75%
    # overlap): nperseg=fs, hop=fs/4, as in run_wrist_expanded.compute_window
    npg = max(16, int(round(fs)))
    hop = max(4, npg // 4)
    st = {"n_candidate": 0, "n_other_activity": 0, "n_impure": 0,
          "n_ref_invalid": 0, "n_ref_outside": 0, "n_used": 0,
          "used_by_act": {a: 0 for a in ACT_MAP.values()},
          "duration_s": round(rec["duration_s"], 1),
          "length_ratios": rec["length_ratios"],
          "n_rpeaks": int(len(rp))}
    rows = []
    for s in range(0, len(ppg) - wl + 1, sl):
        e = s + wl
        lab, purity = window_activity(rec["activity"], s, e, fs)
        if lab not in ACT_MAP:
            st["n_other_activity"] += 1
            continue
        st["n_candidate"] += 1
        if purity < PURITY:
            st["n_impure"] += 1
            continue
        ref = pl.ref_hr(rp, rec["fs_ecg"], s / fs, e / fs)
        if not np.isfinite(ref):
            st["n_ref_invalid"] += 1
            continue
        if ref < C.REF_LO or ref > C.REF_HI:
            st["n_ref_outside"] += 1
            continue
        act = ACT_MAP[lab]
        st["n_used"] += 1
        st["used_by_act"][act] += 1
        p = ppg[s:e]
        row = {"rec": subj, "subject": subj, "act": act,
               "t": round(s / fs, 1), "ref": round(float(ref), 2), "err": {}}
        for cfg in CHANNEL_SETS:
            try:
                y = K.kisr(p, imus[cfg][s:e], fs, nperseg=npg, hop=hop)
                hr, _ = pl.hr_harmonic(y, fs, prev=None)
                row["err"][cfg] = round(abs(float(hr) - ref), 4) \
                    if np.isfinite(hr) else None
            except Exception:
                row["err"][cfg] = None
        rows.append(row)
    st["elapsed_s"] = round(time.time() - t0, 1)
    log(f"  {subj}: {st['n_used']} windows "
        f"(walk {st['used_by_act']['walk']}, bike {st['used_by_act']['bike']}, "
        f"stairs {st['used_by_act']['stairs']}); excluded: other-activity "
        f"{st['n_other_activity']}, impure {st['n_impure']}, ref-invalid "
        f"{st['n_ref_invalid']}, ref-outside {st['n_ref_outside']}; "
        f"{st['elapsed_s']}s")
    return rows, st


def errors_by_subject(rows, cfg):
    out = {}
    for r in rows:
        v = r["err"].get(cfg)
        if v is not None:
            out.setdefault(r["subject"], []).append(v)
    return out


def tost(e_set, e_full, margin=TOST_MARGIN):
    """Two one-sided t-tests on per-subject paired MAE differences (set-full).

    H01: mean diff <= -margin  vs  H11: > -margin  (p_lower)
    H02: mean diff >= +margin  vs  H12: < +margin  (p_upper)
    Equivalence at alpha=0.05 iff both one-sided p < 0.05 (== 90% CI inside
    +/-margin).  Degenerate zero-variance case handled explicitly.
    """
    sa, sb = C.subj_mean_errs(e_set), C.subj_mean_errs(e_full)
    common = sorted(set(sa) & set(sb))
    d = np.array([sa[s] - sb[s] for s in common], dtype=float)
    n = d.size
    if n < 3:
        return {"n_subjects": n, "equivalent": None,
                "note": "fewer than 3 paired subjects"}
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    df = n - 1
    if sd <= 1e-12:
        inside = abs(mean) < margin
        return {"margin_bpm": margin, "mean_diff_bpm": mean, "sd_diff_bpm": 0.0,
                "n_subjects": n, "ci90": [mean, mean],
                "p_lower": 0.0 if inside else 1.0,
                "p_upper": 0.0 if inside else 1.0,
                "equivalent": bool(inside),
                "note": "zero variance (identical per-subject differences)"}
    se = sd / np.sqrt(n)
    t_lo = (mean + margin) / se
    t_hi = (mean - margin) / se
    p_lo = float(spst.t.sf(t_lo, df))
    p_hi = float(spst.t.cdf(t_hi, df))
    tcrit = float(spst.t.ppf(0.95, df))
    return {"margin_bpm": margin, "mean_diff_bpm": mean, "sd_diff_bpm": sd,
            "n_subjects": n, "se_bpm": float(se),
            "ci90": [mean - tcrit * se, mean + tcrit * se],
            "t_lower": float(t_lo), "t_upper": float(t_hi),
            "p_lower": p_lo, "p_upper": p_hi,
            "equivalent": bool(p_lo < 0.05 and p_hi < 0.05)}


def summarize(rows, subj_stats, subjects, step_s, acq, elapsed):
    e_full = errors_by_subject(rows, FULL)
    res = {
        "protocol": {
            "win_s": WIN_S, "step_s": step_s,
            "ref": "ECG R-peaks (ppglib.ecg_rpeaks, neurokit2) at native "
                   "700 Hz + median RR (ppglib.ref_hr); distributed 'rpeaks'/"
                   "'label' fields NOT used (Wrist protocol identity)",
            "ref_exclusion_bpm": [C.REF_LO, C.REF_HI],
            "estimator": "ppglib.hr_harmonic comb (0.6-3.4 Hz, 3 harmonics, "
                         "low-band gate 1.1 Hz, cont_w=0.35), prev=None",
            "effective_fs_hz": FS_PPG,
            "native_fs_hz": dict(RATES),
            "kisr_stft": "nperseg=64, hop=16 at 64 Hz (1-s frame, 75% "
                         "overlap) - same time-frequency geometry as the "
                         "Wrist native run (256/64 at 256 Hz) and the 64-Hz "
                         "sizing config of run_wrist_expanded.py",
            "imu_resampling": "scipy.signal.resample_poly (zero-phase "
                              "polyphase FIR, Kaiser beta=5) onto the 64-Hz "
                              "PPG grid: wrist acc up=2/down=1 (32->64 Hz), "
                              "chest acc up=16/down=175 (700->64 Hz); common "
                              "t=0 start (synchronised distributed pickles), "
                              "length ratios verified per subject",
            "activities": "dataset 4-Hz labels: walking=7, cycling=4, "
                          "stairs=2 (running is NOT in the DaLiA protocol; "
                          "stairs substituted as third vigorous activity); "
                          "modal label per window with >=75% purity "
                          "(transition windows excluded); sitting/soccer/"
                          "driving/lunch/working/transient excluded",
            "channel_sets": "ch_acc = wrist ACC; ch_acc_chest = chest ACC; "
                            "ch_acc_both = wrist+chest ACC union (reference). "
                            "The public PPG-DaLiA release has NO gyroscope "
                            "and NO magnetometer (readme III.1/III.2), so the "
                            "requested gyro/mag ablations cannot be run "
                            "without fabricating data; the inertial union "
                            "law is tested with the two available "
                            "accelerometers.",
            "continuity_prior": "prev=None for every channel set (as in "
                                "run_wrist_expanded channel ablations)",
            "bootstrap": f"subject-clustered, {N_BOOT} resamples, percentile "
                         "95% CI (common.cluster_bootstrap_ci / delta_summary)",
            "tost_margin_bpm": TOST_MARGIN,
            "tost_alpha": 0.05,
        },
        "acquisition": acq,
        "cohort": {
            "n_subjects": len(subjects), "n_records": len(subjects),
            "n_windows": len(rows), "subjects": subjects, "records": subjects,
            "windows_per_activity": {
                a: sum(1 for r in rows if r["act"] == a)
                for a in ("walk", "bike", "stairs")},
            "exclusions": {
                "other_activity_windows": sum(s["n_other_activity"]
                                              for s in subj_stats.values()),
                "impure_transition_windows": sum(s["n_impure"]
                                                 for s in subj_stats.values()),
                "ref_invalid_windows": sum(s["n_ref_invalid"]
                                           for s in subj_stats.values()),
                "ref_outside_40_180_windows": sum(s["n_ref_outside"]
                                                  for s in subj_stats.values()),
            },
            "per_subject": subj_stats,
            "note": "all 15 distributed subjects used; one record (pickle) "
                    "per subject; no subject dropped. S6.pkl is truncated to "
                    "the first ~1.5 h by the dataset itself (hardware issue, "
                    "readme III.3) and contributes only the exercise "
                    "windows inside that span.",
        },
        "channel_ablation": {}, "per_subject_mae": {}, "per_activity_mae": {},
    }
    for cfg in CHANNEL_SETS:
        e = errors_by_subject(rows, cfg)
        s = C.mae_summary(e, N_BOOT)
        if cfg != FULL:
            s["vs_full"] = C.delta_summary(e, e_full, N_BOOT)
        s["tost"] = tost(e, e_full)
        res["channel_ablation"][cfg] = s
        res["per_subject_mae"][cfg] = C.subj_mean_errs(e)
        by_act = {}
        for a in ("walk", "bike", "stairs"):
            vals = [r["err"][cfg] for r in rows
                    if r["act"] == a and r["err"].get(cfg) is not None]
            by_act[a] = {"mae": float(np.mean(vals)) if vals else None,
                         "n_windows": len(vals)}
        res["per_activity_mae"][cfg] = by_act

    sm_acc = res["per_subject_mae"]["ch_acc"]
    sm_full = C.subj_mean_errs(e_full)
    common = sorted(set(sm_acc) & set(sm_full))
    n_acc_better = sum(1 for s in common if sm_acc[s] < sm_full[s])
    n_acc_worse = sum(1 for s in common if sm_acc[s] > sm_full[s])
    res["acc_vs_full_subjects"] = {
        "acc_better": n_acc_better, "acc_worse": n_acc_worse,
        "ties": len(common) - n_acc_better - n_acc_worse,
        "n_subjects_paired": len(common)}
    res["notes"] = []
    res["elapsed_s"] = elapsed
    return res


def main():
    t0 = time.time()
    if not os.path.exists(ACQ_PATH):
        log("ERROR: dalia_acquisition.json missing - run download_dalia.py first")
        sys.exit(1)
    with open(ACQ_PATH) as f:
        acq = json.load(f)
    if not acq.get("ok"):
        log("ERROR: acquisition not ok")
        sys.exit(1)
    acq_block = {
        "url": acq["url"], "share_url": acq.get("share_url"),
        "official_page": acq.get("official_page"),
        "original_page": acq.get("original_page"),
        "original_page_status": acq.get("original_page_status"),
        "uci_mirror_fallback": acq.get("uci_mirror_fallback"),
        "filename": acq.get("filename"),
        "sha256": acq["sha256"], "bytes": acq["bytes"],
        "signal_rates_hz": dict(RATES),
        "pickle_keys": {
            "top": ["activity", "label", "questionnaire", "rpeaks", "signal",
                    "subject"],
            "signal.chest": ["ACC", "ECG", "Resp", "EDA (dummy)", "EMG (dummy)",
                             "Temp (dummy)"],
            "signal.wrist": ["ACC", "BVP (=PPG)", "EDA", "TEMP"],
        },
        "gyro_mag_available": False,
        "gyro_mag_note": "public PPG-DaLiA release contains no gyroscope or "
                         "magnetometer (readme III.1/III.2; raw RespiBAN h5 "
                         "channels CH1-CH8 = ECG/Resp/ACC/EDA/EMG/Temp); "
                         "inertial union = wrist ACC + chest ACC",
        "resampling_method": "scipy.signal.resample_poly onto the 64-Hz PPG "
                             "grid (wrist acc 2/1, chest acc 16/175), "
                             "zero-phase polyphase FIR; effective fs = 64 Hz",
        "activity_label_map": {"0": "transient", "1": "sitting", "2": "stairs",
                               "3": "table soccer", "4": "cycling",
                               "5": "driving", "6": "lunch break",
                               "7": "walking", "8": "working"},
        "running_note": "running is not part of the DaLiA protocol; stairs "
                        "(ID 2) used as the third vigorous activity",
    }

    if len(sys.argv) > 1 and sys.argv[1] == "resummarize":
        with open(os.path.join(OUT, "dalia_windows_cache.json")) as f:
            cache = json.load(f)
        rows, subj_stats = cache["rows"], cache["subject_stats"]
        subjects = sorted({r["subject"] for r in rows},
                          key=lambda s: int(s[1:]))
        res = summarize(rows, subj_stats, subjects, cache["step_s"],
                        acq_block, None)
        with open(os.path.join(OUT, "dalia_channel_law.json"), "w") as f:
            json.dump(res, f, indent=1)
        log("resummarized from cache -> dalia_channel_law.json")
        return

    pkls = find_pickles()
    if len(pkls) != 15:
        log(f"ERROR: expected 15 subject pickles under {DATA}, found {len(pkls)}")
        sys.exit(1)
    subjects = [subject_of(p) for p in pkls]
    log(f"subjects: {subjects}")

    # pilot on the first subject to check the compute budget (spec: raise the
    # step to 16 s for ALL subjects if the full set would exceed ~40 min)
    step_s = STEP_S
    log(f"pilot {subjects[0]} with step {step_s}s ...")
    pilot_rows, pilot_st = process_subject(pkls[0], step_s)
    est = pilot_st["elapsed_s"] * len(pkls)
    log(f"pilot {pilot_st['elapsed_s']}s -> extrapolated {est:.0f}s "
        f"for {len(pkls)} subjects (budget {BUDGET_S}s)")
    if est > BUDGET_S:
        step_s = 16.0
        log(f"BUDGET: raising step to {step_s}s for ALL subjects and "
            f"re-running the pilot")
        pilot_rows, pilot_st = process_subject(pkls[0], step_s)
        est = pilot_st["elapsed_s"] * len(pkls)
        log(f"re-pilot {pilot_st['elapsed_s']}s -> extrapolated {est:.0f}s")

    rows = list(pilot_rows)
    subj_stats = {subjects[0]: pilot_st}
    for p in pkls[1:]:
        r, st = process_subject(p, step_s)
        rows.extend(r)
        subj_stats[subject_of(p)] = st

    with open(os.path.join(OUT, "dalia_windows_cache.json"), "w") as f:
        json.dump({"step_s": step_s, "rows": rows, "subject_stats": subj_stats}, f)

    res = summarize(rows, subj_stats, subjects, step_s, acq_block,
                    round(time.time() - t0, 1))
    ca = res["channel_ablation"]
    notes = []
    ps = [f"{cfg}: Wilcoxon p={ca[cfg]['vs_full']['p_subject_wilcoxon']:.3f}, "
          f"delta={ca[cfg]['vs_full']['delta_mae']:+.3f} bpm"
          for cfg in CHANNEL_SETS if cfg != FULL
          if ca[cfg]["vs_full"]["p_subject_wilcoxon"] is not None]
    notes.append("vs full inertial union wrist+chest ACC (subject-level "
                 "Wilcoxon, pooled-window delta): " + "; ".join(ps))
    eq = [cfg for cfg in CHANNEL_SETS if ca[cfg]["tost"].get("equivalent")]
    notes.append(f"TOST equivalence at +/-{TOST_MARGIN} bpm (alpha=0.05, "
                 f"per-subject paired differences): equivalent sets = "
                 f"{eq if eq else 'NONE'}")
    av = res["acc_vs_full_subjects"]
    notes.append(f"single wrist acc better than full union in "
                 f"{av['acc_better']}/{av['n_subjects_paired']} subjects "
                 f"(worse in {av['acc_worse']}, ties {av['ties']})")
    notes.append("DEVIATIONS from the requested spec, forced by the actual "
                 "public release: (i) no gyro/mag exist in PPG-DaLiA, so the "
                 "channel law is tested on wrist ACC vs chest ACC vs their "
                 "union instead of acc/gyro/mag sets; (ii) no running in the "
                 "protocol, stairs used as third exercise activity; "
                 "(iii) non-overlapping 8-s windows and 64-Hz effective fs "
                 "with resampled IMU as specified.")
    notes.append("Replication of run_wrist_expanded.py on PPG-DaLiA with the "
                 "same kisr.kisr cleaner and ppglib.hr_harmonic estimator "
                 "(prev=None) and the same subject-level statistics.")
    res["notes"] = notes

    with open(os.path.join(OUT, "dalia_channel_law.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("== channel ablation ==")
    for cfg in CHANNEL_SETS:
        v = ca[cfg]
        line = (f"  {cfg:14s} MAE={v['mae']:.2f} "
                f"CI={v['ci95_subject_cluster']} n={v['n_windows']}")
        if cfg != FULL:
            line += (f" | vs full {v['vs_full']['delta_mae']:+.3f} "
                     f"p={v['vs_full']['p_subject_wilcoxon']}")
        t = v["tost"]
        line += (f" | TOST eq={t.get('equivalent')} "
                 f"p=({t.get('p_lower'):.4f},{t.get('p_upper'):.4f})"
                 if t.get("p_lower") is not None else " | TOST n/a")
        log(line)
    log(f"wrist acc better than full in {av['acc_better']}/"
        f"{av['n_subjects_paired']} subjects")
    log(f"windows per activity: {res['cohort']['windows_per_activity']}")
    log(f"elapsed {res['elapsed_s']}s -> dalia_channel_law.json")


if __name__ == "__main__":
    main()