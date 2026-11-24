"""Column-native (3.4 in) re-renders of the manuscript data figures.

Print-native typesetting: point sizes specified here are the printed point
sizes at 3.4 in width (one IEEE JSEN column).  Data, records, windows, colors
and message are identical to the reference scripts (src/fig2_signal.py,
src/fig_freq_mechanism.py, src/fig_sensor_concepts.py::fig15,
out/_novelty/make_figures.py); only physical size, font scale and panel
arrangement change.

All six PNGs are saved at dpi=600 without bbox_inches="tight", so every file
is exactly 3.4*600 = 2040 px wide.  Re-running regenerates identical files.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import figstyle as FS          # Arial family + Okabe-Ito palette
import ppglib as pl
import kisr as K
import fig2_signal as F2S      # record-completeness logic (same data choice)
import fig_freq_mechanism as FM  # hard-window picking (same window choice)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

WRIST = os.path.join(ROOT, "data", "wrist")
NOV = os.path.join(ROOT, "out", "_novelty")

# ---------------------------------------------------------------------------
# print-native typography: >= 6.5 pt ticks, >= 7 pt labels/titles, 7 pt bold
# panel letters.  At 3.4 in the specified pt IS the printed pt.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 100,
    "savefig.dpi": 600,
})
DPI = 600
W_IN = 3.4


def panel_letter(ax, x, y, s, va="center"):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7.0, fontweight="bold",
            ha="center", va=va, clip_on=False)


def save(fig, name):
    """Save at exact 3.4 in width, flatten to RGB white background."""
    from PIL import Image
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=DPI, facecolor="white")
    im = Image.open(path)
    assert im.width == int(W_IN * DPI), (name, im.width)
    im.convert("RGB").save(path)
    plt.close(fig)
    print("saved", path, im.size, "-> %.2f x %.2f in" % (im.width / DPI, im.height / DPI))


# ---------------------------------------------------------------------------
# fig1_col: real signals + spectra (from src/fig2_signal.py, restacked)
# ---------------------------------------------------------------------------
def fig1_col():
    recs = F2S.complete_records()
    recs.sort(key=lambda r: (0 if "run" in r else 1 if "walk" in r else 2, r))
    chosen = {}
    for act in ("bike", "walk", "run"):
        for r in recs:
            if act in r and act not in chosen:
                chosen[act] = r
    print("fig1 records:", chosen)

    show_s = 12.0
    fig = plt.figure(figsize=(W_IN, 3.8))
    gs = fig.add_gridspec(4, 3, left=0.165, right=0.985, top=0.895,
                          bottom=0.145, hspace=0.62, wspace=0.42,
                          height_ratios=[1, 1, 1, 1.05])
    spectra = {}
    letters = ["(a)", "(b)", "(c)"]
    # column titles: same activity + record text as the reference, wrapped so
    # each line fits one ~0.8 in column at 7 pt
    TITLE = {"bike": "Biking\n(s1 high\nresistance bike)",
             "walk": "Walking\n(s1 walk)",
             "run": "Running\n(s3 run)"}
    for col, (act, rec) in enumerate(chosen.items()):
        r = pl.load_wrist(os.path.join(WRIST, rec))
        fs = r["fs"]
        n = int(show_s * fs)
        t = np.arange(n) / fs
        ecg = r["ecg"][:n]
        ppg = pl.bandpass(r["ppg"][:n], fs, 0.5, 8.0)
        acc = pl.accel_magnitude(r["acc"][:n]) - np.mean(
            pl.accel_magnitude(r["acc"][:n]))

        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, ecg, color=FS.C["black"], lw=0.7)
        ax.set_title(TITLE[act])
        ax.set_ylabel("ECG (mV)" if col == 0 else "")
        ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
        ax.set_xlim(0, show_s)
        ax.set_xticks([0, 5, 10])
        ax.tick_params(labelbottom=False)
        if col == 0:
            panel_letter(ax, -0.52, 1.08, letters[0])

        ax = fig.add_subplot(gs[1, col])
        ax.plot(t, ppg, color=FS.C["red"], lw=0.7)
        ax.set_ylabel("PPG (a.u.)" if col == 0 else "")
        ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
        ax.set_xlim(0, show_s)
        ax.set_xticks([0, 5, 10])
        ax.tick_params(labelbottom=False)
        if col == 0:
            panel_letter(ax, -0.52, 1.08, letters[1])

        ax = fig.add_subplot(gs[2, col])
        ax.plot(t, acc, color=FS.C["blue"], lw=0.7)
        ax.set_ylabel(r"$|a|$ (m/s$^2$)" if col == 0 else "")
        ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
        ax.set_xlabel("Time (s)" if col == 2 else "", labelpad=1.0)
        ax.set_xlim(0, show_s)
        ax.set_xticks([0, 5, 10])
        if col == 0:
            panel_letter(ax, -0.52, 1.08, letters[2])

        f, P = pl.welch(ppg, fs, nperseg_s=8.0)
        spectra[act] = (f, P)

    ax = fig.add_subplot(gs[3, :])
    handles, hlabels = [], []
    for act, (f, P) in spectra.items():
        band = (f >= 0.4) & (f <= 5.0)
        Pn = P[band] / P[band].max()
        h, = ax.plot(f[band], Pn, color=FS.ACT_COLORS[act],
                     label=FS.ACT_LABEL[act])
        handles.append(h)
        hlabels.append(FS.ACT_LABEL[act])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PPG PSD")
    ax.set_xlim(0.4, 5.0)
    # the three normalized spectra saturate the upper band wherever an
    # in-axes legend would fit, so the legend moves below the x label
    fig.legend(handles, hlabels, ncol=3, frameon=False, loc="lower center",
               handlelength=1.4, handletextpad=0.4, columnspacing=1.0,
               bbox_to_anchor=(0.5, 0.012 / 3.8))
    panel_letter(ax, -0.14, 1.08, "(d)")

    save(fig, "fig1_col.png")


# ---------------------------------------------------------------------------
# fig2_col: artifact mechanism (from src/fig_freq_mechanism.py, stacked)
# ---------------------------------------------------------------------------
def fig2_col():
    p, a, fs, t0 = FM.pick_hard_window(FM.HARD_REC)
    print("fig2 hard window t0=%.1fs" % t0)
    imu = np.concatenate([a, np.zeros_like(a)], axis=1)
    x = pl.bandpass(p, fs, 0.4, 8.0)
    _, diag = K.kisr(x, imu, fs, nperseg=FM.NPERSEG, hop=FM.HOP,
                     return_diag=True)
    freqs = diag["freqs"]
    gain = diag["gain"]
    fund = diag["fund"]
    f0 = float(np.median(fund[fund > 0])) if np.any(fund > 0) else 0.0

    fig = plt.figure(figsize=(W_IN, 3.35))
    fig.set_constrained_layout(True)
    ax_e, ax_f, ax_g = fig.subplots(3, 1)

    # ---- (a) spectrum before vs after, motion harmonics marked
    Pb = diag["pow_before"].mean(axis=0)
    Pa = diag["pow_after"].mean(axis=0)
    band = (freqs >= 0.4) & (freqs <= 8.0)
    fb = freqs[band]
    ax_e.semilogy(fb, Pb[band] + 1e-12, color=FS.C["black"], lw=0.9,
                  label="before")
    ax_e.semilogy(fb, Pa[band] + 1e-12, color=FS.C["red"], lw=0.9,
                  label="after KISR")
    ax_e.axvspan(FM.CARD_LO, FM.CARD_HI, color="#cfe8f3", alpha=0.45, lw=0,
                 zorder=0)
    for k in (1, 2, 3, 4):
        if f0 > 0 and k * f0 <= 8.0:
            ax_e.axvline(k * f0, color="#7a7a7a", ls=":", lw=0.7)
            ax_e.text(k * f0, 0.06, f"{k}f" if k > 1 else "f",
                      transform=ax_e.get_xaxis_transform(), fontsize=6.5,
                      color="#555555", ha="center", va="bottom")
    ax_e.set_xlim(0.4, 8.0)
    ax_e.set_xlabel("Frequency (Hz)")
    ax_e.set_ylabel("Mean STFT power")
    ax_e.legend(frameon=False, fontsize=6.5, loc="upper right",
                handlelength=1.4, handletextpad=0.4)
    ax_e.set_title("(a)", loc="left", fontweight="bold", x=-0.02)

    # ---- (b) applied gain map
    t_end = gain.shape[0] * FM.HOP / fs
    im = ax_f.imshow(gain.T, origin="lower", aspect="auto",
                     extent=[0, t_end, freqs[0], freqs[-1]],
                     cmap="RdYlGn", vmin=0.3, vmax=1.0,
                     interpolation="nearest")
    ax_f.set_ylim(0, 8.0)
    ax_f.axhline(FM.CARD_LO, color="#1f4e79", lw=0.8, ls="--")
    ax_f.axhline(FM.CARD_HI, color="#1f4e79", lw=0.8, ls="--")
    ax_f.set_xlabel("Time within window (s)")
    ax_f.set_ylabel("Frequency (Hz)")
    cb = fig.colorbar(im, ax=ax_f, fraction=0.046, pad=0.02)
    cb.set_label("applied gain", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6.5)
    ax_f.set_title("(b)", loc="left", fontweight="bold", x=-0.02)

    # ---- (c) mean PPG-IMU coherence per activity
    chosen = {}
    for rec in sorted(os.listdir(WRIST)):
        if not rec.endswith(".dat"):
            continue
        name = rec[:-4]
        for act in ("bike", "walk", "run"):
            if act in name and act not in chosen:
                chosen[act] = name
    for act in ("bike", "walk", "run"):
        rec = chosen.get(act)
        if rec is None:
            continue
        acc_list = []
        for p_, a_, fs_, _ in list(FM.windows(rec))[:24]:
            x_ = pl.bandpass(p_, fs_, 0.4, 8.0)
            ref = pl.accel_magnitude(a_)
            ref = ref - ref.mean()
            acc_list.append(K.ppg_motion_coherence(x_, ref, fs_, FM.NPERSEG,
                                                   FM.HOP))
        if not acc_list:
            continue
        coh = np.concatenate(acc_list, axis=0).mean(axis=0)
        ax_g.plot(freqs[:len(coh)], coh, color=FS.ACT_COLORS[act], lw=0.9,
                  label=FS.ACT_LABEL[act])
    ax_g.set_xlim(0, 8.0)
    ax_g.set_ylim(0, 1.0)
    ax_g.set_xlabel("Frequency (Hz)")
    ax_g.set_ylabel("Coherence")
    # every in-axes corner position collides with a coherence curve except
    # the flat bottom strip, which takes a single-row legend
    ax_g.legend(frameon=False, fontsize=6.5, loc="lower center", ncol=3,
                handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
                borderaxespad=0.2)
    ax_g.set_title("(c)", loc="left", fontweight="bold", x=-0.02)

    save(fig, "fig2_col.png")


# ---------------------------------------------------------------------------
# fig3_col: 6-row multisensor stream (from fig_sensor_concepts.py::fig15,
# same record s1_walk, same t0=20 s, same 12 s window)
# ---------------------------------------------------------------------------
def fig3_col():
    rec = pl.load_wrist(os.path.join(WRIST, "s1_walk"))
    fs = rec["fs"]
    t0 = int(20 * fs)
    dur = int(12 * fs)
    sl = slice(t0, t0 + dur)
    t = np.arange(dur) / fs

    # sensor name + units, wrapped to <= 3 short lines so the horizontal
    # labels (reference style) fit the left gutter of a 3.4 in column
    panels = [
        ("Chest\nECG\n(mV)", rec["ecg"][sl], FS.C["orange"]),
        ("Wrist\nPPG\n(mV)", rec["ppg"][sl], FS.C["red"]),
        ("Accel.\nwide\n(m/s$^2$)", rec["acc"][sl][:, 0], FS.C["blue"]),
        ("Accel.\nlow-noise\n(m/s$^2$)", rec["acc_ln"][sl][:, 0], FS.C["sky"]),
        ("Gyro\n(deg/s)", rec["gyro"][sl][:, 0], FS.C["purple"]),
        ("Mag\n(G)", rec["mag"][sl][:, 0], FS.C["green"]),
    ]
    fig = plt.figure(figsize=(W_IN, 3.4))
    gs = fig.add_gridspec(len(panels), 1, left=0.235, right=0.985,
                          top=0.885, bottom=0.095, hspace=0.42,
                          height_ratios=[1] * len(panels))
    axes = [fig.add_subplot(gs[i]) for i in range(len(panels))]
    for i, (ax, (name, y, col)) in enumerate(zip(axes, panels)):
        y = y - np.mean(y)
        ax.plot(t, y, color=col, lw=0.7)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=3)
        ax.tick_params(labelbottom=(i == len(panels) - 1))
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
        ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(0, dur / fs)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Simultaneous multi-sensor stream from the wrist node\n"
                      "(record s1_walk, 12 s window)", loc="center")
    save(fig, "fig3_col.png")


# ---------------------------------------------------------------------------
# fig4_col: per-record MAE by activity, CURRENT cohort
# (out/_novelty/wrist_windows_cache.json, 1664 windows, 19 records, 8 subjects)
# ---------------------------------------------------------------------------
CFG_SETS = [("ch_acc_wide", "acc wide"), ("ch_acc_ln", "acc low-noise"),
            ("ch_gyro", "gyro"), ("ch_acc_gyro", "acc+gyro"),
            ("ch_acc_gyro_mag", "acc+gyro+mag"), ("ch_full", "full suite")]
CFG_COLORS = [FS.C["black"], FS.C["sky"], FS.C["purple"], FS.C["orange"],
              FS.C["green"], FS.C["red"]]


def fig4_col():
    with open(os.path.join(NOV, "wrist_windows_cache.json"), encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1664, len(data)

    per_rec = {}
    for w in data:
        d = per_rec.setdefault(w["rec"], {"subject": w["subject"],
                                          "act": w["act"], "n": 0,
                                          "err": {k: [] for k in w["err"]}})
        d["n"] += 1
        for k, v in w["err"].items():
            d["err"][k].append(v)
    order = {"bike": 0, "run": 1, "walk": 2}
    recs = sorted(per_rec, key=lambda r: (order[per_rec[r]["act"]], r))

    print("fig4 per-activity mean MAE (mean of per-record means, bpm):")
    for k, lab in CFG_SETS:
        by_act = {}
        for r in recs:
            by_act.setdefault(per_rec[r]["act"], []).append(
                float(np.mean(per_rec[r]["err"][k])))
        line = ", ".join("%s %.2f" % (a, float(np.mean(v)))
                         for a, v in sorted(by_act.items()))
        pooled = float(np.mean([w["err"][k] for w in data]))
        print("  %-14s pooled %.2f | %s" % (k, pooled, line))

    fig = plt.figure(figsize=(W_IN, 3.5))
    ax_a = fig.add_axes([0.125, 0.60, 0.855, 0.30])
    ax_b = fig.add_axes([0.125, 0.15, 0.855, 0.195])

    # ---- (a) per-record dot plot, sorted by activity
    handles = []
    for j, (k, lab) in enumerate(CFG_SETS):
        xs = np.arange(len(recs)) + (j - 2.5) * 0.11
        ys = [float(np.mean(per_rec[r]["err"][k])) for r in recs]
        h, = ax_a.plot(xs, ys, "o", ms=2.1, color=CFG_COLORS[j], ls="none",
                       label=lab)
        handles.append(h)
    bounds = []
    prev = per_rec[recs[0]]["act"]
    for i, r in enumerate(recs):
        if per_rec[r]["act"] != prev:
            bounds.append(i)
            prev = per_rec[r]["act"]
        ax_a.annotate("n=%d" % per_rec[r]["n"], xy=(i, 128), rotation=90,
                      fontsize=6.5, ha="center", va="top", color="#555555")
    for b in bounds:
        ax_a.axvline(b - 0.5, color="#999999", lw=0.6, ls="--")
    ax_a.set_xticks(np.arange(len(recs)))
    ax_a.set_xticklabels([r.replace("_resistance", "_res") for r in recs],
                         rotation=90, fontsize=6.5)
    for i, lab in enumerate(ax_a.get_xticklabels()):
        lab.set_color(FS.ACT_COLORS[per_rec[recs[i]]["act"]])
    ax_a.set_xlim(-0.6, len(recs) - 0.4)
    ax_a.set_ylim(0, 130)
    ax_a.set_ylabel("Per-record MAE (bpm)")
    panel_letter(ax_a, -0.075, 1.045, "(a)")

    # ---- (b) activity summary: mean of per-record means per channel set
    acts = ["bike", "run", "walk"]
    x = np.arange(len(acts))
    wdt = 0.13
    for j, (k, lab) in enumerate(CFG_SETS):
        vals = [float(np.mean([np.mean(per_rec[r]["err"][k]) for r in recs
                               if per_rec[r]["act"] == a])) for a in acts]
        ax_b.bar(x + (j - 2.5) * wdt, vals, width=wdt, color=CFG_COLORS[j])
    n_win = {a: sum(per_rec[r]["n"] for r in recs if per_rec[r]["act"] == a)
             for a in acts}
    n_rec = {a: sum(1 for r in recs if per_rec[r]["act"] == a) for a in acts}
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([FS.ACT_LABEL[a] for a in acts])
    ax_b.set_ylim(0, 62)
    for i, a in enumerate(acts):
        ax_b.annotate("%d rec, %d win" % (n_rec[a], n_win[a]),
                      xy=(i, 61), ha="center", va="top", fontsize=6.5,
                      color="#555555")
    ax_b.set_ylabel("MAE (bpm)")
    panel_letter(ax_b, -0.055, 1.10, "(b)")
    # shared legend for both panels, below the column: no in-axes spot is
    # free of dots/bars at this width
    fig.legend(handles, [lab for _, lab in CFG_SETS], ncol=3, frameon=False,
               loc="lower center", handlelength=1.0, handletextpad=0.35,
               columnspacing=1.0, bbox_to_anchor=(0.5, 0.006))
    save(fig, "fig4_col.png")


# ---------------------------------------------------------------------------
# fig5_col: cross-dataset front-end sizing (make_figures.py::fig_cross_dataset
# restacked: (a) rate on top, (b) bits below, legend below the column)
# ---------------------------------------------------------------------------
def fig5_col():
    with open(os.path.join(NOV, "wrist_expanded.json"), encoding="utf-8") as f:
        wrist = json.load(f)
    with open(os.path.join(NOV, "bidmc_replication.json"), encoding="utf-8") as f:
        bidmc = json.load(f)
    C_WRIST = "#2166ac"
    C_BIDMC = "#b2182b"

    fig = plt.figure(figsize=(W_IN, 3.1))
    ax_a = fig.add_axes([0.135, 0.60, 0.845, 0.29])
    ax_b = fig.add_axes([0.135, 0.215, 0.845, 0.245])

    wr = [(16, wrist["sizing"]["sz_fs16"]), (32, wrist["sizing"]["sz_fs32"]),
          (64, wrist["sizing"]["sz_fs64"]),
          (256, wrist["sizing"]["native_256Hz_16bit"])]
    br = [(16, bidmc["sizing"]["fs16"]), (32, bidmc["sizing"]["fs32"]),
          (64, bidmc["sizing"]["fs64"]),
          (125, bidmc["sizing"]["native_125Hz_16bit"])]
    handles = []
    for pts, c, lab, mk in [(wr, C_WRIST, "Wrist (motion, KISR)", "o"),
                            (br, C_BIDMC, "BIDMC (resting, bandpass)", "^")]:
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1]["mae"] for p in pts])
        lo = np.array([p[1]["ci95_subject_cluster"][0] for p in pts])
        hi = np.array([p[1]["ci95_subject_cluster"][1] for p in pts])
        h = ax_a.errorbar(x, y, yerr=[y - lo, hi - y], marker=mk, ms=3.5,
                          lw=1.0, capsize=2.0, color=c, label=lab)
        handles.append(h)
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks([16, 32, 64, 125, 256])
    ax_a.set_xticklabels(["16", "32", "64", "125", "256"])
    ax_a.set_xlim(13, 300)
    ax_a.set_xlabel("Sampling rate (Hz)")
    ax_a.set_ylabel("HR MAE (bpm)")
    panel_letter(ax_a, -0.045, 1.04, "(a)")

    wb = [(8, wrist["sizing"]["sz_b8"]), (10, wrist["sizing"]["sz_b10"]),
          (12, wrist["sizing"]["sz_b12"]),
          (16, wrist["sizing"]["native_256Hz_16bit"])]
    bb = [(8, bidmc["sizing"]["b8"]), (10, bidmc["sizing"]["b10"]),
          (12, bidmc["sizing"]["b12"]),
          (16, bidmc["sizing"]["native_125Hz_16bit"])]
    for pts, c in [(wb, C_WRIST), (bb, C_BIDMC)]:
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1]["mae"] for p in pts])
        lo = np.array([p[1]["ci95_subject_cluster"][0] for p in pts])
        hi = np.array([p[1]["ci95_subject_cluster"][1] for p in pts])
        ax_b.errorbar(x, y, yerr=[y - lo, hi - y], marker="s", ms=3.5, lw=1.0,
                      capsize=2.0, color=c)
    ax_b.set_xticks([8, 10, 12, 16])
    ax_b.set_xlim(7, 17)
    ax_b.set_xlabel("ADC resolution (bits)")
    ax_b.set_ylabel("HR MAE (bpm)")
    panel_letter(ax_b, -0.045, 1.10, "(b)")

    # the two datasets span 6-50 bpm with wide CIs everywhere an in-axes
    # legend would fit, so the shared legend sits under the lower panel
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               handlelength=1.6, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.004))
    save(fig, "fig5_col.png")


# ---------------------------------------------------------------------------
# fig6_col: transient-aware gating (make_figures.py::fig_transient_gate
# restacked vertically; arm colors match the (a) x labels, so the redundant
# (b) legend is dropped)
# ---------------------------------------------------------------------------
def fig6_col():
    with open(os.path.join(NOV, "transient_gate.json"), encoding="utf-8") as f:
        tg = json.load(f)
    ov = tg["overall"]
    ba = tg["by_activity"]
    n_sub = tg["cohort"]["n_subjects"]
    n_win = tg["cohort"]["n_windows"]
    arms = ["kisr", "gate_soft", "gate_bp"]
    labels = ["KISR", "KISR +\nsoft gate", "KISR +\nBP gate"]
    C_ARMS = ["#4d4d4d", "#2166ac", "#b2182b"]

    fig = plt.figure(figsize=(W_IN, 2.85))
    ax_a = fig.add_axes([0.135, 0.615, 0.845, 0.285])
    ax_b = fig.add_axes([0.135, 0.135, 0.845, 0.28])

    x = np.arange(3)
    y = [ov[a]["mae"] for a in arms]
    lo = [ov[a]["ci95_subject_cluster"][0] for a in arms]
    hi = [ov[a]["ci95_subject_cluster"][1] for a in arms]
    ax_a.bar(x, y, width=0.55, color=C_ARMS,
             yerr=[np.array(y) - lo, np.array(hi) - y], capsize=2.5,
             error_kw={"lw": 0.9})
    for i, a in enumerate(arms[1:], start=1):
        d = ov[a]["vs_kisr"]
        pv = d["p_subject_wilcoxon"]
        ptxt = f"p = {pv:.3f}" if pv is not None else "p = n/a"
        ax_a.annotate(f"$\\Delta$ = {d['delta_mae']:+.2f}\n{ptxt}",
                      xy=(i, hi[i]), xytext=(0, 2.5),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=6.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("HR MAE (bpm)")
    ax_a.set_title("Hard activities (%d subjects, %d windows)" % (n_sub, n_win),
                   loc="left", fontsize=7.0)
    ax_a.set_ylim(0, max(hi) * 1.42)
    panel_letter(ax_a, -0.045, 1.10, "(a)")

    acts = ["bike_high", "run"]
    act_labels = ["High-resistance bike", "Run"]
    w = 0.24
    for j, a in enumerate(arms):
        xs = np.arange(len(acts)) + (j - 1) * w
        ys = [ba[act][a]["mae"] for act in acts]
        los = [ba[act][a]["ci95_subject_cluster"][0] for act in acts]
        his = [ba[act][a]["ci95_subject_cluster"][1] for act in acts]
        ax_b.bar(xs, ys, width=w, color=C_ARMS[j],
                 yerr=[np.array(ys) - los, np.array(his) - ys], capsize=2.0,
                 error_kw={"lw": 0.8})
    ax_b.set_xticks(np.arange(len(acts)))
    ax_b.set_xticklabels(act_labels)
    ax_b.set_ylabel("HR MAE (bpm)")
    allhi = max(ba[act][a]["ci95_subject_cluster"][1]
                for act in acts for a in arms)
    ax_b.set_ylim(0, allhi * 1.18)
    panel_letter(ax_b, -0.045, 1.10, "(b)")
    save(fig, "fig6_col.png")


def main():
    fig1_col()
    fig2_col()
    fig3_col()
    fig4_col()
    fig5_col()
    fig6_col()


if __name__ == "__main__":
    main()
