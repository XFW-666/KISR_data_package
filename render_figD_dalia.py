"""figD_dalia.png - column-native (3.4 in, 2040 px @ 600 dpi) PPG-DaLiA
channel-law replication figure, mirroring render_col.py::fig4_col style.

Panels
  (a) per-subject MAE dots, 15 subjects x 3 inertial channel sets;
  (b) per-activity MAE bars (walking / cycling / stairs) x 3 sets;
  (c) forest plot of delta-MAE vs the full inertial union (wrist+chest ACC)
      with subject-clustered bootstrap 95% CIs, the pre-specified +/-2 bpm
      TOST equivalence band, Wilcoxon p and the TOST verdict per set.

The public PPG-DaLiA release has no gyro/mag and no running activity, so the
channel sets are wrist ACC / chest ACC / their union and the third exercise
activity is stairs (see dalia_channel_law.json notes).

Data: out/_novelty/dalia_channel_law.json (no re-computation).
Arial >= 6.5 pt everywhere; exact 2040 px width asserted on save.
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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

NOV = os.path.join(ROOT, "out", "_novelty")

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

CFG_SETS = [("ch_acc", "wrist acc"), ("ch_acc_chest", "chest acc"),
            ("ch_acc_both", "wrist+chest acc")]
CFG_COLORS = [FS.C["black"], FS.C["sky"], FS.C["red"]]
ACTS = ["walk", "bike", "stairs"]
ACT_LABEL = {"walk": "Walking", "bike": "Cycling", "stairs": "Stairs"}


def panel_letter(ax, x, y, s, va="center"):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7.0, fontweight="bold",
            ha="center", va=va, clip_on=False)


def save(fig, name):
    from PIL import Image
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=DPI, facecolor="white")
    im = Image.open(path)
    assert im.width == int(W_IN * DPI), (name, im.width)
    im.convert("RGB").save(path)
    plt.close(fig)
    print("saved", path, im.size, "-> %.2f x %.2f in" % (im.width / DPI, im.height / DPI))


def main():
    with open(os.path.join(NOV, "dalia_channel_law.json"), encoding="utf-8") as f:
        D = json.load(f)
    subjects = D["cohort"]["subjects"]
    n_win = D["cohort"]["n_windows"]
    n_sub = D["cohort"]["n_subjects"]
    psm = D["per_subject_mae"]
    pam = D["per_activity_mae"]
    ca = D["channel_ablation"]
    wpa = D["cohort"]["windows_per_activity"]

    fig = plt.figure(figsize=(W_IN, 3.05))
    ax_b = fig.add_axes([0.115, 0.50, 0.865, 0.34])
    # narrow forest axes leave a right-hand column for the TOST verdict text
    ax_c = fig.add_axes([0.160, 0.14, 0.560, 0.24])
    handles = []

    # ---- (a) per-activity bars
    x = np.arange(len(ACTS))
    wdt = 0.20
    bmax = 0.0
    for j, (k, lab) in enumerate(CFG_SETS):
        vals = [pam[k][a]["mae"] for a in ACTS]
        bc = ax_b.bar(x + (j - 1) * wdt, vals, width=wdt, color=CFG_COLORS[j])
        handles.append(bc[0])
        bmax = max(bmax, max(vals))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([ACT_LABEL[a] for a in ACTS])
    ax_b.set_ylim(0, bmax * 1.52)
    for i, a in enumerate(ACTS):
        ax_b.annotate("%d win" % wpa[a], xy=(i, bmax * 1.20), ha="center",
                      va="top", fontsize=6.5, color="#555555")
    ax_b.set_ylabel("MAE (bpm)")
    ax_b.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
    ax_b.set_title("PPG-DaLiA (%d subjects, %d windows; walking/cycling/stairs)"
                   % (n_sub, n_win), loc="left", fontsize=7.0, x=0.0)
    panel_letter(ax_b, -0.075, 1.10, "(a)")

    # ---- (c) forest: delta vs full union + TOST band
    sets4 = [(k, lab) for k, lab in CFG_SETS if k != "ch_acc_both"]
    ypos = np.arange(len(sets4))[::-1]
    margin = D["protocol"]["tost_margin_bpm"]
    ax_c.axvspan(-margin, margin, color="#d8efe4", alpha=0.7, lw=0, zorder=0)
    ax_c.axvline(0, color="#666666", lw=0.7, zorder=1)
    lo_all, hi_all = 0.0, 0.0
    for y, (k, lab) in zip(ypos, sets4):
        d = ca[k]["vs_full"]
        t = ca[k]["tost"]
        m = d["delta_mae"]
        lo, hi = d["ci95_subject_cluster"]
        j = [kk for kk, _ in CFG_SETS].index(k)
        ax_c.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", ms=2.6,
                      color=CFG_COLORS[j], lw=1.0, capsize=2.0, zorder=3)
        lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
        pt = max(t["p_lower"], t["p_upper"])
        ptxt = "TOST p<0.001" if pt < 0.001 else "TOST p=%.3f" % pt
        verdict = "equiv." if t.get("equivalent") else "not equiv."
        ax_c.annotate("%s\n%s" % (verdict, ptxt),
                      xy=(1.04, y), xycoords=("axes fraction", "data"),
                      fontsize=6.5, ha="left", va="center",
                      color="#1a7a4a" if t.get("equivalent") else "#555555")
    pad = max(0.5, (hi_all - lo_all) * 0.12)
    # keep the +/-2 bpm equivalence band edges inside the view
    ax_c.set_xlim(min(lo_all - pad, -2.4), max(hi_all + pad, 2.4))
    ax_c.set_ylim(-0.6, len(sets4) - 0.4)
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([lab for _, lab in sets4])
    ax_c.set_xlabel(r"$\Delta$MAE vs union (bpm); $\pm$2 bpm TOST band shaded")
    ax_c.set_xticks([-2, -1, 0, 1, 2])
    panel_letter(ax_c, -0.105, 1.10, "(b)")

    ax_b.legend(handles, [lab for _, lab in CFG_SETS], ncol=3, frameon=False,
                loc="upper center", handlelength=1.0, handletextpad=0.35,
                columnspacing=1.2, fontsize=6.5, bbox_to_anchor=(0.5, 1.02))
    save(fig, "figD_dalia.png")


if __name__ == "__main__":
    main()