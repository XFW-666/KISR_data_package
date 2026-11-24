"""figC_optics - column-native (3.4 in) optical front end & artifact pathways.

Re-typesetting of ORIGINAL manuscript artwork at exactly one IEEE column
(3.4 in = 2040 px at dpi 600, saved WITHOUT bbox_inches="tight", so the point
sizes below ARE the printed sizes; nothing is drawn below 6.5 pt):

  panel (a)  reflective optical front-end geometry
             source of truth: src/fig_sensor_concepts.py::fig14_optics
             (figures/fig14_optics.png).  That artwork is drawn with svgkit,
             so its geometry is re-drawn here with matplotlib patches in the
             SAME coordinate space, with the SAME shapes, colours and wording
             (skin slab / epidermis / dermis + capillary bed / subcutaneous &
             venous pooling, LED, PD, the shallow green and the deeper IR
             photon paths).  Only the physical size and the type size change.

  panel (b)  the artifact pathways into the same channel
             same source: the "(b) Artifact channels at the PD" cards of
             src/fig_sensor_concepts.py::fig14_optics - contact modulation,
             motion harmonics, venous pooling, ambient leakage and the wanted
             perfusion dV(t) term - restacked into a 3 x 2 grid above the
             photodiode-current sum bar.

  panels (c)-(f)  the measured mechanism evidence of that pathway
             source of truth: src/fig3_mechanism.py (figures/fig3_mechanism.
             png, figsize 7.0 x 4.6).  Same complete-record set of
             data/wrist, same bike/walk/run choice, same 8 s windows with a
             4 s step, same 0.4-8 Hz pre-band, same nperseg=256 / hop=64 STFT
             with 3-bin smoothing, same 0.5-6 Hz band-power correlation, same
             hardest-bike-window spectrum before/after kisr(), same motion-
             subspace dominance vs SQI scatter.  Nothing is simulated; only
             the arrangement (2 x 3 -> 2 x 2) and the type size change.

Content rules kept from the manuscript: this stays a mechanism/geometry
illustration.  No prototype claims, no measured power or battery numbers, no
MCU part numbers.
"""
import os
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import figstyle as FS          # Arial family + Okabe-Ito palette
import ppglib as pl
import kisr as K
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
from PIL import Image

WRIST = os.path.join(ROOT, "data", "wrist")

plt.rcParams.update({
    "font.size": 6.5,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 100,
    "savefig.dpi": 600,
})
DPI = 600
W_IN = 3.4
W_U = 340.0            # canvas width  in 1/100 in
H_U = 456.0            # canvas height in 1/100 in  ->  3.4 x 4.56 in

FS_NOTE = 6.5          # smallest face used anywhere in this figure
FS_BOX = 6.5
FS_TITLE = 7.0

# ---- palette taken verbatim from src/fig_sensor_concepts.py::fig14_optics --
SKIN = ("#FBD5B5", "#D97706")
EPI = ("#FDE68A", "#F59E0B")
DERM = ("#FECACA", "#EF4444")
SUBQ = ("#E7D5B8", "#A16207")
LED_F, LED_E = "#86EFAC", "#15803D"
PD_F, PD_E = "#BFDBFE", "#2563EB"
GREEN_PATH = "#15803D"
IR_PATH = "#C2410C"
RBC = "#DC2626"
INK = "#1a1a1a"

TEXTS = []             # (artist, note) for the renderer-based QC pass
FITS = []              # (text artist, patch artist) box-containment checks


def note(t, s):
    TEXTS.append((t, s.replace("\n", " / ")[:28]))
    return t


def ax_at(fig, x, y, w, h):
    """Add axes from a top-origin 1/100 in layout box."""
    return fig.add_axes([x / W_U, (H_U - y - h) / H_U, w / W_U, h / H_U])


def head(fig, x, y, letter, title):
    """Bold 7 pt panel letter + 7 pt panel title (top-origin units)."""
    note(fig.text(x / W_U, 1 - y / H_U, letter, fontsize=FS_TITLE,
                  fontweight="bold", ha="left", va="top"), letter)
    note(fig.text((x + 15) / W_U, 1 - y / H_U, title, fontsize=FS_TITLE,
                  ha="left", va="top"), title[:24])


def card(ax, x, y, w, h, fc, ec, r=8, lw=1.0, z=2):
    """svgkit-style rounded card, in the axes' own data coordinates."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0,rounding_size=%g" % r,
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=DPI, facecolor="white")
    im = Image.open(path)
    assert im.width == int(W_IN * DPI), (name, im.width)
    im.convert("RGB").save(path)
    plt.close(fig)
    print("saved", path, im.size, "-> %.2f x %.2f in"
          % (im.width / DPI, im.height / DPI))
    return path


# ===========================================================================
# (a) reflective optical front-end geometry - re-typeset of fig14_optics
# ===========================================================================
def optics_panel(fig):
    ax = ax_at(fig, 6, 16, 328, 106)
    # SVG user space of fig14_optics, top-origin (y grows downwards)
    ax.set_xlim(30, 486)
    ax.set_ylim(352, 44)          # top-origin, like the source SVG
    ax.set_aspect("auto")
    ax.axis("off")

    def put(x, y, s, fs=FS_NOTE, color=INK, ha="center", bold=False, z=5):
        ha = {"start": "left", "end": "right"}.get(ha, ha)
        return note(ax.text(x, y, s, fontsize=fs, color=color, ha=ha,
                            va="center", zorder=z,
                            fontweight="bold" if bold else "normal"), s[:26])

    # skin slab and the three tissue layers
    card(ax, 60, 70, 360, 282, SKIN[0], SKIN[1], r=14, lw=1.2)
    put(240, 90, "Skin cross-section (reflective PPG)", FS_NOTE, "#92400E",
        bold=True)
    card(ax, 72, 104, 336, 34, EPI[0], EPI[1], r=6, lw=0.9)
    put(240, 121, "Epidermis (scattering layer)", FS_NOTE, "#92400E")
    card(ax, 72, 142, 336, 144, DERM[0], DERM[1], r=6, lw=0.9)
    put(150, 156, "Dermis: arteriolar / capillary bed", FS_NOTE, "#9F1239",
        ha="start")
    for vx, vy in [(112, 202), (185, 236), (300, 236), (376, 202)]:
        ax.add_patch(Ellipse((vx, vy), 40, 34, fc=RBC, ec="none", alpha=0.8,
                             zorder=3))
        put(vx, vy, "RBC", FS_NOTE, "#ffffff", z=6)
    put(240, 272, "blood-volume pulsation dV(t)  <--  cardiac source",
        FS_NOTE, "#9F1239", bold=True)
    card(ax, 72, 292, 336, 52, SUBQ[0], SUBQ[1], r=6, lw=0.9)
    put(240, 318, "Subcutaneous tissue / venous pooling (slow drift)",
        FS_NOTE, "#78350F")

    # light source and detector, outside the slab
    card(ax, 34, 156, 46, 34, LED_F, LED_E, r=5, lw=1.0)
    put(57, 173, "LED", FS_NOTE, "#14532D", bold=True)
    card(ax, 434, 156, 46, 34, PD_F, PD_E, r=5, lw=1.0)
    put(457, 173, "PD", FS_NOTE, "#1E3A8A", bold=True)

    # photon paths: the two cubic Beziers of the source artwork
    def bez(p0, p1, p2, p3, n=140):
        t = np.linspace(0, 1, n)[:, None]
        pts = ((1 - t) ** 3 * np.asarray(p0, float)
               + 3 * (1 - t) ** 2 * t * np.asarray(p1, float)
               + 3 * (1 - t) * t ** 2 * np.asarray(p2, float)
               + t ** 3 * np.asarray(p3, float))
        return pts[:, 0], pts[:, 1]

    for segs, color in [
        ([((80, 178), (140, 168), (190, 186), (250, 200)),
          ((250, 200), (310, 214), (370, 196), (434, 180))], GREEN_PATH),
        ([((80, 192), (150, 225), (200, 250), (255, 250)),
          ((255, 250), (310, 250), (380, 215), (434, 192))], IR_PATH),
    ]:
        for s0, s1, s2, s3 in segs:
            xs, ys = bez(s0, s1, s2, s3)
            ax.plot(xs, ys, color=color, lw=1.0, zorder=4,
                    solid_capstyle="round")

    # wavelength key - the same two entries as the source artwork
    ax.plot([60, 88], [54, 54], color=GREEN_PATH, lw=1.1, zorder=4)
    put(92, 54, "green 525 nm: shallow (HR)", FS_NOTE, GREEN_PATH, ha="start")
    ax.plot([282, 310], [54, 54], color=IR_PATH, lw=1.1, zorder=4)
    put(314, 54, "IR 940 nm: deep (SpO2)", FS_NOTE, IR_PATH, ha="start")


# ===========================================================================
# (b) artifact pathways at the photodiode - re-typeset fig14_optics cards
# ===========================================================================
ARTS = [
    ("1. Contact modulation",
     "strap bounce changes LED-PD coupling -> step + spike noise",
     "#FEE2E2", "#B91C1C"),
    ("2. Motion harmonics",
     "limb periodicity mixes into the cardiac band (0.7-3.2 Hz)",
     "#E0E7FF", "#3730A3"),
    ("3. Venous pooling",
     "non-pulsatile blood shift -> baseline wander",
     "#FECACA", "#9F1239"),
    ("4. Ambient leakage",
     "sunlight / lamps couple through the gasket into the PD",
     "#FEF3C7", "#92400E"),
    ("Wanted: perfusion dV",
     "arterial pulsation, the only component we want to keep",
     "#DCFCE7", "#166534"),
]


def pathway_panel(fig):
    cv = ax_at(fig, 2, 140, 336, 140)
    cv.set_xlim(0, 336)
    cv.set_ylim(140, 0)
    cv.set_aspect("auto")
    cv.axis("off")

    bw = (336 - 2 * 4) / 3.0
    bh = 50.0

    def box(x, y, title, body, fc, ec):
        p = card(cv, x, y, bw, bh, fc, ec, r=5, lw=0.8)
        t = cv.text(x + 3, y + 10, title, fontsize=FS_BOX, color=ec,
                    ha="left", va="center", fontweight="bold", zorder=5)
        note(t, title[:26])
        FITS.append((t, p))
        for i, ln in enumerate(textwrap.wrap(body, width=22,
                                             break_long_words=False)):
            tt = cv.text(x + 3, y + 21 + i * 9.5, ln, fontsize=FS_NOTE,
                         color="#475467", ha="left", va="center", zorder=5)
            note(tt, ln[:26])
            FITS.append((tt, p))

    xs = [i * (bw + 4) for i in range(3)]
    for i, (t, b, f, e) in enumerate(ARTS[:3]):
        box(xs[i], 2, t, b, f, e)
    for i, (t, b, f, e) in enumerate(ARTS[3:]):
        box(xs[i], 54, t, b, f, e)

    # what the photodiode integrates
    bar = card(cv, 0, 116, 336, 22, "#F2F4F7", "#444444", r=5, lw=0.9)
    t = cv.text(168, 127, "PD current = perfusion dV(t) + contact + harmonics"
                          " + pooling + ambient",
                fontsize=FS_NOTE, color=INK, ha="center", va="center",
                zorder=5)
    note(t, "PD current")
    FITS.append((t, bar))
    for x, col in ((52, "#444444"), (168, "#166534"), (284, "#444444")):
        cv.annotate("", xy=(x, 116), xytext=(x, 105), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=col,
                                    shrinkA=0, shrinkB=0, mutation_scale=6))


# ===========================================================================
# (c)-(f) the four mechanism panels of src/fig3_mechanism.py, restacked
# ===========================================================================
WIN_S, STEP_S = 8.0, 4.0


def complete_records():
    from fetch import expected_dat_size
    out = []
    for f in sorted(os.listdir(WRIST)):
        if f.endswith(".dat"):
            hea = os.path.join(WRIST, f[:-4] + ".hea")
            if os.path.exists(hea):
                exp = expected_dat_size(hea)
                if exp is not None and os.path.getsize(
                        os.path.join(WRIST, f)) == exp:
                    out.append(f[:-4])
    return out


def data_panels(fig, y0=298.0):
    recs = complete_records()
    recs.sort(key=lambda r: (0 if "run" in r else 1 if "walk" in r else 2, r))
    chosen = {}
    for act in ("bike", "walk", "run"):
        for r in recs:
            if act in r and act not in chosen:
                chosen[act] = r
    print("figC records:", chosen, "| complete records:", len(recs))

    cells = [
        (44, y0, "(c) PPG-IMU coherence", "Coherence", ""),
        (206, y0, "(d) Band power vs motion", "Power corr.",
         ""),
        (44, y0 + 74, "(e) Spectrum, KISR effect", "Norm. PSD",
         "Frequency (Hz)"),
        (206, y0 + 74, "(f) Dominance vs SQI", "Dom. share",
         "PPG signal quality index (SQI)"),
    ]
    CW, CH = 120.0, 48.0
    axes = [ax_at(fig, x, y, CW, CH) for (x, y, _t, _l, _x) in cells]
    for ax, (_x, _y, title, yl, _xl) in zip(axes, cells):
        ax.set_title(title, fontsize=FS_TITLE, pad=2.5, loc="left")
        ax.tick_params(length=1.8, width=0.5, pad=1.2)
        ax.set_ylabel(yl, labelpad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_linewidth(0.5)

    # ---- (c) coherence spectra, pooled per activity -----------------------
    ax = axes[0]
    for act, rec in chosen.items():
        r = pl.load_wrist(os.path.join(WRIST, rec))
        fs = r["fs"]
        coh_list = []
        wl = int(WIN_S * fs)
        for s in range(0, len(r["ppg"]) - wl, int(STEP_S * fs)):
            e = s + wl
            x = pl.bandpass(r["ppg"][s:e], fs, 0.4, 8.0)
            ref = pl.accel_magnitude(r["acc"][s:e])
            ref = ref - ref.mean()
            Zp, _ = K.stft(x, 256, 64)
            Zr, _ = K.stft(ref, 256, 64)
            n = min(Zp.shape[0], Zr.shape[0])
            f = np.fft.rfftfreq(256, d=1.0 / fs)
            ker = np.ones(3) / 3.0
            Sxx = np.apply_along_axis(
                lambda v: np.convolve(v, ker, "same"), 0, np.abs(Zp[:n]) ** 2)
            Syy = np.apply_along_axis(
                lambda v: np.convolve(v, ker, "same"), 0, np.abs(Zr[:n]) ** 2)
            Sxy = np.apply_along_axis(
                lambda v: np.convolve(v, ker, "same"), 0,
                np.conj(Zp[:n]) * Zr[:n])
            den = Sxx * Syy
            c = np.zeros_like(Sxx.real)
            nz = den > 1e-18
            c[nz] = np.abs(Sxy[nz]) ** 2 / den[nz]
            coh_list.append(np.clip(c, 0, 1))
        coh = np.mean(coh_list, axis=0)
        band = (f >= 0.4) & (f <= 6.0)
        ax.plot(f[band], coh.mean(axis=0)[band], color=FS.ACT_COLORS[act],
                lw=0.8, label=FS.ACT_LABEL[act])
    ax.axvspan(0.7, 3.2, color=FS.C["yellow"], alpha=0.15, lw=0)
    ax.set_xlim(0.4, 6.0)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(0, 2.6 * max(y1, 0.05))   # empty top band keeps legend off data
    ax.set_xticks([1, 3, 5])
    ax.set_yticks([0.0, 0.5, 1.0])
    h, l = ax.get_legend_handles_labels()
    ax.add_artist(ax.legend(h[:2], l[:2], frameon=False, handlelength=1.0,
                            handletextpad=0.3, borderaxespad=0.3,
                            labelspacing=0.2, fontsize=FS_NOTE,
                            loc="upper left"))
    ax.legend([h[2]], [l[2]], frameon=False, handlelength=1.0,
              handletextpad=0.3, borderaxespad=0.3, fontsize=FS_NOTE,
              loc="upper right")

    # ---- (d) motion intensity vs PPG band-power correlation ---------------
    ax = axes[1]
    lv, corr = [], []
    for rec in recs:
        r = pl.load_wrist(os.path.join(WRIST, rec))
        fs = r["fs"]
        wl = int(WIN_S * fs)
        a_full = pl.accel_magnitude(r["acc"])
        a_std = a_full.std() + 1e-12
        for s in range(0, len(r["ppg"]) - wl, int(STEP_S * fs)):
            e = s + wl
            x = pl.bandpass(r["ppg"][s:e], fs, 0.4, 8.0)
            am = pl.accel_magnitude(r["acc"][s:e])
            ref = pl.bandpass(am - am.mean(), fs, 0.4, 8.0)
            fx, Px = pl.welch(x, fs)
            fr, Pr = pl.welch(ref, fs)
            bx = (fx >= 0.5) & (fx <= 6.0)
            br = (fr >= 0.5) & (fr <= 6.0)
            cc = np.corrcoef(Px[bx], Pr[br])[0, 1]
            if np.isfinite(cc):
                lv.append(float(np.clip(a_full[s:e].std() / (3 * a_std), 0, 1)))
                corr.append(cc)
    lv = np.asarray(lv)
    corr = np.asarray(corr)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(lv, bins) - 1
    mid, mean_c, err_c = [], [], []
    for b in range(10):
        sel = idx == b
        if sel.sum() >= 3:
            mid.append(0.5 * (bins[b] + bins[b + 1]))
            mean_c.append(corr[sel].mean())
            err_c.append(corr[sel].std() / np.sqrt(sel.sum()))
    ax.errorbar(mid, mean_c, yerr=err_c, fmt="o-", color=FS.C["blue"],
                capsize=1.5, lw=0.8, ms=2.5)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])

    # ---- (e) spectrum before / after KISR on the hardest window -----------
    ax = axes[2]
    hard = chosen.get("bike") or recs[0]
    r = pl.load_wrist(os.path.join(WRIST, hard))
    fs = r["fs"]
    wl = int(8 * fs)
    s = (len(r["ppg"]) - wl) // 2
    x = pl.bandpass(r["ppg"][s:s + wl], fs, 0.4, 8.0)
    imu = np.concatenate([r["acc"][s:s + wl], r["gyro"][s:s + wl]], axis=1)
    y = K.kisr(x, imu, fs)
    f, P0 = pl.welch(x, fs)
    _, P1 = pl.welch(y, fs)
    band = (f >= 0.4) & (f <= 6.0)
    ax.plot(f[band], P0[band] / P0[band].max(), color=FS.C["black"], lw=0.8,
            label="Before")
    ax.plot(f[band], P1[band] / P0[band].max(), color=FS.C["red"], lw=0.8,
            label="After KISR")
    ax.set_xlim(0.4, 6.0)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 3, 5])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.legend(frameon=False, handlelength=1.0, handletextpad=0.3,
              borderaxespad=0.3, labelspacing=0.2, fontsize=FS_NOTE,
              loc="upper right")

    # ---- (f) motion-subspace dominance vs window SQI ----------------------
    ax = axes[3]
    dom_all, sqi_all, act_all = [], [], []
    for rec in recs:
        act = "run" if "run" in rec else ("walk" if "walk" in rec else "bike")
        r = pl.load_wrist(os.path.join(WRIST, rec))
        fs = r["fs"]
        wl = int(WIN_S * fs)
        for s in range(0, len(r["ppg"]) - wl, int(STEP_S * fs)):
            e = s + wl
            x = pl.bandpass(r["ppg"][s:e], fs, 0.4, 8.0)
            imu = np.concatenate([r["acc"][s:e], r["gyro"][s:e]], axis=1)
            _, d = K.kisr(x, imu, fs, return_diag=True)
            dom_all.append(float(d["dom"].mean()))
            sqi_all.append(float(pl.ppg_sqi(r["ppg"][s:e], fs)))
            act_all.append(act)
    dom_all = np.asarray(dom_all)
    sqi_all = np.asarray(sqi_all)
    for act in sorted(set(act_all)):
        sel = np.asarray(act_all) == act
        ax.scatter(sqi_all[sel], dom_all[sel], s=3, alpha=0.55,
                   color=FS.ACT_COLORS[act], label=FS.ACT_LABEL[act],
                   edgecolors="none", linewidths=0)
    r_p = np.corrcoef(sqi_all, dom_all)[0, 1]
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.35 * (y1 - y0), y1 + 0.90 * (y1 - y0))
    h, l = ax.get_legend_handles_labels()
    ax.add_artist(ax.legend(h[:2], l[:2], frameon=False, ncols=2,
                            handlelength=0.8, handletextpad=0.25,
                            columnspacing=0.7, borderaxespad=0.2,
                            fontsize=FS_NOTE, loc="upper right"))
    ax.legend([h[2]], [l[2]], frameon=False, handlelength=0.8,
              handletextpad=0.25, borderaxespad=0.2, fontsize=FS_NOTE,
              loc="lower right")
    ax.set_yticks([0.0, 0.5, 1.0])
    print("figC panel (f): Pearson r = %.2f, n = %d windows"
          % (r_p, len(sqi_all)))

    # x tick labels / axis labels only on the bottom row
    for i in (0, 1):
        axes[i].set_xticklabels([])
        axes[i].tick_params(labelbottom=False)
    for i in (2, 3):
        axes[i].set_xlabel(cells[i][4])
    return r_p, len(sqi_all)


# ===========================================================================
def qc(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ext = [(n, t.get_window_extent(renderer=r)) for t, n in TEXTS]
    hits = []
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            a, b = ext[i][1], ext[j][1]
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > 1.5 and
                    min(a.y1, b.y1) - max(a.y0, b.y0) > 1.5):
                hits.append((ext[i][0], ext[j][0]))
    wpx = fig.get_size_inches()[0] * fig.dpi
    hpx = fig.get_size_inches()[1] * fig.dpi
    off = [n for n, bb in ext
           if bb.x0 < -0.5 or bb.x1 > wpx + 0.5 or bb.y0 < -0.5
           or bb.y1 > hpx + 0.5]
    for h in hits:
        print("C OVERLAP:", h)
    for o in off:
        print("C OFF-CANVAS:", o)
    bad_fit = []
    for t, p in FITS:
        tb = t.get_window_extent(renderer=r)
        pb = p.get_window_extent(renderer=r)
        pad = 1.0
        if not (tb.x0 >= pb.x0 - pad and tb.x1 <= pb.x1 + pad
                and tb.y0 >= pb.y0 - pad and tb.y1 <= pb.y1 + pad):
            bad_fit.append("%s: dx=%.1f dy=%.1f px"
                           % (t.get_text()[:24],
                              max(pb.x0 - tb.x0, tb.x1 - pb.x1),
                              max(pb.y0 - tb.y0, tb.y1 - pb.y1),))
    for b in bad_fit:
        print("C BOX-OVERFLOW:", b)
    assert not hits and not off and not bad_fit, (hits, off, bad_fit)


def main():
    global TEXTS, FITS, H_U
    TEXTS, FITS = [], []
    ua, ub = os.path.join(HERE, "fig2a_user.png"), os.path.join(HERE, "fig2b_user.png")
    if os.path.exists(ua) and os.path.exists(ub):
        # author-supplied artwork for (a)/(b); (c)-(f) stay renderer-native
        ia = Image.open(ua).convert("RGB")
        ib = Image.open(ub).convert("RGB")
        ha = W_U * ia.height / ia.width
        hb = W_U * ib.height / ib.width
        y_a = 16.0
        y_b = y_a + ha + 16.0
        y_data = y_b + hb + 16.0
        H_U = y_data + 160.0
        fig = plt.figure(figsize=(W_IN, H_U / 100.0))
        head(fig, 4, 4, "(a)", "Reflective optical front end: LED, skin, PD")
        ax = fig.add_axes([0, (H_U - y_a - ha) / H_U, 1, ha / H_U])
        ax.imshow(ia)
        ax.set_axis_off()
        head(fig, 4, y_b - 12, "(b)", "Motion-artifact pathways into the channel")
        ax = fig.add_axes([0, (H_U - y_b - hb) / H_U, 1, hb / H_U])
        ax.imshow(ib)
        ax.set_axis_off()
        data_panels(fig, y_data)
        qc(fig)
        save(fig, "figC_optical_front_end.png")
        return
    fig = plt.figure(figsize=(W_IN, H_U / 100.0))
    head(fig, 4, 4, "(a)", "Reflective optical front end: LED, skin, PD")
    optics_panel(fig)
    head(fig, 4, 127, "(b)", "Motion-artifact pathways into the channel")
    pathway_panel(fig)
    data_panels(fig)
    qc(fig)
    save(fig, "figC_optical_front_end.png")


if __name__ == "__main__":
    main()
