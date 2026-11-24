"""Column-native (3.4 in) re-renders of the two ILLUSTRATION figures.

Brings back the sensor/device artwork that the compression pass removed:

  figA_sensor_chain.png  - sensing-chain / wrist-node overview diagram
             source of truth: src/fig_system_diagram.py (the file behind the
             old composite figures/g1_methods.png), with the KISR chain part
             cross-checked against src/fig4_kisr_blocks.py.  The four panels
             (node/channel map + wearing position, shared sample clock,
             processing chain) are stacked vertically instead of the original
             2 x 2 side-by-side arrangement, and the block wording was kept to
             what the manuscript actually claims: the recorded channel set of
             the public Wrist database, its 256 Hz / 16-bit grid and the
             16-256 Hz, 8-16 bit sizing axes this study sweeps.  No MCU part,
             battery, LED wavelength, BLE output or power number is drawn,
             because the paper states that no hardware prototype or measured
             power data is part of the study.

  figB_subspace.png      - motion-subspace projection process figure
             source of truth: src/fig5_subspace.py (the file behind the old
             composite figures/g2_subspace.png).  Same record
             (s1_high_resistance_bike), same hardest-motion 8 s window
             selection, same nperseg=256 / hop=64, same six real-data panels
             from kisr(..., return_diag=True); only restacked 3 x 2 -> 6 x 1.

Print-native typesetting: at 3.4 in width the point sizes set here ARE the
printed sizes, so nothing is drawn below 6.5 pt.  Both PNGs are saved with
fig.savefig(path, dpi=600) and NO bbox_inches="tight", so each file is exactly
3.4*600 = 2040 px wide.  Re-running regenerates identical files.
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import figstyle as FS          # Arial family + Okabe-Ito palette
import ppglib as pl
import kisr as K
import matplotlib.pyplot as plt

WRIST = os.path.join(ROOT, "data", "wrist")
D3 = os.path.join(ROOT, "figures", "_3d")

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

FS_NOTE = 6.5      # smallest face used anywhere in these two figures
FS_BOX = 6.5       # block-diagram text
FS_TITLE = 7.0     # panel titles / axis titles

# diagram ink (identical values to src/fig_system_diagram.py)
LINE = "#333333"
EDGE = "#444444"
FILL = "#ffffff"
FILL_ALT = "#f2f4f7"
ACC = "#1f4e79"


def save(fig, name):
    """Save at exact 3.4 in width, flatten to RGB white background."""
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
# figA - sensing chain / wrist node overview
# ===========================================================================
def remove_dangling_leader(im):
    """The source artwork carries a call-out leader whose label lies outside
    the raster (it enters at the top edge and ends on the node face).  Erase
    only that thin line and its two dash remnants by blending across it; the
    arm, node outline and the complete 'forearm' call-out are untouched.
    """
    a = np.asarray(im.convert("RGB")).astype(np.float64)
    h, w, _ = a.shape
    mask = np.zeros((h, w), bool)
    for y in range(0, 72):                      # main diagonal leader
        xc = int(round(850 + (907 - 850) * y / 65.0))
        mask[y, max(0, xc - 4):xc + 5] = True
    mask[0:36, 948:966] = True                  # dash remnants on the face
    mask[8:36, 874:896] = True
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        xl, xr = x - 12, x + 12
        if xl < 0 or xr >= w:
            continue
        cl, cr = a[y, xl], a[y, xr]
        if min(cl.mean(), cr.mean()) > 150:     # white bg or plain node face
            a[y, x] = 0.5 * (cl + cr)
    out = Image.fromarray(a.astype(np.uint8))
    print("wrist_worn: removed dangling top leader (%d px inpainted)" % len(ys))
    return out


def trim_white_margin(im, tol=6, pad=4):
    """Remove ONLY the near-white outer margin of a raster artwork.

    The previous version of this script hard-cropped wrist_worn.png to
    (700, 60, 1400, 545), which cut off the forearm at the left, the hand at
    the right and the top of the node.  This keeps every pixel of the arm /
    wrist / node silhouette (and its baked-in call-out text): it trims the
    empty border so that the drawing can be shown at the full panel width.
    """
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    ink = (np.abs(a - 255).max(axis=2) > tol)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    x0 = max(0, int(cols[0]) - pad)
    y0 = max(0, int(rows[0]) - pad)
    x1 = min(im.width, int(cols[-1]) + 1 + pad)
    y1 = min(im.height, int(rows[-1]) + 1 + pad)
    print("wrist_worn: full %dx%d -> margin-trimmed %dx%d (nothing cropped)"
          % (im.width, im.height, x1 - x0, y1 - y0))
    return im.crop((x0, y0, x1, y1))


# canvas: 100 units = 1 in, top-origin coordinates (y grows downwards)
AW = 340.0
AH = 470.0


class ChainCanvas:
    """Block-diagram canvas with top-origin layout helpers."""

    def __init__(self, fig, ax, h):
        self.ax = ax
        self.h = h
        self.texts = []      # (artist, note)
        self.fits = []       # (text artist, rect artist)

    def Y(self, v):
        return self.h - v

    def text(self, x, y, s, fs=FS_NOTE, ha="left", va="top", color="#1a1a1a",
             bold=False, italic=False, ls=1.45, z=4):
        t = self.ax.text(x, self.Y(y), s, fontsize=fs, ha=ha, va=va,
                         color=color, zorder=z, linespacing=ls,
                         fontweight="bold" if bold else "normal",
                         style="italic" if italic else "normal")
        self.texts.append((t, s.split("\n")[0][:24]))
        return t

    def box(self, x, y, w, h, lines, fs=FS_BOX, fill=FILL, edge=EDGE, lw=0.8,
            bold_first=False):
        """(x, y) = top-left corner in top-origin units; `lines` is a list."""
        ax = self.ax
        s = "\n".join(lines)
        rect = plt.Rectangle((x, self.Y(y) - h), w, h, fc=fill, ec=edge,
                             lw=lw, zorder=2)
        ax.add_patch(rect)
        t = ax.text(x + w / 2, self.Y(y) - h / 2, s, ha="center", va="center",
                    fontsize=fs, color="#1a1a1a", zorder=3, linespacing=1.25,
                    fontweight="normal")
        self.texts.append((t, lines[0][:24]))
        self.fits.append((t, rect))
        return rect

    def seg(self, pts, lw=0.7, color=LINE, z=1):
        xs = [p[0] for p in pts]
        ys = [self.Y(p[1]) for p in pts]
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=z,
                     solid_capstyle="round")

    def arrow(self, p0, p1, color=LINE, lw=0.7, z=1):
        self.ax.annotate("", xy=(p1[0], self.Y(p1[1])),
                         xytext=(p0[0], self.Y(p0[1])), zorder=z,
                         arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                         shrinkA=0, shrinkB=0,
                                         mutation_scale=6))

    def route(self, pts, color=LINE, lw=0.7):
        """Orthogonal polyline, arrow head on the last segment."""
        self.seg(pts[:-1], lw=lw, color=color)
        self.arrow(pts[-2], pts[-1], color=color, lw=lw)


def qc_text(fig, canvas):
    """Renderer-based label QC: no text-text collisions, text inside boxes."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ext = []
    for t, note in canvas.texts:
        bb = t.get_window_extent(renderer=r)
        ext.append((note, bb))
    hits = []
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            a, b = ext[i][1], ext[j][1]
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > 1.5 and
                    min(a.y1, b.y1) - max(a.y0, b.y0) > 1.5):
                hits.append((ext[i][0], ext[j][0]))
    bad_fit = []
    for t, rect in canvas.fits:
        tb = t.get_window_extent(renderer=r)
        rb = rect.get_window_extent(renderer=r)
        pad = 1.0
        if not (tb.x0 >= rb.x0 - pad and tb.x1 <= rb.x1 + pad
                and tb.y0 >= rb.y0 - pad and tb.y1 <= rb.y1 + pad):
            bad_fit.append("%s: dx=%.1f dy=%.1f px (box %.0fx%.0f)"
                           % (t.get_text().split("\n")[0][:24],
                              max(rb.x0 - tb.x0, tb.x1 - rb.x1),
                              max(rb.y0 - tb.y0, tb.y1 - rb.y1),
                              rb.width, rb.height))
    # anything drawn outside the canvas is clipped
    out = []
    for note, bb in ext:
        if bb.x0 < -0.5 or bb.x1 > fig.get_size_inches()[0] * fig.dpi + 0.5 \
                or bb.y0 < -0.5 or bb.y1 > fig.get_size_inches()[1] * fig.dpi + 0.5:
            out.append(note)
    return hits, bad_fit, out


def figA():
    fig = plt.figure(figsize=(W_IN, AH / 100.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, AW)
    ax.set_ylim(0, AH)
    ax.axis("off")
    c = ChainCanvas(fig, ax, AH)

    # ---------------- (a) wrist node: channels + wearing position ----------
    c.text(4, 8, "(a)", fs=FS_TITLE, bold=True, va="top")
    c.text(20, 8, "Wrist node: sensed channels and wearing position",
           fs=FS_TITLE, va="top")

    worn = trim_white_margin(remove_dangling_leader(
        Image.open(os.path.join(D3, "wrist_worn.png")).convert("RGB")))
    # the COMPLETE wearing illustration, drawn as wide as the column allows
    ix0, ix1, iy0 = 6.0, 334.0, 24.0
    iw = ix1 - ix0
    ih = iw * worn.height / worn.width
    ax.imshow(worn, extent=[ix0, ix1, c.Y(iy0 + ih), c.Y(iy0)], zorder=1,
              aspect="auto")
    c.seg([(ix0, iy0), (ix1, iy0), (ix1, iy0 + ih), (ix0, iy0 + ih), (ix0, iy0)],
          lw=0.6, color="#bbbbbb")

    # channel / sizing boxes, now stacked below the artwork in a 3 x 2 grid
    bw = (iw - 2 * 8) / 3.0
    bxs = [ix0, ix0 + bw + 8, ix0 + 2 * (bw + 8)]
    bh_a = 27.0
    box_lines = [["Wrist PPG", "reflective, 256 Hz"],
                 ["Accelerometers", "wide + low-noise"],
                 ["Magnetometer", "3-ax, 256 Hz"],
                 ["Chest ECG", "reference, 256 Hz"],
                 ["Gyroscope", "3-ax, 256 Hz"],
                 ["Front-end sizing", "16-256 Hz, 8-16 bit"]]
    y_a1 = iy0 + ih + 7.0
    for i, lines in enumerate(box_lines):
        r, k = divmod(i, 3)
        c.box(bxs[k], y_a1 + r * (bh_a + 8), bw, bh_a, lines,
              fill=FILL_ALT if lines[0] == "Front-end sizing" else FILL)
    y_a_note = y_a1 + 2 * (bh_a + 8) + 1.0
    c.text(6, y_a_note, "The strap presses the optical window to the inner wrist.",
           fs=FS_NOTE, va="top", color="#333333")

    # ---------------- (b) shared sample clock ------------------------------
    y_b = y_a_note + 15.0
    c.text(4, y_b, "(b)", fs=FS_TITLE, bold=True, va="top")
    c.text(20, y_b, "Shared sample clock (one record grid)", fs=FS_TITLE,
           va="top")

    lanes = [("wrist PPG", FS.C["red"]), ("accel. wide", FS.C["blue"]),
             ("accel. low-noise", FS.C["sky"]), ("gyro", FS.C["purple"]),
             ("mag", FS.C["green"]), ("chest ECG", FS.C["orange"])]
    lx0, lx1 = 104.0, 326.0
    span_s, fs_rec = 0.5, 256.0
    step = (lx1 - lx0) / (span_s * fs_rec)
    ly0, ldy = y_b + 19.0, 11.0
    for i, (name, col) in enumerate(lanes):
        y = ly0 + i * ldy
        c.seg([(lx0, y), (lx1, y)], lw=0.6, color=LINE)
        c.text(lx0 - 3, y, name, fs=FS_NOTE, ha="right", va="center",
               color="#333333")
        for k in range(0, int(span_s * fs_rec) + 1, 16):
            x = lx0 + k * step
            c.seg([(x, y), (x, y - (3.2 if k % 64 == 0 else 2.0))], lw=0.5,
                  color=col)
    # time ruler under the lanes
    ry = ly0 + (len(lanes) - 1) * ldy + 12
    c.seg([(lx0, ry), (lx1, ry)], lw=0.6, color="#777777")
    for k in range(0, int(span_s * fs_rec) + 1, 64):
        x = lx0 + k * step
        c.seg([(x, ry), (x, ry + 2.5)], lw=0.6, color="#777777")
        lab = "0 s" if k == 0 else ("%g s" % (k / fs_rec))
        c.text(x, ry + 3.5, lab, fs=FS_NOTE,
               ha="right" if k == int(span_s * fs_rec) else "center",
               va="top", color="#555555")
    ny = ry + 15.0
    c.text(4, ny, "ticks: every 16th sample; all streams share one 256 Hz grid",
           fs=FS_NOTE, va="top", color="#555555")

    # ---------------- (c) signal path into the harmonic-comb estimator -----
    top = ny + 12.0
    c.text(4, top, "(c)", fs=FS_TITLE, bold=True, va="top")
    c.text(20, top, "Signal path into the harmonic-comb estimator",
           fs=FS_TITLE, va="top")

    cols = [(4.0, 100.0), (122.0, 100.0), (240.0, 96.0)]
    bh = 27.0
    r1, r2 = top + 14, top + 14 + bh + 10
    chain1 = [["wrist PPG + IMU", "256 Hz, 16 bit"],
              ["Stage 1", "0.4-8 Hz, quality gate"],
              ["Stage 2", "IMU-harmonic mask"]]
    chain2 = [["Stage 3", "gain, 0.7-3.2 Hz floor"],
              ["Stage 4", "iSTFT + comb"],
              ["HR (bpm)", "per 8 s window"]]
    for i, (lines, (x, w)) in enumerate(zip(chain1, cols)):
        c.box(x, r1, w, bh, lines, edge=EDGE if i == 0 else ACC,
              lw=0.8 if i == 0 else 1.0)
    for i, (lines, (x, w)) in enumerate(zip(chain2, cols)):
        c.box(x, r2, w, bh, lines, edge=ACC if i < 2 else EDGE,
              lw=1.0 if i < 2 else 0.8,
              fill=FILL_ALT if i == 2 else FILL)
    for i in range(2):
        x0 = cols[i][0] + cols[i][1]
        c.arrow((x0, r1 + bh / 2), (cols[i + 1][0], r1 + bh / 2))
        c.arrow((x0, r2 + bh / 2), (cols[i + 1][0], r2 + bh / 2))
    # wrap: end of row 1 -> start of row 2
    xm, xn = cols[2][0] + cols[2][1] / 2, cols[0][0] + cols[0][1] / 2
    c.route([(xm, r1 + bh), (xm, r1 + bh + 5), (xn, r1 + bh + 5), (xn, r2)],
            lw=0.7)

    fy = r2 + bh + 6
    c.text(4, fy, "Schematic of the recorded sensing arrangement (public Wrist\n"
                  "data): no hardware prototype and no measured power figure.",
           fs=FS_NOTE, va="top", color="#555555", italic=True, ls=1.45)

    # ---------------- (c-legacy-removed) ----------------------------------
    hits, bad_fit, out = qc_text(fig, c)
    for h in hits:
        print("A OVERLAP:", h)
    for b in bad_fit:
        print("A BOX-OVERFLOW:", b)
    for o in out:
        print("A OFF-CANVAS:", o)
    assert not hits and not bad_fit and not out, (hits, bad_fit, out)
    save(fig, "figA_sensor_chain.png")


# ===========================================================================
# figB - motion-subspace projection, six real-data panels stacked
# ===========================================================================
REC = "s1_high_resistance_bike"      # same record as src/fig5_subspace.py
WIN_S = 8.0                          # same window length
NPERSEG, HOP = 256, 64               # same STFT settings


def figB():
    rec = pl.load_wrist(os.path.join(WRIST, REC))
    fs = rec["fs"]
    ppg = rec["ppg"]
    imu = np.concatenate([rec["acc"], rec["gyro"]], axis=1)
    wl = int(WIN_S * fs)

    # same selection rule as the source script: largest accelerometer variance
    n_win = max(1, (len(ppg) - wl) // wl)
    best, best_v = 0, -1.0
    for i in range(n_win):
        s = i * wl
        v = float(np.var(pl.accel_magnitude(imu[s:s + wl, :3])))
        if v > best_v:
            best_v, best = v, s
    p = ppg[best:best + wl]
    i_ = imu[best:best + wl]
    print("figB record %s window t0=%.1fs accel var=%.3f"
          % (REC, best / fs, best_v))

    cleaned, diag = K.kisr(p, i_, fs, nperseg=NPERSEG, hop=HOP,
                           return_diag=True)
    freqs = diag["freqs"]
    mask, coh, gain = diag["mask"], diag["coh"], diag["gain"]
    Pb, Pa = diag["pow_before"], diag["pow_after"]

    band = (freqs >= 0.4) & (freqs <= 5.0)
    fb = freqs[band]
    t = np.arange(p.size) / fs
    tt = np.arange(Pb.shape[0]) * HOP / fs

    H = 4.30
    fig = plt.figure(figsize=(W_IN, H))
    L = 0.38                   # left margin (ylabel + tick labels), inches
    WP = 2.68                  # data-panel width, inches (colourbar fits in 3.4)
    ph, gap, top = 0.50, 0.16, 0.16

    def axes_at(i, h=ph):
        y = H - top - i * (ph + gap) - h
        return fig.add_axes([L / W_IN, y / H, WP / W_IN, h / H])

    def place(ax, s):
        # panel label lives in the gap ABOVE the axes, never over the data
        ax.text(0.0, 1.04, s, transform=ax.transAxes, fontsize=FS_NOTE,
                fontweight="bold", va="bottom", ha="left", zorder=6,
                clip_on=False)

    def frame(ax, xmax):
        ax.set_xlim(0, xmax)
        ax.set_ylim(0.4, 5.0)
        ax.set_ylabel("Hz", labelpad=1.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_xticks([0, 2, 4, 6])
        ax.set_xticklabels([])
        ax.tick_params(length=1.8, width=0.5, pad=1.0)

    # the STFT frames cover 0 .. last frame + one hop; same extent as the
    # source script, which let these panels autoscale
    tmax = (Pb.shape[0] - 1) * HOP / fs + HOP / fs

    def cbar(ax, im, ticks):
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, ticks=ticks)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=FS_NOTE, length=1.6, pad=1.0)
        return cb

    # (a) PPG spectrogram before
    ax = axes_at(0)
    frame(ax, tmax)
    Pdb = 10 * np.log10(Pb[:, band].T + 1e-12)
    vmax = np.percentile(Pdb, 99)
    im = ax.pcolormesh(tt, fb, Pdb - vmax, cmap="magma", vmin=-40, vmax=0,
                       shading="auto")
    place(ax, "(a) PPG spectrogram, before (dB rel. peak)")
    cbar(ax, im, [-40, 0])

    # (b) kinematic motion subspace mask
    ax = axes_at(1)
    frame(ax, tmax)
    im = ax.pcolormesh(tt, freqs, mask.astype(float).T, cmap="Blues", vmin=0,
                       vmax=1, shading="auto")
    place(ax, "(b) Motion subspace marked by the IMU (0/1)")
    cbar(ax, im, [0, 1])

    # (c) PPG-motion coherence
    ax = axes_at(2)
    frame(ax, tmax)
    im = ax.pcolormesh(tt, freqs, coh.T, cmap="viridis", vmin=0, vmax=1,
                       shading="auto")
    place(ax, "(c) PPG-motion coherence")
    cbar(ax, im, [0, 1])

    # (d) applied gain
    ax = axes_at(3)
    frame(ax, tmax)
    im = ax.pcolormesh(tt, freqs, gain.T, cmap="turbo", vmin=0, vmax=1,
                       shading="auto")
    place(ax, "(d) Applied gain (cardiac bins protected)")
    cbar(ax, im, [0, 1])

    # (e) reconstructed spectrogram
    ax = axes_at(4)
    frame(ax, tmax)
    Pda = 10 * np.log10(Pa[:, band].T + 1e-12)
    vmax2 = np.percentile(Pda, 99)
    im = ax.pcolormesh(tt, fb, Pda - vmax2, cmap="magma", vmin=-40, vmax=0,
                       shading="auto")
    place(ax, "(e) Reconstructed spectrogram (dB rel. peak)")
    cbar(ax, im, [-40, 0])

    # (f) time domain before / after
    ax = axes_at(5, h=0.55)
    ax.plot(t, p - p.mean(), color=FS.C["black"], lw=0.6, label="Raw PPG")
    ax.plot(t, cleaned - cleaned.mean() - 2.2 * np.std(p), color=FS.C["red"],
            lw=0.6, label="KISR output")
    ax.set_xlim(0, WIN_S)
    ax.set_xlabel("Time (s)", labelpad=1.0)
    ax.set_ylabel("a.u.", labelpad=1.5)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([])
    ax.tick_params(length=1.8, width=0.5, pad=1.0)
    # headroom for the panel title, so it does not sit on the traces
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.42 * (y1 - y0))
    ax.legend(frameon=False, ncols=2, columnspacing=0.8,
              fontsize=FS_NOTE, loc="upper right", handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.3, labelspacing=0.35)
    place(ax, "(f) Time domain (%s)" % REC)

    save(fig, "figB_subspace.png")


def main():
    figA()
    figB()


if __name__ == "__main__":
    main()
