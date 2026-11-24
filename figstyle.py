"""Paper figure style and shared plotting helpers (IEEE two-column friendly)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE-friendly: the template lists Times New Roman, Helvetica, Arial, Cambria
# and Symbol as the accepted fonts inside figures; Arial is the sans face
# shipped with Windows, so it is used here.
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 950,
    "font.size": 8.9,
    "axes.titlesize": 9.3,
    "axes.labelsize": 8.9,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.6,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.grid": False,
})

# colorblind-safe 5-color palette (Okabe-Ito)
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
METHOD_COLORS = {
    "Raw (bandpass)": C["black"],
    "LMS adaptive": C["sky"],
    "Spectral subtraction": C["green"],
    "Wavelet+IMU": C["purple"],
    "ICA+IMU": C["orange"],
    "KISR (proposed)": C["red"],
}
ACT_COLORS = {"bike": C["red"], "walk": C["blue"], "run": C["green"]}
ACT_LABEL = {"bike": "Biking", "walk": "Walking", "run": "Running"}
