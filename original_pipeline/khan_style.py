"""
Nature-style figure defaults for the Khan 2 revision.

Journal conventions applied here:
  * sans-serif throughout (Arial where installed, otherwise the metrically
    identical Liberation Sans / Nimbus Sans)
  * 5-7 pt type, hairline axes, no top/right spines
  * single-column 89 mm, 1.5-column 120 mm, double-column 183 mm widths
  * lowercase bold panel letters set outside the axes
  * Okabe-Ito colourblind-safe palette used for every categorical scale
  * every figure exported as 600 dpi TIFF (LZW), 600 dpi PNG and vector PDF
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ------------------------------------------------------------------ typography
SANS = ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": SANS,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # editable text in vector output
    "svg.fonttype": "none",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "figure.titlesize": 8,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.minor.size": 1.3, "ytick.minor.size": 1.3,
    "lines.linewidth": 0.9, "lines.markersize": 3,
    "patch.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 1.2,
    "legend.handletextpad": 0.5, "legend.columnspacing": 1.0,
    "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

MM = 1 / 25.4
W1, W15, W2 = 89 * MM, 120 * MM, 183 * MM        # Nature column widths

# ----------------------------------------------------------------- palettes
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
             "#56B4E9", "#F0E442", "#000000"]

GROUP = {"Healthy": "#009E73", "Diabetic": "#E69F00",
         "Healing": "#0072B2", "Nonhealing": "#D55E00",
         "DiabeticSkin": "#999999"}

STATE = {"Resident M2-like macrophage": "#D55E00",
         "Inflammatory monocyte-macrophage": "#0072B2",
         "Lipid-associated macrophage": "#E69F00",
         "cDC1": "#009E73", "cDC2": "#56B4E9", "pDC": "#CC79A7",
         "LC/Langerhans": "#F0E442", "Mature/migratory DC": "#7F7F7F",
         "Proliferating myeloid": "#000000"}

SEQ = "viridis"
DIV = "RdBu_r"


def panel(ax, letter, dx=-0.085, dy=1.06, size=8):
    """Lowercase bold panel letter, Nature style, placed outside the axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="top", ha="left")


def tidy(ax, grid_axis=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, lw=0.3, color="#CCCCCC", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
    return ax


def save(fig, outdir, name, formats=("png", "pdf", "tiff")):
    """Write one figure in every submission format; returns the paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for f in formats:
        p = os.path.join(outdir, f"{name}.{f}")
        if f == "tiff":
            fig.savefig(p, dpi=600, format="tiff",
                        pil_kwargs={"compression": "tiff_lzw"})
        elif f == "pdf":
            fig.savefig(p, format="pdf")               # vector
        else:
            fig.savefig(p, dpi=600, format=f)
        paths.append(p)
    plt.close(fig)
    print(f"    saved {name}: " + ", ".join(os.path.basename(p) for p in paths), flush=True)
    return paths


def source_data(df, outdir, name):
    """Write the exact numbers behind a panel next to the figure."""
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{name}.csv")
    df.to_csv(p, index=False)
    return p
