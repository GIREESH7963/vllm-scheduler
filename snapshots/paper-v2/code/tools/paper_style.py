"""Shared figure style for the paper.

Colour choices are not free-hand. The categorical slots below are a validated palette: assigned
in fixed order and never cycled, so a policy keeps its hue across every figure even when a
figure shows a subset. The set was checked for colour-vision-deficiency separation (worst
adjacent pair ΔE 9.1 protan, normal-vision floor 19.6) against the light surface used here.

Three slots sit below 3:1 contrast on white, so every figure that uses them also carries a
second, non-colour encoding — distinct markers and dashes, plus a legend — and identity is never
communicated by colour alone.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- validated categorical palette, fixed order -----------------------------------------
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Stable identity: a policy owns a slot for the whole paper.
POLICY_COLOR = {
    "nocap":    SLOTS[0],
    "fcfs":     SLOTS[1],
    "srpt":     SLOTS[2],
    "edf":      SLOTS[3],
    "adaptive": SLOTS[4],
}
POLICY_MARKER = {"nocap": "o", "fcfs": "s", "srpt": "^", "edf": "D", "adaptive": "v"}
POLICY_DASH = {
    "nocap": (None, None), "fcfs": (5, 2), "srpt": (1, 1.5),
    "edf": (7, 2, 1, 2), "adaptive": (3, 1.5),
}
POLICY_ORDER = ["nocap", "fcfs", "srpt", "edf", "adaptive"]

MODEL_COLOR = {"1.5b": SLOTS[0], "3b": SLOTS[1]}
MODEL_MARKER = {"1.5b": "o", "3b": "s"}

# Single-hue sequential ramp (blue, light -> dark) for magnitude encodings.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e3e2dd"
SURFACE = "#ffffff"

STATUS_BAD = "#e34948"
STATUS_WARN = "#eda100"
STATUS_GOOD = "#1baf7a"


def apply() -> None:
    """Install the paper rcParams. Recessive axes and grid; data carries the emphasis."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save(fig, out_dir, name: str) -> list:
    """Write PNG (review) and PDF (vector, for the manuscript)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def annotate(ax, text: str, xy, xytext, color=INK_2, fontsize=8) -> None:
    """Callout with a thin leader line — used sparingly, never one per point."""
    ax.annotate(
        text, xy=xy, xytext=xytext, fontsize=fontsize, color=color,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=0, shrinkB=3),
        va="center",
    )
