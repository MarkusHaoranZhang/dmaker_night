"""Shared setup for experiment scripts: matplotlib config and output directories."""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Suppress only the matplotlib font / glyph warnings that appear during
# headless rendering. Numerical warnings from NumPy and SciPy are kept
# enabled so genuine bugs (division by zero, log of zero, etc.) remain
# visible during development.
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")

matplotlib.use("Agg")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def project_root():
    """Resolve the repository root regardless of where the script is launched from."""
    return Path(__file__).resolve().parents[1]


def figures_dir():
    p = project_root() / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def tables_dir():
    p = project_root() / "tables"
    p.mkdir(parents=True, exist_ok=True)
    return p


# Number of independent random seeds used to aggregate experimental results.
# The paper averages every metric over 30 independent runs.
N_SEEDS = 30
