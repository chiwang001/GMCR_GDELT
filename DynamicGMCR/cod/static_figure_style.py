"""Shared publication style for DynamicGMCR static-experiment figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


OKABE_ITO = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#242424",
    "gray": "#8A8A8A",
    "light_gray": "#D9D9D9",
}


def configure_static_figure_style() -> str:
    """Configure English-only Times New Roman publication typography."""
    font_path = Path("C:/Windows/Fonts/times.ttf")
    family = "Times New Roman"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family, "Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": OKABE_ITO["black"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return family


def save_static_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Write an editable vector pair and a 600-dpi PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            bbox_inches="tight",
            pad_inches=0.08,
            **kwargs,
        )
    plt.close(fig)
