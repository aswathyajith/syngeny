# Default multiplier for every font size below and the hardcoded sizes in the heatmap plots.
# The plot functions take a `font_scale` argument that overrides it.
FONT_SCALE = 2.5


def plot_style(font_scale: float = FONT_SCALE) -> dict:
    """Returns the matplotlib rcParams with every font size multiplied by `font_scale`."""

    return {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8 * font_scale,
        "axes.labelsize": 10.2 * font_scale,
        "axes.titlesize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 9 * font_scale,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


PLOT_STYLE = plot_style()
