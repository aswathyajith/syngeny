"""Plot cross-temperature similarity heatmaps, one panel per model.

Expects the long table from `build_temperature_sim_df`, with the columns
['model', 'word', 'temp_a', 'temp_b', 'seed_a', 'seed_b', 'sim']. Similarities are summarised (mean or
std) over words and seeds into a temperature x temperature matrix, and drawn on a
shared colour scale. The matrix is symmetric, so only one triangle is shown
(temperature increases up the y-axis and to the left on the x-axis). Off-diagonal cells compare
temperatures within a seed; the diagonal compares seeds within a temperature.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from plotting import configs

# Single-hue sequential ramps: light = low, dark = high. Blue for mean similarity,
# orange for its spread, so the two are never mistaken for one another.
STAT_STYLES = {
    "mean": (LinearSegmentedColormap.from_list("sim_mean", ["#eaf2fc", "#8cbcf0", "#2a78d6", "#14467f"]),
             "Mean cosine similarity"),
    "std": (LinearSegmentedColormap.from_list("sim_std", ["#fdf0ea", "#f7b396", "#eb6834", "#8f3410"]),
            "Std of cosine similarity"),
}


def cell_ink(im, value: float) -> str:
    """Dark or white text colour, whichever contrasts with the colour `im` draws `value` in."""

    red, green, blue = im.cmap(im.norm(value))[:3]
    return "#2e2d2b" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.5 else "white"


def _model_size(model: str) -> float:
    """Parameter count in billions parsed from the model name (e.g. 'Qwen2.5-0.5B-Instruct' -> 0.5); unparseable names sort last."""

    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)b(?![a-z])", str(model), flags=re.IGNORECASE)
    return float(match.group(1)) if match else float("inf")


def _sim_matrix(model_df: pd.DataFrame, temps: list, stat: str = "mean") -> pd.DataFrame:
    """`stat` of the similarity as a symmetric temperature x temperature matrix (diagonal = across seeds)."""

    agg = model_df.groupby(["temp_a", "temp_b"])["sim"].agg(stat)
    mat = pd.DataFrame(np.nan, index=temps, columns=temps)
    for (a, b), sim in agg.items():
        mat.loc[a, b] = mat.loc[b, a] = sim
    return mat


def plot_temp_heatmap(sim_df: pd.DataFrame, label_map: dict | None = None, ncols: int = 4, vmin=None, vmax=None, stat: str = "mean", relative: bool = False, cmap=None, font_scale: float | None = None):
    """Draws one temperature x temperature heatmap per model; returns the figure.

    `stat` is "mean" or "std": the statistic of the similarities over words and
    seeds shown in each cell. The std is the sample std (ddof=1), so cells backed
    by a single similarity are blank.

    With `relative=True`, each cell (j, k) is shown relative to the higher of its two
    temperatures: the diagonal cell of that temperature, (i, i) with i = max(j, k), is
    subtracted, so the diagonal becomes 0. Drawn with a diverging PiYG map centred on 0.

    `cmap` is a matplotlib colormap or its name, replacing the default for `stat` (or
    PiYG when `relative`). `font_scale` multiplies every font size and defaults to
    `configs.FONT_SCALE`.
    """

    font_scale = configs.FONT_SCALE if font_scale is None else font_scale
    plt.rcParams.update(configs.plot_style(font_scale))  # read at call time so edits to configs.py take effect on reload

    if stat not in STAT_STYLES:
        raise ValueError(f"stat must be one of {sorted(STAT_STYLES)} (got '{stat}').")
    default_cmap, cbar_label = STAT_STYLES[stat]
    if relative:
        default_cmap = "PiYG"
        cbar_label = f"{cbar_label}\n(relative to higher temperature)"
    cmap = default_cmap if cmap is None else cmap

    temps = sorted(set(sim_df["temp_a"]) | set(sim_df["temp_b"]))
    models = sorted(sim_df["model"].unique(), key=_model_size)
    # Scale to the plotted cell values (aggregates), not the raw per-word similarities.
    mats = {m: _sim_matrix(sim_df[sim_df["model"] == m], temps, stat) for m in models}
    if relative:
        # temps are sorted ascending, so the higher temperature of cell (j, k) is index max(j, k).
        idx = np.maximum.outer(np.arange(len(temps)), np.arange(len(temps)))
        for m, mat in mats.items():
            diag = np.diag(mat.values)
            mats[m] = pd.DataFrame(mat.values - diag[idx], index=mat.index, columns=mat.columns)
    all_values = np.concatenate([mat.values.ravel() for mat in mats.values()])
    if relative:  # symmetric about 0 so the midpoint of the diverging map is "no change"
        bound = np.nanmax(np.abs(all_values))
        vmin = -bound if vmin is None else vmin
        vmax = bound if vmax is None else vmax
    else:
        vmin = np.nanmin(all_values) if vmin is None else vmin
        vmax = np.nanmax(all_values) if vmax is None else vmax
    ncols = min(ncols, len(models))
    nrows = -(-len(models) // ncols)

    # Size panels and text with the number of temperatures so cells stay legible.
    n = len(temps)
    panel = max(3.4, 0.6 * n + 1.4)
    sign = "+" if relative else ""  # show +/- on relative differences
    annot_size = float(np.clip(13 - n, 6, 9) * font_scale)
    tick_labels = [f"{t:g}" for t in temps]

    fig, axes = plt.subplots(nrows, ncols, figsize=(0.88 * panel * ncols, panel * nrows + 1.8), squeeze=False, constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.25, hspace=0.1, w_pad=0.2, wspace=0.01)  # extra vertical space between rows, tight columns
    for k, (ax, model) in enumerate(zip(axes.flat, models)):
        mat = mats[model]
        # Mean over the empty (upper) triangle, which mirrors the strictly-lower cells shown.
        empty_mean = np.nanmean(mat.values[np.triu_indices(n, k=1)])
        empty_std = np.nanstd(mat.values[np.triu_indices(n, k=1)])
        mat = mat.where(np.triu(np.ones(mat.shape, dtype=bool)))  # one triangle only
        im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        ax.set_xticks(range(n), tick_labels, fontsize=annot_size + 4)
        ax.set_yticks(range(n), tick_labels, fontsize=annot_size + 4)
        # ax.invert_xaxis()
        ax.set_xlabel("Temperature", labelpad=14)
        if k % ncols == 0:  # only the first column gets the y-axis label
            ax.set_ylabel("Temperature", labelpad=14)
        else:  # y-axis is shared, so hide tick marks and labels on the other columns
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.text(0.4, 0.9, rf"${empty_mean:.2f} \pm {empty_std:.2f}$", transform=ax.transAxes, ha="left", va="top",
                fontsize=annot_size + 8, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", edgecolor="#2e2d2b", linewidth=0.8))
        ax.set_title((label_map or {}).get(model, model), fontsize=10 * font_scale)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, j in np.ndindex(mat.shape):
            if not np.isnan(mat.values[i, j]):
                value = mat.values[i, j]
                ax.text(j, i, f"{value:{sign}.1f}", ha="center", va="center", fontsize=annot_size,
                        color=cell_ink(im, value))
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, location="bottom", orientation="horizontal", shrink=0.6, aspect=40, label=cbar_label)
    cbar.outline.set_visible(False)
    return fig
