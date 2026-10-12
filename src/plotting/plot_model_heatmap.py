"""Plot cross-model similarity at one temperature as a model x model heatmap.

Expects the long table from `build_model_sim_df`, with the columns
['model_a', 'model_b', 'word', 'seed_a', 'seed_b', 'sim']. Similarities are
summarised (mean or std) over words and seeds into a model x model matrix. The
matrix is symmetric. Off-diagonal cells compare different models on the same
word; the diagonal compares seeds within a model, i.e. how similar a model is to
itself. Lines separate model families, and a box marks the most similar other
model in each row: orange if it is in the same family, cornflower blue if not.

Run as a script to build the table from processed_results and save one figure per temperature:

    python src/plotting/plot_model_heatmap.py processed_results/sample-10/Qwen2.5.jsonl \
        processed_results/sample-10/Gemma3.jsonl --temperature 0.0 1.0 --output-dir figures/sample-10/model-sim
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference.models import MODEL_LABEL_MAP
from plotting import configs
from plotting.plot_temp_heatmap import STAT_STYLES, cell_ink
from plotting.process_data.model_sim import build_model_sim_df, load_wide_df

# Display labels in the order models are drawn: grouped by family, smallest first.
LABELS = {model: label for family in MODEL_LABEL_MAP.values() for model, label in family.items()}
FAMILIES = {model: name for name, family in MODEL_LABEL_MAP.items() for model in family}

# Box colours for each row's most similar other model.
SAME_FAMILY_COLOR = "darkorange"
OTHER_FAMILY_COLOR = "royalblue"
BOX_LINEWIDTH = 6

# Default tick label size relative to the base annotation size.
TICK_SIZE_FACTOR = 1.25


def _sim_matrix(sim_df: pd.DataFrame, models: list, stat: str = "mean") -> pd.DataFrame:
    """`stat` of the similarity as a symmetric model x model matrix (diagonal = across seeds)."""

    agg = sim_df.groupby(["model_a", "model_b"])["sim"].agg(stat)
    mat = pd.DataFrame(np.nan, index=models, columns=models)
    for (a, b), sim in agg.items():
        mat.loc[a, b] = mat.loc[b, a] = sim
    return mat


def plot_model_heatmap(sim_df: pd.DataFrame, label_map: dict | None = None, stat: str = "mean", vmin=None, vmax=None, title: str | None = None, cmap=None, font_scale: float | None = None, xtick_size: float | None = None, ytick_size: float | None = None):
    """Draws the model x model heatmap for one temperature; returns the figure.

    `stat` is "mean" or "std": the statistic of the similarities over words and
    seeds shown in each cell. The std is the sample std (ddof=1), so cells backed
    by a single similarity are blank. Models are ordered as in `LABELS`, with
    unknown models last.

    `cmap` is a matplotlib colormap or its name, replacing the default for `stat`.
    `font_scale` multiplies every font size and defaults to `configs.FONT_SCALE`.
    `xtick_size` and `ytick_size` set the tick label sizes in points, independently of
    `font_scale`; each defaults to `TICK_SIZE_FACTOR` times the base annotation size.
    """

    font_scale = configs.FONT_SCALE if font_scale is None else font_scale
    plt.rcParams.update(configs.plot_style(font_scale))  # read at call time so edits to configs.py take effect on reload

    if stat not in STAT_STYLES:
        raise ValueError(f"stat must be one of {sorted(STAT_STYLES)} (got '{stat}').")
    default_cmap, cbar_label = STAT_STYLES[stat]
    cmap = default_cmap if cmap is None else cmap
    label_map = LABELS if label_map is None else label_map

    present = set(sim_df["model_a"]) | set(sim_df["model_b"])
    models = [m for m in label_map if m in present] + sorted(present - set(label_map))
    mat = _sim_matrix(sim_df, models, stat)
    vmin = np.nanmin(mat.values) if vmin is None else vmin
    vmax = np.nanmax(mat.values) if vmax is None else vmax

    n = len(models)
    cell = 1.2
    annot_size = float(np.clip(13 - n, 6, 9) * font_scale)
    labels = [label_map.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(cell * n + 4, cell * n + 3), constrained_layout=True)
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    xtick_size = annot_size * TICK_SIZE_FACTOR if xtick_size is None else xtick_size
    ytick_size = annot_size * TICK_SIZE_FACTOR if ytick_size is None else ytick_size
    ax.set_xticks(range(n), labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=xtick_size)
    ax.set_yticks(range(n), labels, fontsize=ytick_size)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, j in np.ndindex(mat.shape):
        value = mat.values[i, j]
        if not np.isnan(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=annot_size * 0.8,
                    color=cell_ink(im, value))
    # Separate model families with lines
    for k in range(1, n):
        if FAMILIES.get(models[k]) != FAMILIES.get(models[k - 1]):
            ax.axhline(k - 0.5, color="#2e2d2b", linewidth=2)
            ax.axvline(k - 0.5, color="#2e2d2b", linewidth=2)
    # Box the largest value in each row, leaving out the diagonal (the model against itself):
    # orange if the best match is in the same family, cornflower blue if it is in another.
    off_diag = mat.values.copy()
    np.fill_diagonal(off_diag, np.nan)
    for i in range(n):
        if not np.isnan(off_diag[i]).all():
            j = int(np.nanargmax(off_diag[i]))
            same_family = FAMILIES.get(models[i]) == FAMILIES.get(models[j])
            # clip_on=False so boxes on the outer edge of the heatmap keep their full stroke
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, linewidth=BOX_LINEWIDTH, clip_on=False, zorder=3,
                                   edgecolor=SAME_FAMILY_COLOR if same_family else OTHER_FAMILY_COLOR))
    if title:
        ax.set_title(title, fontsize=10 * font_scale, pad=14)  # room for boxes on the top row
    cbar = fig.colorbar(im, ax=ax, location="right", shrink=0.6, label=cbar_label)
    cbar.outline.set_visible(False)
    return fig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot cross-model similarity at a given temperature.")
    parser.add_argument("input_paths", nargs="+", help="Wide per-family .jsonl tables (word, temperature, seed, one column per model).")
    parser.add_argument("--temperature", type=float, nargs="+", required=True, help="Temperature(s) to plot, one figure each.")
    parser.add_argument("--stat", choices=sorted(STAT_STYLES), default="mean", help="Statistic over words and seeds shown in each cell.")
    parser.add_argument("--cmap", default=None, help="Matplotlib colormap name (default: the blue ramp for mean, the orange ramp for std).")
    parser.add_argument("--font-scale", type=float, default=None, help=f"Multiplier for every font size (default: {configs.FONT_SCALE}).")
    parser.add_argument("--xtick-size", type=float, default=None, help="x tick label size in points (default: scales with --font-scale).")
    parser.add_argument("--ytick-size", type=float, default=None, help="y tick label size in points (default: scales with --font-scale).")
    parser.add_argument("--no-title", action="store_true", help="Leave the temperature title off the figure.")
    parser.add_argument("--same-seed-only", action="store_true", help="Compare different models only at the same seed.")
    parser.add_argument("--output-dir", required=True, help="Directory for the figures, saved as <stat>/temp=<temperature>.pdf.")
    parser.add_argument("--dpi", type=int, default=200, help="Resolution of raster figures.")
    args = parser.parse_args()

    wide_df = load_wide_df(args.input_paths)
    model_cols = [c for c in wide_df.columns if c not in ("word", "temperature", "seed")]
    for temperature in args.temperature:
        sim_df = build_model_sim_df(wide_df, temperature, model_cols=model_cols, same_seed_only=args.same_seed_only)
        skipped = sorted(set(model_cols) - set(sim_df["model_a"]) - set(sim_df["model_b"]))
        if skipped:
            print(f"temperature {temperature:g}: no outputs for {skipped}, left out of the heatmap.")
        fig = plot_model_heatmap(sim_df, stat=args.stat, title=None if args.no_title else f"Temperature {temperature:g}", cmap=args.cmap, font_scale=args.font_scale, xtick_size=args.xtick_size, ytick_size=args.ytick_size)
        output_path = Path(args.output_dir) / args.stat / f"temp={temperature:g}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to {output_path}")
