"""Plot mean cross-model similarity, as grouped bars or as a real-vs-fake heatmap.

Both plots expect a long table with the columns ['model_pair', 'sim_type',
'word_source', 'mean_sim']: one row per (model pair, similarity metric, word
source).

`plot_mean_sim` draws the aggregates themselves: each model_pair becomes a
subplot, each sim_type a group of bars with its own hue, and each word_source a
shade within that hue, dark to light. `plot_real_fake_diff` instead draws the
gap between two word sources as a model x model heatmap, one panel per sim_type.
"""

import argparse
import ast
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex, to_rgb
from matplotlib.patches import Patch
from inference.models import HF_MODEL_LABEL_MAP

FIELDS = ["model_pair", "sim_type", "word_source", "mean_sim"]

# One ramp per sim_type, assigned in a fixed order and never cycled. Each ramp
# is (dark, base, light); word_source shades are interpolated between them.
SERIES_RAMPS = [
    ("#14467f", "#2a78d6", "#8cbcf0"),  # blue
    ("#8f3410", "#eb6834", "#f7b396"),  # orange
    ("#0b6b48", "#1baf7a", "#86dcbc"),  # aqua
    ("#2e2470", "#6a58c9", "#bdb2f5"),  # violet
]
# Neutral ramp, used for the word_source legend so shade reads as its own channel.
NEUTRAL_RAMP = ("#2e2d2b", "#7a7872", "#c9c7c0")

MAX_SHADES = len(NEUTRAL_RAMP)

# Diverging ramp for signed differences: a cool and a warm arm either side of a
# neutral midpoint, so the sign reads as hue and the magnitude as lightness. The
# two arms are matched step for step in lightness (OKLab L within 0.013), which
# keeps an equal difference equally prominent whichever way it points.
DIVERGING_RAMP = (
    ("#0d366b", "#2a78d6", "#9ec5f4"),  # blue, the negative arm (dark to light)
    "#f0efec",  # neutral midpoint: zero difference must read as "nothing"
    ("#f4a5a5", "#d13a39", "#661817"),  # red, the positive arm (light to dark)
)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dcdcd8"
SURFACE = "#fcfcfb"


def pair_members(model_pair) -> tuple[str, str]:
    """Returns the two models of a pair, ordered by their display label.

    Accepts the same inputs as `pair_label`, and raises if the pair does not
    parse into exactly two members (a heatmap cell needs both of them).
    """

    parsed = _parse_pair(model_pair)
    if isinstance(parsed, str) or len(parsed) != 2:
        raise ValueError(f"Expected a pair of two models, got {model_pair!r}.")
    return tuple(sorted((str(model) for model in parsed), key=model_label))


def model_label(model) -> str:
    """Returns the display label of a model, falling back to its raw name."""

    return MODEL_LABEL_MAP.get(str(model), str(model))


def pair_label(model_pair) -> str:
    """Formats a model pair as "model_a / model_b", with the members sorted.

    Accepts a frozenset/iterable of model names, or the string a round-trip
    through CSV leaves behind (e.g. "frozenset({'gpt-4o', 'claude'})").
    """

    parsed = _parse_pair(model_pair)
    if isinstance(parsed, str):
        return parsed
    return " / ".join(sorted(MODEL_LABEL_MAP[str(model)] for model in parsed))


def _parse_pair(model_pair):
    """Recovers a model pair from the string a round-trip through CSV leaves behind.

    Returns an iterable of model names, or `model_pair` unchanged if it is a
    string that does not parse as one (e.g. an already-formatted pair label).
    """

    if not isinstance(model_pair, str):
        return model_pair
    try:
        return ast.literal_eval(model_pair.removeprefix("frozenset(").removesuffix(")"))
    except (ValueError, SyntaxError):
        return model_pair


def shades(ramp: tuple[str, str, str], n: int) -> list[str]:
    """Picks `n` evenly spaced shades from a (dark, base, light) ramp.

    A single shade is the base hue; more are spread over the full ramp so that
    neighbouring shades stay distinguishable.
    """

    if not 1 <= n <= MAX_SHADES:
        raise ValueError(f"n must be between 1 and {MAX_SHADES} (got {n}).")

    anchors = [to_rgb(color) for color in ramp]
    positions = [0.5] if n == 1 else [i / (n - 1) for i in range(n)]

    picked = []
    for position in positions:
        # Position 0 is the dark anchor, 0.5 the base, 1 the light anchor.
        lower = min(int(position * 2), 1)
        weight = position * 2 - lower
        picked.append(to_hex([
            a + weight * (b - a) for a, b in zip(anchors[lower], anchors[lower + 1])
        ]))
    return picked


def plot_mean_sim(
        df: pd.DataFrame,
        model_pairs: list | None = None,
        sim_types: list | None = None,
        word_sources: list | None = None,
        value_labels: bool = True,
        ncols: int = 3,
        subplot_width: float = 4.0,
        subplot_height: float = 3.5,
        title: str | None = None,
    ) -> plt.Figure:
    """Draws one subplot of grouped `mean_sim` bars per model pair.

    Within a subplot, bars are grouped by `sim_type` (hue) and shaded by
    `word_source` (dark to light). The orders default to the sorted unique
    values; pass `model_pairs`, `sim_types` or `word_sources` to override them
    or to plot a subset. `value_labels` prints each bar's value above it, which
    keeps the lighter shades readable without relying on color.
    """

    missing = [field for field in FIELDS if field not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} (expected {FIELDS}).")

    df = df.assign(pair_label=[pair_label(pair) for pair in df.model_pair])
    cell = ["pair_label", "sim_type", "word_source"]
    duplicates = df[df.duplicated(cell)]
    if not duplicates.empty:
        raise ValueError(
            "Expected one row per (model_pair, sim_type, word_source); duplicates: "
            f"{duplicates[cell].head().to_dict('records')}"
        )

    if model_pairs is None:
        model_pairs = sorted(df.pair_label.unique())
    else:
        model_pairs = [pair_label(pair) for pair in model_pairs]
    sim_types = sorted(df.sim_type.unique()) if sim_types is None else list(sim_types)
    word_sources = sorted(df.word_source.unique()) if word_sources is None else list(word_sources)

    if len(sim_types) > len(SERIES_RAMPS):
        raise ValueError(
            f"{len(sim_types)} sim_types exceeds the {len(SERIES_RAMPS)} available hues; "
            "plot a subset or facet by sim_type instead."
        )
    if len(word_sources) > MAX_SHADES:
        raise ValueError(
            f"{len(word_sources)} word_sources exceeds the {MAX_SHADES} distinguishable "
            "shades per hue; plot a subset or facet by word_source instead."
        )
    if len(sim_types) > 2 and len(word_sources) > 2:
        warnings.warn(
            f"{len(sim_types)} hues x {len(word_sources)} shades: the lightest shades of "
            "different hues are hard to tell apart (and near-indistinguishable under "
            "colorblindness). Consider faceting by sim_type instead.",
            stacklevel=2,
        )

    # colors[sim_type][word_source]: hue carries the metric, shade the word source.
    colors = {
        sim_type: dict(zip(word_sources, shades(ramp, len(word_sources))))
        for sim_type, ramp in zip(sim_types, SERIES_RAMPS)
    }

    nrows = -(-len(model_pairs) // ncols)
    ncols = min(ncols, len(model_pairs))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(subplot_width * ncols, subplot_height * nrows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.flatten()

    # Bars within a group are inset slightly so neighbours are separated by the
    # subplot background rather than touching.
    group_width = 0.8
    bar_width = group_width / len(word_sources)

    for ax, pair in zip(flat_axes, model_pairs):
        facet = df[df.pair_label == pair]
        for i, word_source in enumerate(word_sources):
            series = facet[facet.word_source == word_source].set_index("sim_type").mean_sim
            offset = (i - (len(word_sources) - 1) / 2) * bar_width
            positions = [x + offset for x in range(len(sim_types))]
            values = [series.get(sim_type, float("nan")) for sim_type in sim_types]
            bars = ax.bar(
                positions,
                values,
                width=bar_width * 0.9,
                color=[colors[sim_type][word_source] for sim_type in sim_types],
                zorder=2,
            )
            if value_labels:
                ax.bar_label(
                    bars, fmt="%.2f", padding=2, fontsize=7, color=TEXT_SECONDARY
                )

        ax.set_title(pair, fontsize=11, color=TEXT_PRIMARY)
        ax.set_xticks(range(len(sim_types)), sim_types, fontsize=9, color=TEXT_SECONDARY)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.tick_params(length=0, labelsize=8, colors=TEXT_SECONDARY)

    for ax in flat_axes[len(model_pairs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean similarity", fontsize=9, color=TEXT_SECONDARY)
    if df.mean_sim.min() >= 0:
        flat_axes[0].set_ylim(0, min(1.0, df.mean_sim.max() * 1.25))

    _add_legends(fig, sim_types, word_sources)
    if title:
        fig.suptitle(title, fontsize=13, color=TEXT_PRIMARY, x=0.01, y=1, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (subplot_height * nrows)), h_pad=2)
    return fig


def _add_legends(fig: plt.Figure, sim_types: list, word_sources: list):
    """Adds the hue (sim_type) and shade (word_source) legends above the subplots."""

    hue_handles = [
        Patch(facecolor=shades(ramp, 1)[0], label=sim_type)
        for sim_type, ramp in zip(sim_types, SERIES_RAMPS)
    ]
    shade_handles = [
        Patch(facecolor=shade, label=word_source)
        for word_source, shade in zip(word_sources, shades(NEUTRAL_RAMP, len(word_sources)))
    ]
    for handles, anchor, align in (
        (hue_handles, (0.5, 1), "upper right"),
        (shade_handles, (0.52, 1), "upper left"),
    ):
        legend = fig.legend(
            handles=handles,
            loc=align,
            bbox_to_anchor=anchor,
            ncols=len(handles),
            frameon=False,
            fontsize=9,
            labelcolor=TEXT_SECONDARY,
            handlelength=1.2,
            handleheight=1.2,
        )
        fig.add_artist(legend)


def pct_diff(df: pd.DataFrame, real: str = "Real", fake: str = "Fake") -> pd.DataFrame:
    """Reduces the two word sources of each (model_pair, sim_type) to one signed number.

    Returns a table with the columns ['model_pair', 'sim_type', 'pct_diff'],
    where `pct_diff` is the percentage difference of the `real` aggregate from
    the `fake` one, `100 * (real - fake) / abs(fake)`. A positive value means
    the pair agrees more on real words than on fake ones. The denominator is
    taken as an absolute value so that the sign comes from the numerator alone,
    which matters for the metrics that can go negative (e.g. cosine).

    Rows whose word_source is neither `real` nor `fake` are ignored, and pairs
    missing either side come back as NaN rather than being dropped, so the
    caller can see the gap.
    """

    missing = [field for field in FIELDS if field not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} (expected {FIELDS}).")

    sources = set(df.word_source.unique())
    if not {real, fake} <= sources:
        raise ValueError(
            f"word_source must contain both '{real}' and '{fake}' (found {sorted(sources)})."
        )

    df = df[df.word_source.isin([real, fake])]
    cell = ["model_pair", "sim_type", "word_source"]
    duplicates = df[df.duplicated(cell)]
    if not duplicates.empty:
        raise ValueError(
            "Expected one row per (model_pair, sim_type, word_source); duplicates: "
            f"{duplicates[cell].head().to_dict('records')}"
        )

    wide = df.pivot(
        index=["model_pair", "sim_type"], columns="word_source", values="mean_sim"
    ).reset_index()
    return wide.assign(
        pct_diff=100 * (wide[real] - wide[fake]) / wide[fake].abs()
    )[["model_pair", "sim_type", "pct_diff"]]


def plot_real_fake_diff(
        df: pd.DataFrame,
        sim_types: list | None = None,
        models: list | None = None,
        real: str = "Real",
        fake: str = "Fake",
        value_labels: bool = True,
        vmax: float | None = None,
        ncols: int = 3,
        subplot_size: float = 3.6,
        title: str | None = None,
    ) -> plt.Figure:
    """Draws the real-vs-fake percentage difference as a model x model heatmap.

    One panel per `sim_type`; within a panel, the cell at (row model, column
    model) is that model pair's `pct_diff` (see `pct_diff`). Only the lower
    triangle is drawn, since a pair is unordered and the full matrix would show
    every cell twice. The diagonal and any pair absent from `df` are left blank.

    Color is diverging about zero on a single scale shared by every panel, so
    cells are comparable across metrics: blue where the pair agrees more on fake
    words, red where it agrees more on real ones, neutral where the two match.
    `vmax` sets the end of the scale, defaulting to the largest difference
    present; values beyond it are clipped to the arm's darkest step.

    `models` sets the axis order, defaulting to the order of MODEL_LABEL_MAP
    restricted to the models that appear in `df`. `value_labels` prints each
    cell's percentage, which keeps the figure readable without relying on color.
    """

    diffs = pct_diff(df, real=real, fake=fake)
    diffs = diffs.assign(members=[pair_members(pair) for pair in diffs.model_pair])

    sim_types = sorted(diffs.sim_type.unique()) if sim_types is None else list(sim_types)
    if models is None:
        present = {model for members in diffs.members for model in members}
        models = [model for model in MODEL_LABEL_MAP if model in present]
        models += sorted(present - set(models))
        print(models)
    else:
        models = [str(model) for model in models]
    if len(models) < 2:
        raise ValueError(f"At least 2 models are needed for a pairwise heatmap (got {models}).")

    # cells[sim_type] is a lower-triangular matrix over `models`; everything not
    # filled in below (the diagonal, the upper triangle, absent pairs) stays NaN
    # and is masked out when drawn.
    index = {model: i for i, model in enumerate(models)}
    cells = {
        sim_type: np.full((len(models), len(models)), np.nan) for sim_type in sim_types
    }
    for row in diffs.itertuples():
        if row.sim_type not in cells:
            continue
        model_a, model_b = row.members
        if model_a in index and model_b in index:
            i, j = sorted((index[model_a], index[model_b]), reverse=True)
            cells[row.sim_type][i, j] = row.pct_diff

    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in cells.values()])
    if finite.size == 0:
        raise ValueError("No model pair has a difference to plot.")
    if vmax is None:
        vmax = float(np.abs(finite).max())
    if vmax <= 0:
        # Every pair scores the same on both sources, so there is no polarity to
        # scale to. An arbitrary but readable width keeps TwoSlopeNorm valid and
        # renders the whole heatmap at the neutral midpoint, which is the point.
        vmax = 1.0
    cmap, norm = _diverging_scale(vmax)

    nrows = -(-len(sim_types) // ncols)
    ncols = min(ncols, len(sim_types))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(subplot_size * ncols + 1.0, subplot_size * nrows),
        squeeze=False,
        # The cells are square and the row labels sit inside the panel spacing,
        # so the gaps are left to the layout engine rather than fixed here.
        layout="constrained",
    )
    flat_axes = axes.flatten()
    # The first row and the last column of a lower triangle are empty by
    # construction, so they are dropped rather than drawn as blank strips.
    row_labels = [model_label(model) for model in models[1:]]
    col_labels = [model_label(model) for model in models[:-1]]

    for ax, sim_type in zip(flat_axes, sim_types):
        matrix = cells[sim_type][1:, :-1]
        mesh = ax.imshow(
            np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, interpolation="nearest"
        )
        # A surface-colored grid on the cell edges gives neighbouring fills the
        # 2px separation that keeps two similar shades from merging.
        ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

        if value_labels:
            _label_cells(ax, matrix, mesh)

        ax.set_title(sim_type, fontsize=11, color=TEXT_PRIMARY, pad=8)
        ax.set_xticks(range(len(col_labels)), col_labels, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
        ax.tick_params(length=0, colors=TEXT_SECONDARY)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for ax in flat_axes[len(sim_types):]:
        ax.set_visible(False)

    colorbar = fig.colorbar(mesh, ax=axes, fraction=0.04, pad=0.03, extend="both")
    colorbar.set_label(
        f"{real} vs {fake}: % difference in mean similarity",
        fontsize=9,
        color=TEXT_SECONDARY,
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8, colors=TEXT_SECONDARY)
    if title:
        fig.suptitle(title, fontsize=13, color=TEXT_PRIMARY, x=0.01, ha="left")
    return fig


def _diverging_scale(vmax: float) -> tuple[LinearSegmentedColormap, TwoSlopeNorm]:
    """Builds the diverging colormap and the norm that centres it on zero."""

    negative, midpoint, positive = DIVERGING_RAMP
    cmap = LinearSegmentedColormap.from_list(
        "real_fake_diff", [*negative, midpoint, *positive]
    )
    cmap = cmap.with_extremes(bad=SURFACE, under=negative[0], over=positive[-1])
    return cmap, TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)


def _label_cells(ax: plt.Axes, matrix: np.ndarray, mesh):
    """Prints each cell's percentage, inked for contrast against its own fill."""

    for (i, j), value in np.ndenumerate(matrix):
        if not np.isfinite(value):
            continue
        # The ramp's outer steps are dark enough that primary ink disappears on
        # them, so the label flips to the surface color over a dark fill.
        red, green, blue = mesh.cmap(mesh.norm(value))[:3]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        ax.text(
            j,
            i,
            f"{-value:+.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color=TEXT_PRIMARY if luminance > 0.55 else SURFACE,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot mean cross-model similarity.")
    parser.add_argument("input_path", help="Path to a CSV with columns model_pair, sim_type, word_source, mean_sim.")
    parser.add_argument(
        "--plot",
        choices=["bars", "diff"],
        default="bars",
        help="'bars': mean similarity by sim type, faceted by model pair. "
             "'diff': real-vs-fake %% difference as a model x model heatmap per sim type.",
    )
    parser.add_argument("--real", default="Real", help="word_source taken as the real words (--plot diff).")
    parser.add_argument("--fake", default="Fake", help="word_source taken as the fake words (--plot diff).")
    parser.add_argument("--output_path", help="Path to save the figure. Shown interactively if omitted.")
    parser.add_argument("--title", help="Figure title.")
    parser.add_argument("--ncols", type=int, default=3, help="Number of subplots per row.")
    parser.add_argument("--no_value_labels", action="store_true", help="Hide the value printed on each bar or cell.")
    parser.add_argument("--dpi", type=int, default=200, help="Resolution of the saved figure.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    if args.plot == "diff":
        fig = plot_real_fake_diff(
            df,
            real=args.real,
            fake=args.fake,
            value_labels=not args.no_value_labels,
            ncols=args.ncols,
            title=args.title,
        )
    else:
        fig = plot_mean_sim(
            df,
            value_labels=not args.no_value_labels,
            ncols=args.ncols,
            title=args.title,
        )
    if args.output_path:
        fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Figure saved to {args.output_path}")
    else:
        plt.show()
