"""Builds the table behind the cross-model similarity heatmaps.

`build_model_sim_df` takes the wide per-model table written to
processed_results/<sample>/<family>.jsonl (columns: word, temperature, seed, and
one column of generated meanings per model) and returns one long table with the
similarity between the meanings that every pair of models generated at one temperature.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from evals.text_similarity import TextSimilarity

KEY_COLS = ["word", "temperature", "seed"]


def load_wide_df(paths: list[str | Path]) -> pd.DataFrame:
    """Joins the per-family wide tables on (word, temperature, seed) into one table with every model's column."""

    dfs = [pd.read_json(path, lines=True) for path in paths]
    merged = dfs[0]
    for path, df in zip(paths[1:], dfs[1:]):
        overlap = (set(merged.columns) & set(df.columns)) - set(KEY_COLS)
        if overlap:
            raise ValueError(f"{path} repeats model columns already loaded: {sorted(overlap)}.")
        merged = merged.merge(df, on=KEY_COLS, how="outer")
    return merged


def build_model_sim_df(
        df: pd.DataFrame | str | Path,
        temperature: float,
        text_sim: TextSimilarity | None = None,
        model_cols: list[str] | None = None,
        same_seed_only: bool = False,
    ) -> pd.DataFrame:
    """Returns the similarity between models' meanings at one temperature.

    `df` is the wide table (or the path to its .jsonl). For every word, meanings
    are compared with `text_sim` (cosine over sentence embeddings by default) in
    two ways:

    - across models (model_a != model_b, in `model_cols` order), over every
      combination of seeds (seed_a, seed_b), so each cell averages independent samples;
    - across seeds (seed_a < seed_b) within the same model (model_a == model_b),
      which gives the heatmap diagonal.

    With `same_seed_only=True`, cross-model pairs are restricted to the same seed
    (seed_a == seed_b). Models are sampled independently, so this is rarely meaningful.

    Pairs with a missing meaning on either side are skipped, so a model with no
    outputs at this temperature does not appear in the result.

    Output columns: ['model_a', 'model_b', 'word', 'seed_a', 'seed_b', 'sim'].
    """

    if not isinstance(df, pd.DataFrame):
        df = pd.read_json(df, lines=True)
    text_sim = TextSimilarity("embedding_cosine") if text_sim is None else text_sim
    model_cols = [c for c in df.columns if c not in KEY_COLS] if model_cols is None else list(model_cols)

    duplicates = df.duplicated(KEY_COLS)
    if duplicates.any():
        raise ValueError(f"Rows must be unique per {KEY_COLS} (found {duplicates.sum()} duplicates).")

    df = df[np.isclose(df["temperature"], temperature)]
    if df.empty:
        raise ValueError(f"No rows at temperature {temperature}.")

    rows = []
    for word, group in df.groupby("word"):
        cells = group.set_index("seed")
        seeds = sorted(group["seed"].unique())
        for model_a, model_b in itertools.combinations_with_replacement(model_cols, 2):
            if model_a == model_b:
                seed_pairs = list(itertools.combinations(seeds, 2))
            elif same_seed_only:
                seed_pairs = [(s, s) for s in seeds]
            else:
                seed_pairs = list(itertools.product(seeds, seeds))
            for seed_a, seed_b in seed_pairs:
                text_a, text_b = cells.at[seed_a, model_a], cells.at[seed_b, model_b]
                if pd.isna(text_a) or pd.isna(text_b):
                    continue
                rows.append({
                    "model_a": model_a, "model_b": model_b, "word": word,
                    "seed_a": seed_a, "seed_b": seed_b,
                    "sim": text_sim.compute_pairwise_sim(text_a, text_b),
                })
    return pd.DataFrame(rows, columns=["model_a", "model_b", "word", "seed_a", "seed_b", "sim"])
