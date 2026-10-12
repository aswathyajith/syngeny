"""Builds the table behind the cross-temperature similarity heatmaps.

`build_temperature_sim_df` takes the wide per-model table written to
processed_results/<sample>/<family>.jsonl (columns: word, temperature, seed, and
one column of generated meanings per model) and returns one long table with the
similarity between the meanings generated at every pair of temperatures.
"""

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from evals.text_similarity import TextSimilarity

KEY_COLS = ["word", "temperature", "seed"]


def build_temperature_sim_df(
        df: pd.DataFrame | str | Path,
        text_sim: TextSimilarity | None = None,
        model_cols: list[str] | None = None,
        same_seed_only: bool = False,
    ) -> pd.DataFrame:
    """Returns the similarity between meanings at each temperature pair.

    `df` is the wide table (or the path to its .jsonl). For every model and
    word, meanings are compared with `text_sim` (cosine over sentence
    embeddings by default) in two ways:

    - across temperatures (temp_a < temp_b), over every combination of seeds
      (seed_a, seed_b), so each cell averages independent samples;
    - across seeds (seed_a < seed_b), within the same temperature
      (temp_a == temp_b), which gives the heatmap diagonal.

    With `same_seed_only=True`, cross-temperature pairs are restricted to the same
    seed (seed_a == seed_b). Use this only for runs that shared one seed across
    temperatures, where the pairing is meaningful.

    Pairs with a missing meaning on either side are skipped.

    Output columns: ['model', 'word', 'temp_a', 'temp_b', 'seed_a', 'seed_b',
    'sim']. Off-diagonal pairs are in one orientation only, so a full symmetric
    heatmap needs both.
    """

    if not isinstance(df, pd.DataFrame):
        df = pd.read_json(df, lines=True)
    text_sim = TextSimilarity("embedding_cosine") if text_sim is None else text_sim
    model_cols = [c for c in df.columns if c not in KEY_COLS] if model_cols is None else list(model_cols)

    duplicates = df.duplicated(KEY_COLS)
    if duplicates.any():
        raise ValueError(f"Rows must be unique per {KEY_COLS} (found {duplicates.sum()} duplicates).")

    rows = []
    for word, group in df.groupby("word"):
        cells = group.set_index(["temperature", "seed"])
        temps, seeds = sorted(group["temperature"].unique()), sorted(group["seed"].unique())
        seed_pairs = [(s, s) for s in seeds] if same_seed_only else list(itertools.product(seeds, seeds))
        pairs = [(ta, tb, sa, sb) for ta, tb in itertools.combinations(temps, 2) for sa, sb in seed_pairs]
        pairs += [(t, t, sa, sb) for t in temps for sa, sb in itertools.combinations(seeds, 2)]
        for model in model_cols:
            for temp_a, temp_b, seed_a, seed_b in pairs:
                try:
                    text_a, text_b = cells.at[(temp_a, seed_a), model], cells.at[(temp_b, seed_b), model]
                except KeyError:
                    continue
                if pd.isna(text_a) or pd.isna(text_b):
                    continue
                rows.append({
                    "model": model, "word": word,
                    "temp_a": temp_a, "temp_b": temp_b,
                    "seed_a": seed_a, "seed_b": seed_b,
                    "sim": text_sim.compute_pairwise_sim(text_a, text_b),
                })
    return pd.DataFrame(rows, columns=["model", "word", "temp_a", "temp_b", "seed_a", "seed_b", "sim"])
