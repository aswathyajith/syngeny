import itertools
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

AGGREGATORS = {
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "std": np.std,
    "var": np.var,
}


def _ngram_counts(text: str, n: int, level: str, lowercase: bool) -> Counter:
    """Counts the n-grams of a string over characters or whitespace-delimited words."""

    if lowercase:
        text = text.lower()
    tokens = text.split() if level == "word" else list(text)
    if n < 1:
        raise ValueError(f"n must be at least 1 (got n={n}).")
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


class TextSimilarity:
    """Measures similarity between texts under a configurable similarity metric.

    The metric is resolved once (see `sim_metric_setter`) so that any heavy
    resources, e.g. an embedding model, are loaded a single time and reused by
    `compute_pairwise_sim` and `aggregate_similarity`.
    """

    def __init__(self, metric: str | Callable = "embedding_cosine", **metric_kwargs):
        self.sim_metric_setter(metric, **metric_kwargs)

    def sim_metric_setter(self, metric: str | Callable = "embedding_cosine", **metric_kwargs):
        """Sets the core function that measures similarity between two strings.

        `metric` is either the name of a built-in metric ("embedding_cosine" or
        "ngram_overlap") or a user-defined callable with the signature
        `fn(text_a, text_b, **metric_kwargs) -> float`. Any `metric_kwargs` are
        passed to the metric: see `_set_embedding_cosine` and
        `_set_ngram_overlap` for the options of the built-in metrics.
        """

        self.metric_kwargs = metric_kwargs

        if callable(metric):
            self.metric_name = getattr(metric, "__name__", "custom")
            self.sim_fn = lambda text_a, text_b: metric(text_a, text_b, **metric_kwargs)
        elif metric == "embedding_cosine":
            self.metric_name = metric
            self.sim_fn = self._set_embedding_cosine(**metric_kwargs)
        elif metric == "ngram_overlap":
            self.metric_name = metric
            self.sim_fn = self._set_ngram_overlap(**metric_kwargs)
        else:
            raise ValueError(
                f"Unknown metric '{metric}' (built-in metrics are 'embedding_cosine' "
                "and 'ngram_overlap'; a callable can be passed instead)."
            )

        return self.sim_fn

    def _set_embedding_cosine(
            self,
            model_name: str = DEFAULT_EMBEDDING_MODEL,
            cache_embeddings: bool = True,
            rescale: bool = False,
            model_kwargs: dict | None = None,
            encode_kwargs: dict | None = None,
        ) -> Callable[[str, str], float]:
        """Builds a cosine similarity metric over sentence embeddings.

        The embedding model is loaded once here. Embeddings are cached per
        string by default, so repeated comparisons of the same text do not
        re-run the encoder.

        Cosine similarity ranges over [-1, 1]. Setting `rescale` maps it to
        [0, 1] via `(1 + cos) / 2`, which puts it on the same scale as
        `ngram_overlap` so that the two metrics can be compared or averaged.
        A degenerate (zero-norm) embedding gives 0.0, the floor of the scale
        under `rescale` and an orthogonal pair without it.
        """

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The 'embedding_cosine' metric requires sentence-transformers "
                "(pip install sentence-transformers)."
            ) from e

        model = SentenceTransformer(model_name, **(model_kwargs or {}))
        encode_kwargs = {"convert_to_numpy": True, **(encode_kwargs or {})}
        self.embedding_model = model
        self.embedding_cache = {} if cache_embeddings else None

        def embed(text: str) -> np.ndarray:
            if self.embedding_cache is not None and text in self.embedding_cache:
                return self.embedding_cache[text]
            embedding = np.asarray(model.encode(text, **encode_kwargs), dtype=float)
            if self.embedding_cache is not None:
                self.embedding_cache[text] = embedding
            return embedding

        def embedding_cosine(text_a: str, text_b: str) -> float:
            a, b = embed(text_a), embed(text_b)
            norms = np.linalg.norm(a) * np.linalg.norm(b)
            if norms == 0:
                return 0.0
            cosine = float(np.dot(a, b) / norms)
            return (1 + cosine) / 2 if rescale else cosine

        return embedding_cosine

    def _set_ngram_overlap(
            self,
            n: int = 2,
            level: str = "char",
            measure: str = "jaccard",
            lowercase: bool = True,
        ) -> Callable[[str, str], float]:
        """Builds an n-gram overlap metric.

        `level` is "char" or "word", and `measure` is "jaccard" (intersection
        over union), "dice" (harmonic overlap) or "containment" (intersection
        over the size of the shorter text). Counts are treated as multisets, so
        repeated n-grams contribute more than once.
        """

        if level not in ("char", "word"):
            raise ValueError(f"level must be 'char' or 'word' (got '{level}').")
        if measure not in ("jaccard", "dice", "containment"):
            raise ValueError(
                f"measure must be 'jaccard', 'dice' or 'containment' (got '{measure}')."
            )

        def ngram_overlap(text_a: str, text_b: str) -> float:
            counts_a = _ngram_counts(text_a, n, level, lowercase)
            counts_b = _ngram_counts(text_b, n, level, lowercase)
            total_a, total_b = sum(counts_a.values()), sum(counts_b.values())
            if total_a == 0 or total_b == 0:
                return 0.0

            intersection = sum((counts_a & counts_b).values())
            if measure == "jaccard":
                return intersection / sum((counts_a | counts_b).values())
            if measure == "dice":
                return 2 * intersection / (total_a + total_b)
            return intersection / min(total_a, total_b)

        return ngram_overlap

    def compute_pairwise_sim(self, text_a: str, text_b: str) -> float:
        """Returns the similarity between two strings under the current metric."""

        return self.sim_fn(text_a, text_b)

    def compute_pairwise_sim_df(
            self,
            df: pd.DataFrame,
            key_col: str | None = None,
            model_cols: Sequence[str] | None = None,
            skip_missing: bool = True,
        ) -> dict:
        """Computes the pairwise similarities across models for every row of `df`.

        `df` holds one key column (the word, by default the first column) and
        one column of model outputs per model. Returns a
        {frozenset({model_a, model_b}): {key: similarity}} mapping, with no
        aggregation over the keys. Pairs with a missing output on either side
        are skipped when `skip_missing` is True, and raise otherwise.
        """

        key_col = df.columns[0] if key_col is None else key_col
        model_cols = (
            [col for col in df.columns if col != key_col]
            if model_cols is None
            else list(model_cols)
        )
        if len(model_cols) < 2:
            raise ValueError(
                f"At least 2 model columns are needed (got {model_cols})."
            )

        duplicates = df[key_col].duplicated()
        if duplicates.any():
            raise ValueError(
                f"Column '{key_col}' must be unique to key the output "
                f"(duplicates: {sorted(df.loc[duplicates, key_col].unique())})."
            )

        results = {
            frozenset(pair): {} for pair in itertools.combinations(model_cols, 2)
        }
        for _, row in df.iterrows():
            for model_a, model_b in itertools.combinations(model_cols, 2):
                text_a, text_b = row[model_a], row[model_b]
                if pd.isna(text_a) or pd.isna(text_b):
                    if skip_missing:
                        continue
                    raise ValueError(
                        f"Missing output for '{row[key_col]}' on "
                        f"'{model_a if pd.isna(text_a) else model_b}'."
                    )
                results[frozenset((model_a, model_b))][row[key_col]] = (
                    self.compute_pairwise_sim(text_a, text_b)
                )
        return results

    def aggregate_similarity(
            self,
            texts: Sequence[str],
            aggregators: str | Callable | Iterable[str | Callable] | dict[str, Callable] = "mean",
            return_pairwise: bool = False,
        ) -> dict:
        """Aggregates the pairwise similarities across a list of `texts`.

        `aggregators` is the name of a built-in aggregator (any of
        `AGGREGATORS`), a callable over a 1-D array of similarities, an iterable
        of either, or a {name: callable} mapping. Returns a {name: value}
        mapping, with the pairwise similarities included under "pairwise" when
        `return_pairwise` is True.
        """

        texts = list(texts)
        if len(texts) < 2:
            raise ValueError(
                f"At least 2 texts are needed to aggregate similarities (got {len(texts)})."
            )

        pairs = list(itertools.combinations(range(len(texts)), 2))
        sims = np.array(
            [self.compute_pairwise_sim(texts[i], texts[j]) for i, j in pairs],
            dtype=float,
        )

        results = {
            name: float(fn(sims)) for name, fn in self._resolve_aggregators(aggregators).items()
        }
        if return_pairwise:
            results["pairwise"] = {
                (texts[i], texts[j]): float(sim) for (i, j), sim in zip(pairs, sims)
            }
        return results

    def _resolve_aggregators(
            self,
            aggregators: str | Callable | Iterable[str | Callable] | dict[str, Callable],
        ) -> dict[str, Callable]:
        """Normalizes the `aggregators` argument into a {name: callable} mapping."""

        if isinstance(aggregators, dict):
            return dict(aggregators)
        if isinstance(aggregators, str) or callable(aggregators):
            aggregators = [aggregators]

        resolved = {}
        for aggregator in aggregators:
            if callable(aggregator):
                resolved[getattr(aggregator, "__name__", f"aggregator_{len(resolved)}")] = aggregator
            elif aggregator in AGGREGATORS:
                resolved[aggregator] = AGGREGATORS[aggregator]
            else:
                raise ValueError(
                    f"Unknown aggregator '{aggregator}' (built-in aggregators are "
                    f"{sorted(AGGREGATORS)}; a callable can be passed instead)."
                )
        return resolved
