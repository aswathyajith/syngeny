"""Merge the three source word datasets into the unified `nonce_words` table.

The schema produced here is defined in configs/schemas/inputs.yml: one row per word,
pooling real-but-rare words, coined words, and pseudoword soundalikes of the rare words.
"""

import os
import argparse
import datetime
import pandas as pd

# Column order of the nonce_words table, per configs/schemas/inputs.yml.
FIELDS = [
    "word_id",
    "word_string",
    "word_source",
    "word_exists",
    "word_soundalike",
    "word_usage",
    "word_meaning",
]

# Filenames of the source datasets, relative to the stimuli directory.
RARE_WORDS_FILE = "rare-words.csv"
COINED_WORDS_FILE = "coined-words.csv"
SOUNDALIKES_FILE = "rare-words-soundalikes.csv"


def word_id(word_source: str, n: int) -> str:
    """Formats a word id as "<word_source>-<n>", with n zero-padded to 4 digits."""

    return f"{word_source}-{n:04d}"


class StimuliMerger:
    def __init__(self, stimuli_path: str):
        self.stimuli_path = stimuli_path

    def merge_stimuli(self) -> pd.DataFrame:
        """
        Merges the source stimuli CSV files into a single DataFrame following the
        nonce_words schema, with one row per word and `word_usage` holding a list
        of example sentences.
        """

        merged_df = pd.concat(
            [self._rare_words(), self._coined_words(), self._soundalikes()],
            ignore_index=True,
        )
        self.validate(merged_df)
        return merged_df

    def _read(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.stimuli_path, filename))

    def _rare_words(self) -> pd.DataFrame:
        """Real English words: the five example sentences are grouped into `word_usage`."""

        df = self._read(RARE_WORDS_FILE).sort_values(["word_id", "sentence_id"])
        words = df.groupby("word_id", as_index=False).agg(
            word=("word", "first"),
            definition=("definition", "first"),
            sentences=("sentence", list),
        )
        return pd.DataFrame({
            "word_id": [word_id("rare", n) for n in words.word_id],
            "word_string": words.word,
            "word_source": "rare",
            "word_exists": True,
            "word_soundalike": None,
            "word_usage": words.sentences,
            "word_meaning": words.definition,
        })

    def _coined_words(self) -> pd.DataFrame:
        """Invented words: the single `example` becomes a one-element `word_usage`."""

        df = self._read(COINED_WORDS_FILE)
        return pd.DataFrame({
            "word_id": [word_id("coined", n) for n in df.id],
            "word_string": df.word,
            "word_source": "coined",
            "word_exists": False,
            "word_soundalike": None,
            "word_usage": [[example] for example in df.example],
            "word_meaning": df.meaning,
        })

    def _soundalikes(self) -> pd.DataFrame:
        """
        Pseudowords perturbed from a rare word. Usage and meaning are withheld:
        recovering a meaning for these is the task, and the parent word's definition
        would leak the answer.
        """

        df = self._read(SOUNDALIKES_FILE)
        return pd.DataFrame({
            "word_id": [
                f"{word_id('soundalike', parent)}-{variant}"
                for parent, variant in zip(df.word_id, df.pseudoword_id)
            ],
            "word_string": df.pseudoword,
            "word_source": "soundalike",
            "word_exists": False,
            "word_soundalike": [word_id("rare", parent) for parent in df.word_id],
            "word_usage": None,
            "word_meaning": None,
        })

    @staticmethod
    def validate(df: pd.DataFrame):
        """Checks a merged table against the constraints in configs/schemas/inputs.yml."""

        errors = []
        if list(df.columns) != FIELDS:
            errors.append(f"columns are {list(df.columns)}, expected {FIELDS}")
        for field in ["word_id", "word_string", "word_source", "word_exists"]:
            if df[field].isna().any():
                errors.append(f"{field} is not nullable but has null values")
        for field in ["word_id", "word_string"]:
            duplicates = df[field][df[field].duplicated()].tolist()
            if duplicates:
                errors.append(f"{field} must be unique, found duplicates: {duplicates[:5]}")

        is_soundalike = df.word_source == "soundalike"
        if not (df.word_soundalike.notna() == is_soundalike).all():
            errors.append('word_soundalike must be non-null if and only if word_source == "soundalike"')
        unknown_parents = sorted(set(df.word_soundalike.dropna()) - set(df.word_id))
        if unknown_parents:
            errors.append(f"word_soundalike must match an existing word_id, missing: {unknown_parents[:5]}")
        if not (df.word_exists == (df.word_source == "rare")).all():
            errors.append('word_exists must be true if and only if word_source == "rare"')
        if df.loc[is_soundalike, ["word_usage", "word_meaning"]].notna().any().any():
            errors.append("soundalikes must have null word_usage and word_meaning")

        if errors:
            raise ValueError("Merged stimuli do not match the schema:\n- " + "\n- ".join(errors))


def explode_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens the table for CSV serialization: the `word_usage` list becomes one row per
    sentence, with an added `usage_id` numbering the sentences within a word.
    """

    exploded = df.explode("word_usage", ignore_index=True)
    usage_id = exploded.groupby("word_id").cumcount() + 1
    exploded.insert(
        exploded.columns.get_loc("word_usage") + 1,
        "usage_id",
        usage_id.where(exploded.word_usage.notna()).astype("Int64"),
    )
    return exploded


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge stimuli CSV files into a single DataFrame.")
    parser.add_argument("--stimuli_path", default="data/datasets", help="Path to the directory containing the source stimuli CSV files.")
    parser.add_argument("--output_path", required=False, help="Path to save the merged DataFrame as a CSV file.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode to print additional information. Dataset will be saved with a timestamp in the filename.")
    parser.add_argument("--overwrite", action='store_true', help="Enable overwrite mode to overwrite the existing output_path file.")
    args = parser.parse_args()

    merger = StimuliMerger(stimuli_path=args.stimuli_path)
    merged_df = merger.merge_stimuli()

    # save the merged DataFrame to a CSV file if output_path is provided
    if args.output_path:
        output_file = args.output_path
        if args.debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            stem, ext = os.path.splitext(output_file)
            output_file = f"{stem}_{timestamp}{ext}"
        if os.path.exists(output_file) and not args.overwrite:
            raise FileExistsError(f"{output_file} already exists, pass --overwrite to replace it.")
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        explode_usage(merged_df).to_csv(output_file, index=False)
        print(f"Merged DataFrame saved to {output_file}")
    print(merged_df)
