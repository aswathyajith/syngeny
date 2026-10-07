
import os
from pathlib import Path
import pandas as pd

def load_data(input_path: str) -> pd.DataFrame:
    """
    Loads the CSV file into a DataFrame.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")
    
    fmt = Path(input_path).suffix.lower()

    if fmt == ".csv":
        return pd.read_csv(input_path)
    elif fmt == ".jsonl":
        return pd.read_json(input_path, lines=True, orient="records")
    else:
        raise ValueError(f"Unsupported file format: {fmt}")

def save_data(df: pd.DataFrame, output_path: str):
    """
    Saves the DataFrame to a CSV file.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fmt = Path(output_path).suffix.lower()

    if fmt == ".csv": 
        df.to_csv(output_path, index=False)
    elif fmt == ".jsonl":
        df.to_json(output_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")

def extract_answer(
        response_text, 
        tag="answer", 
        error_message="Parsing error: answer tags not found in the response."
    ) -> str: 
    """
    Extracts the answer embedded within the model's response text between the tag.
    """

    tag_start = f"<{tag}>"
    tag_end = f"</{tag}>"

    start_idx = response_text.find(tag_start)
    end_idx = response_text.find(tag_end)

    if start_idx == -1 or end_idx == -1:
        return error_message

    return response_text[start_idx + len(tag_start):end_idx].strip()

def append_data(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Appends new data to the existing DataFrame, ensuring no duplicates based on 'word'.
    """

    existing_data = load_data(path) if os.path.exists(path) else pd.DataFrame()
    if (not existing_data.empty and "word" not in existing_data.columns) or "word" not in df.columns:
        raise ValueError("Both DataFrames must contain a 'word' column for duplicate checking.")

    combined_df = pd.concat([existing_data, df]).drop_duplicates(subset="word", keep="last").reset_index(drop=True)

    save_data(combined_df, path)
    return combined_df