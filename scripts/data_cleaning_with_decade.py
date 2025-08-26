# scripts/data_cleaning_with_decade.py
import pandas as pd
import glob
import os
import re

def infer_decade_from_path(path):
    """
    Extract decade from file name like 'dataset-of-60s.csv' -> '1960s'
    """
    m = re.search(r"dataset-of-(\d{2})s\.csv", os.path.basename(path))
    if not m:
        return "unknown"
    two = m.group(1)
    mapping = {
        "60": "1960s",
        "70": "1970s",
        "80": "1980s",
        "90": "1990s",
        "00": "2000s",
        "10": "2010s",
    }
    return mapping.get(two, "unknown")

def load_and_clean_data(data_path="data/dataset-of-*.csv"):
    # Load all CSV files
    all_files = glob.glob(data_path)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found with pattern: {data_path}")

    # Read, tag decade, stack
    frames = []
    for fp in all_files:
        tmp = pd.read_csv(fp)
        tmp["decade"] = infer_decade_from_path(fp)
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Ensure target -> popularity
    if "target" in df.columns and "popularity" not in df.columns:
        df.rename(columns={"target": "popularity"}, inplace=True)
    if "popularity" not in df.columns:
        raise KeyError("No 'target' or 'popularity' column found in dataset.")
    df.dropna(subset=["popularity"], inplace=True)

    # Keep only relevant features + target + decade
    features = [
        "acousticness", "danceability", "energy", "instrumentalness",
        "liveness", "speechiness", "valence", "tempo", "loudness", "popularity", "decade"
    ]
    df = df[[c for c in features if c in df.columns]]

    # Feature engineering
    df["dance_energy"] = df["danceability"] * df["energy"]
    df["valence_acoustic"] = df["valence"] * df["acousticness"]

    # Loudness min-max normalisation
    loudness_min = df["loudness"].min()
    loudness_max = df["loudness"].max()
    df["loudness_norm"] = (df["loudness"] - loudness_min) / (loudness_max - loudness_min)

    # Save processed file
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "cleaned_music_data_with_decade.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned dataset (with decade) to {output_path}")
    print(f"Dataset shape: {df.shape}")

    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())