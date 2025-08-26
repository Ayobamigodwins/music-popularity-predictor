import pandas as pd
import glob
import os

def load_and_clean_data(data_path="data/dataset-of-*.csv"):
    # Load all CSV files
    all_files = glob.glob(data_path)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found with pattern: {data_path}")

    # Combine datasets
    df_list = [pd.read_csv(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows with missing target/popularity
    if "target" in df.columns:
        df.dropna(subset=["target"], inplace=True)
        df.rename(columns={"target": "popularity"}, inplace=True)
    elif "popularity" in df.columns:
        df.dropna(subset=["popularity"], inplace=True)
    else:
        raise KeyError("No target or popularity column found in dataset")

    # Select only relevant features + target
    features = [
        "acousticness", "danceability", "energy", "instrumentalness",
        "liveness", "speechiness", "valence", "tempo", "loudness", "popularity"
    ]
    df = df[features]

    # Feature engineering
    df["dance_energy"] = df["danceability"] * df["energy"]
    df["valence_acoustic"] = df["valence"] * df["acousticness"]

    loudness_min = df["loudness"].min()
    loudness_max = df["loudness"].max()
    df["loudness_norm"] = (df["loudness"] - loudness_min) / (loudness_max - loudness_min)

    # Save processed file
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "cleaned_music_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned dataset to {output_path}")
    print(f"Dataset shape: {df.shape}")

    # Optionally save loudness range for Streamlit use
    loudness_info = {"min": loudness_min, "max": loudness_max}
    pd.Series(loudness_info).to_json(os.path.join(processed_dir, "loudness_info.json"))

    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())