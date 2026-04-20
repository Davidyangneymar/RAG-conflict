from pathlib import Path

import pandas as pd


def load_and_split_data(
    bodies_csv: str | Path,
    stances_csv: str | Path,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FNC-style CSVs, merge by Body ID, and split into train/val DataFrames.

    Returns:
        train_df, val_df with columns: headline, body, label
    """
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    bodies_df = pd.read_csv(bodies_csv)
    stances_df = pd.read_csv(stances_csv)

    merged_df = stances_df.merge(
        bodies_df,
        on="Body ID",
        how="inner",
        validate="many_to_one",
    )

    dataset_df = merged_df[["Headline", "articleBody", "Stance"]].rename(
        columns={
            "Headline": "headline",
            "articleBody": "body",
            "Stance": "label",
        }
    )

    dataset_df = dataset_df.dropna(subset=["headline", "body", "label"])
    dataset_df = dataset_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    split_idx = int(len(dataset_df) * (1 - val_size))
    train_df = dataset_df.iloc[:split_idx].reset_index(drop=True)
    val_df = dataset_df.iloc[split_idx:].reset_index(drop=True)

    return train_df, val_df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    bodies_path = root / "train_bodies.csv"
    stances_path = root / "train_stances.csv"
    train_out_path = root / "train_processed.csv"
    val_out_path = root / "val_processed.csv"

    train_df, val_df = load_and_split_data(
        bodies_csv=bodies_path,
        stances_csv=stances_path,
        val_size=0.1,
        random_state=42,
    )

    train_df.to_csv(train_out_path, index=False)
    val_df.to_csv(val_out_path, index=False)

    print("train shape:", train_df.shape)
    print("val shape:", val_df.shape)
    print("columns:", list(train_df.columns))
    print("train saved to:", train_out_path)
    print("val saved to:", val_out_path)
    print("\ntrain sample:")
    print(train_df.head(10))
