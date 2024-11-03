from make_pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import argparse
from sklearn.pipeline import Pipeline
from typing import List


def train_model(file_path: str, X_cols: List[str], y_col: str, ts: float, random: int) -> Pipeline:
    """
    Defines a preprocessing pipeline for the DataFrame.
    Splits the data into train and test sets.
    Trains the model in the pipeline and stores its metrics in a dictionary.

    Args:
        file_path (str): Path to the CSV file.
        X_cols (List[str]): List of feature columns.
        y_col (str): Target variable column.
        ts (float): Test set size ratio.
        random (int): Random seed for stochastic operations.

    Returns:
        Pipeline: Trained machine learning pipeline with preprocessing and model.
    """
    # Instantiate preprocessing pipeline and load DataFrame
    pipe, df = make_pipeline(file_path, X_cols)

    # Verify that y_col exists in the DataFrame
    if y_col not in df.columns:
        raise ValueError(f"Target column '{y_col}' not found in the data.")

    # Separate the X and y data
    X, y = df[X_cols], df[y_col].astype(int)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=random
    )

    # Fit the model to the training data
    pipe.fit(x_train, y_train)

    # Store model metrics in a dictionary
    model_metrics = {
        "train_data": {
            "score": pipe.score(x_train, y_train),
            "mae": mean_absolute_error(y_train, pipe.predict(x_train)),
        },
        "test_data": {
            "score": pipe.score(x_test, y_test),
            "mae": mean_absolute_error(y_test, pipe.predict(x_test)),
        },
    }
    print("Model Metrics:", model_metrics)

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with preprocessing pipeline.")
    parser.add_argument('file_path', type=str, help="Path to the CSV file.")
    parser.add_argument('--X_cols', nargs='+', required=True, help="List of feature columns.")
    parser.add_argument('--y_col', type=str, required=True, help="Target variable column.")
    parser.add_argument('--ts', type=float, default=0.2, help="Test set size as a ratio.")
    parser.add_argument('--random', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Call train_model with parsed arguments
    train_model(
        file_path=args.file_path,
        X_cols=args.X_cols,
        y_col=args.y_col,
        ts=args.ts,
        random=args.random
    )
