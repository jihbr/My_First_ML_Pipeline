from data_load import load_csv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import argparse
import pandas as pd
from typing import List, Tuple


def make_pipeline(file_path: str, X_cols: List[str]) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Loads CSV into a DataFrame, separates categorical from numeric columns,
    and defines a preprocessing pipeline for the DataFrame.
    Args:
        file_path (str): Path to the CSV file.
        X_cols (List[str]): List of feature columns.
    Returns:
        Tuple[Pipeline, pd.DataFrame]: Data preprocessing and model pipeline, loaded DataFrame.
    """
    # Load the CSV file into a DataFrame
    df = load_csv(file_path)

    # Select only the specified columns from the DataFrame
    X = df[X_cols].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Initialize lists to store categorical and numerical columns
    cat_cols = []
    num_cols = []
    # Loop through each column to classify as categorical or numerical
    for col in X.columns:
        if X[col].dtype == 'bool' or X[col].dtype == 'object':
            # Convert to string in the original df to avoid SettingWithCopyWarning
            df.loc[:, col] = df[col].astype(str)
            cat_cols.append(col)
        elif X[col].dtype in ["int64", "float64"]:
            # Ensure numerical columns are either int or float
            num_cols.append(col)
        else:
            # Log a warning for unsupported types
            print(f"Warning: Column '{col}' has an unsupported type and will be skipped.")
    # Define the transformation pipeline for categorical features
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('one_hot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Define the transformation pipeline for numerical features
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Combine transformations into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # Create the final pipeline with the preprocessor and a RandomForest model
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('forest', RandomForestRegressor())
    ])

    return pipe, df


# If the script is executed directly, this block will handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a preprocessing and model pipeline.")
    parser.add_argument('file_path', type=str, help="Path to the CSV file.")
    parser.add_argument('--X_cols', nargs='+', required=True, help="List of feature columns.")
    args = parser.parse_args()

    # Unpack the pipeline and DataFrame returned by make_pipeline
    pipeline, df = make_pipeline(args.file_path, args.X_cols)

    # Output to confirm successful creation and preview the DataFrame
    print("Pipeline created successfully.")
    print("Loaded DataFrame preview:")
    print(df.head())
