from data_load import load_csv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import argparse


def make_pipeline(file_path, X_cols, cardinality_threshold):
    """
    Loads CSV into a DataFrame, separates categorical from numeric columns,
    and defines a preprocessing pipeline for the DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        X_cols (list): List of feature columns.
        cardinality_threshold (int): Max number of unique values
        for a column to be considered categorical.

    Returns:
        pipe (Pipeline): Data preprocessing and model pipeline.
    """
    df = load_csv(file_path)
    X = df[X_cols]
    # Separate categorical and numerical columns
    cat_cols = [
        cname for cname in X.columns
        if (X[cname].dtype == "object" or X[cname].dtype in ["int64", "float64"])
           and X[cname].nunique() < cardinality_threshold
    ]
    num_cols = [
        cname for cname in X.columns
        if X[cname].dtype in ['int64', 'float64'] and cname not in cat_cols
    ]
    # Define transformations for categorical and numerical features
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('one_hot', OneHotEncoder(handle_unknown='ignore'))
    ])
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
    # Create a pipeline with the preprocessor and a RandomForest model
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('forest', RandomForestRegressor())
    ])
    return pipe,df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a preprocessing and model pipeline.")
    parser.add_argument('file_path', type=str, help="Path to the CSV file.")
    parser.add_argument('--X_cols', nargs='+', required=True, help="List of feature columns.")
    parser.add_argument('--cardinality_threshold', type=int, default=10,
                        help="Max unique values for a column to be categorical.")
    args = parser.parse_args()

    # Unpack the pipeline and DataFrame returned by make_pipeline
    pipeline, df = make_pipeline(args.file_path, args.X_cols, args.cardinality_threshold)

    print("Pipeline created successfully.")
    print("Loaded DataFrame preview:")
    print(df.head())
