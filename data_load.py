import argparse
import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file using pandas.
    Parameters:
    file_path (str): Path to the CSV file.
    Returns:
    DataFrame: Loaded pandas DataFrame with the first column as the index.
    """
    try:
        dataframe = pd.read_csv(file_path, index_col=[0])
        return dataframe
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a CSV file as a DataFrame.")
    parser.add_argument("file_path", help="The path to the CSV file")
    args = parser.parse_args()

    data = load_csv(args.file_path)

    # Print basic info for demonstration
    print(data.head())
    print(f"DataFrame loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
