import argparse
import pandas as pd
def load_csv(file_path):
    """
    Load a CSV file using pandas.
    Parameters:
    file_path (str): Path to the CSV file.
    Returns:
    DataFrame: Loaded pandas DataFrame with the first column as the index.
    """
    # Typically, the first column is used as the index.
    dataframe = pd.read_csv(file_path, index_col=[0])
    return dataframe
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a CSV file as a DataFrame.")
    parser.add_argument("file_path", help="The path to the CSV file")
    args = parser.parse_args()
    # Call the load_csv function with the parsed file path
    data = load_csv(args.file_path)
    # For demonstration, you print the DataFrame
    print(data)
