import pandas as pd
import io

class DataReader:
    @staticmethod
    def read_data(file, delimiter=None, header=True, column_names=None):
        """
        Reads a dataset from CSV, JSON, Excel, or .data format.

        Args:
            file (str or UploadedFile): Path to the dataset file or a file-like object.
            delimiter (str, optional): Delimiter for .data files (default: None, tries auto-detection).
            header (bool, optional): If True, uses the first row as column names. If False, uses `column_names` instead.
            column_names (list, optional): List of column names if `header` is False.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        # Check if file is a Streamlit UploadedFile (file-like object)
        if isinstance(file, io.BytesIO) or isinstance(file, io.StringIO):
            file_name = file.name.lower()  # Get the file extension
            file.seek(0)  # Ensure reading starts from the beginning
        else:
            file_name = file.lower()

        if file_name.endswith('.csv'):
            return pd.read_csv(file,sep=";")
        elif file_name.endswith('.json'):
            return pd.read_json(file)
        elif file_name.endswith('.xlsx'):
            return pd.read_excel(file, engine='openpyxl')
        elif file_name.endswith('.data'):
            return pd.read_csv(file, delimiter=",", header=0 if header else None, names=column_names)
        else:
            raise ValueError("Unsupported file format! Use CSV, JSON, Excel, or .data.")
