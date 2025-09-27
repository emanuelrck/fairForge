import pandas as pd
import io

class DataReader:
    @staticmethod
    def read_data(file, delimiter=None, header=True, column_names=None):
        if isinstance(file, io.BytesIO) or isinstance(file, io.StringIO):
            file_name = file.name.lower()  
            file.seek(0)  
        else:
            file_name = file.lower()

        if file_name.endswith('.csv'):
            return pd.read_csv(file, delimiter=delimiter, header=0 if header else None, names=column_names)
        elif file_name.endswith('.json'):
            return pd.read_json(file)
        elif file_name.endswith('.xlsx'):
            return pd.read_excel(file, engine='openpyxl')
        elif file_name.endswith('.data'):
            return pd.read_csv(file, delimiter=delimiter, header=0 if header else None, names=column_names)
        else:
            raise ValueError("Unsupported file format! Use CSV, JSON, Excel, or .data.")
