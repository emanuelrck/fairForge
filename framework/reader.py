import pandas as pd
import io

class DataReader:
    @staticmethod
    def read_data(file, delimiter=None, header=True, column_names=None):
        try:
            if isinstance(file, io.BytesIO) or isinstance(file, io.StringIO):
                file_name = file.name.lower()
                file.seek(0)
            elif hasattr(file, 'name'):
                file_name = file.name.lower()
            else:
                file_name = str(file).lower()

            if file_name.endswith('.csv'):
                df = pd.read_csv(file, delimiter=delimiter, header=0 if header else None, names=column_names)
            elif file_name.endswith('.json'):
                df = pd.read_json(file)
            elif file_name.endswith('.xlsx'):
                df = pd.read_excel(file, engine='openpyxl')
            elif file_name.endswith('.data'):
                df = pd.read_csv(file, delimiter=delimiter, header=0 if header else None, names=column_names)
            else:
                raise ValueError("Unsupported file format! Use CSV, JSON, Excel, or .data.")

            if df.empty or len(df.columns) == 0:
                raise ValueError("The uploaded file is empty or has no columns.")
            return df
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
