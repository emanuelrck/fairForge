import pandas as pd
import matplotlib.pyplot as plt
import os

class DataAnalyzer:
    @staticmethod
    def check_class_imbalance(df, target_column, threshold=0.1, file_path=None):
        
        class_counts = df[target_column].value_counts(normalize=True) * 100 
        imbalance_report = {
            "target_column": target_column,
            "class_distribution": class_counts.to_dict(),
            "imbalanced_classes": []
        }

        imbalanced_classes = class_counts[class_counts < (threshold * 100)].index.tolist()
        imbalance_report["imbalanced_classes"] = imbalanced_classes

        print(f"\n=== Class Distribution for {target_column} ===")
        print(class_counts)

        if imbalanced_classes:
            print(f" Warning: Imbalanced classes detected in '{target_column}': {imbalanced_classes}")
        else:
            print(f" No significant imbalance detected in '{target_column}'")


        return imbalance_report

    @staticmethod
    def analyze_multiple_targets(df, target_columns, threshold=0.1, file_path=None):
        
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            
        reports = []
        for column in target_columns:
            if column in df.columns:
                report = DataAnalyzer.check_class_imbalance(df, column, threshold, file_path)
                reports.append(report)
            else:
                msg = f" Warning: Column '{column}' not found in dataset.\n"
                print(msg)
        return reports

    @staticmethod
    def plot_class_distribution(df, target_columns):
        for column in target_columns:
            if column in df.columns:
                plt.figure(figsize=(6, 4)) 
                df[column].value_counts().plot(kind='bar', color=['blue', 'red', 'green', 'purple'])
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title(f"Class Distribution for {column}")
                plt.xticks(rotation=45)
                plt.tight_layout()  

        plt.show()

    @staticmethod
    def check_missing_values(df, file_path=None, custom_missing_values=None):
        if custom_missing_values is None:
            custom_missing_values = ["NA", "Na", "na", "nan", "?", "NULL", "None", ""]  # Lista padrão

        missing_counts = df.isnull().sum()
        for value in custom_missing_values:
            missing_counts += (df == value).sum()

        total_rows = len(df)
        if total_rows == 0:
            missing_report_df = pd.DataFrame({"column": ["N/A"], "missing_percentage": [0]})
            return missing_report_df
        missing_percent = (missing_counts / total_rows) * 100  

        # Gerar relatório
        missing_report = {"column": [], "missing_percentage": []}
        for column, percent in missing_percent.items():
            if percent > 0:
                missing_report["column"].append(column)
                missing_report["missing_percentage"].append(percent)

        if not missing_report["column"]:
            missing_report["column"].append("N/A")
            missing_report["missing_percentage"].append(0)

        missing_report_df = pd.DataFrame(missing_report)
        missing_report_df = missing_report_df.sort_values(by="missing_percentage", ascending=False)

        print("\n===== Missing Values Report =====")
        for index, row in missing_report_df.iterrows():
            print(f"{row['column']}: {row['missing_percentage']:.2f}% missing")


        return missing_report_df

