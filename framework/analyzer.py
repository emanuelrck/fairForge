import pandas as pd
import matplotlib.pyplot as plt
import os

class DataAnalyzer:
    @staticmethod
    def check_class_imbalance(df, target_column, threshold=0.1, file_path=None):
        """
        Checks for class imbalance in a given target column and optionally writes to a CSV file.

        Args:
            df (pd.DataFrame): The dataset.
            target_column (str): The column to analyze.
            threshold (float): The minimum percentage below which a class is considered imbalanced.
            file_path (str, optional): If provided, saves the analysis to this file.

        Returns:
            dict: Summary of the imbalance analysis.
        """
        class_counts = df[target_column].value_counts(normalize=True) * 100  # Convert to percentage
        imbalance_report = {
            "target_column": target_column,
            "class_distribution": class_counts.to_dict(),
            "imbalanced_classes": []
        }

        # Identify imbalanced classes
        imbalanced_classes = class_counts[class_counts < (threshold * 100)].index.tolist()
        imbalance_report["imbalanced_classes"] = imbalanced_classes

        # Print to console
        print(f"\n=== Class Distribution for {target_column} ===")
        print(class_counts)

        if imbalanced_classes:
            print(f"⚠️ Warning: Imbalanced classes detected in '{target_column}': {imbalanced_classes}")
        else:
            print(f"✅ No significant imbalance detected in '{target_column}'")

        # Save to CSV if file_path is provided
        if file_path:
            imbalance_report_df = pd.DataFrame([imbalance_report])
            imbalance_report_df.to_csv(file_path, mode='a', header=not pd.io.common.file_exists(file_path), index=False, sep=';')

        return imbalance_report

    @staticmethod
    def analyze_multiple_targets(df, target_columns, threshold=0.1, file_path=None):
        """
        Runs imbalance analysis on multiple target columns.

        Args:
            df (pd.DataFrame): The dataset.
            target_columns (list): List of columns to analyze.
            threshold (float): The imbalance detection threshold.
            file_path (str, optional): If provided, saves the analysis to this file.

        Returns:
            list: List of imbalance reports for each target column.
        """
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            
        reports = []
        for column in target_columns:
            if column in df.columns:
                report = DataAnalyzer.check_class_imbalance(df, column, threshold, file_path)
                reports.append(report)
            else:
                msg = f"⚠️ Warning: Column '{column}' not found in dataset.\n"
                print(msg)
                if file_path:
                    # Save the warning to CSV
                    warning_report_df = pd.DataFrame([{"target_column": column, "class_distribution": "N/A", "imbalanced_classes": msg}])
                    warning_report_df.to_csv(file_path, mode='w', header=not pd.io.common.file_exists(file_path), index=False, sep=';')

        return reports

    @staticmethod
    def plot_class_distribution(df, target_columns):
        """Generates and displays class distribution plots for multiple columns."""
        for column in target_columns:
            if column in df.columns:
                plt.figure(figsize=(6, 4))  # Create a new figure for each plot
                df[column].value_counts().plot(kind='bar', color=['blue', 'red', 'green', 'purple'])
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title(f"Class Distribution for {column}")
                plt.xticks(rotation=45)
                plt.tight_layout()  # Adjust layout to prevent overlapping text

        plt.show()

    @staticmethod
    def check_missing_values(df, file_path=None, custom_missing_values=None):
        """
        Verifica valores ausentes (NaN + valores comuns de missing data) em cada atributo e gera um relatório.

        Args:
            df (pd.DataFrame): O dataset a ser analisado.
            file_path (str, optional): Se fornecido, salva o relatório neste arquivo.
            custom_missing_values (list, optional): Lista de valores personalizados a serem considerados como missing.

        Returns:
            dict: Dicionário contendo a percentagem de missing values por coluna.
        """
        if custom_missing_values is None:
            custom_missing_values = ["NA", "Na", "na", "nan", "?", "NULL", "None", ""]  # Lista padrão

        # Conta valores NaN padrão
        missing_counts = df.isnull().sum()

        # Conta valores customizados de missing
        for value in custom_missing_values:
            missing_counts += (df == value).sum()

        total_rows = len(df)  # Número total de linhas
        missing_percent = (missing_counts / total_rows) * 100  # Percentagem

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

        # Print final report
        print("\n===== Missing Values Report =====")
        for index, row in missing_report_df.iterrows():
            print(f"{row['column']}: {row['missing_percentage']:.2f}% missing")

        # Save to CSV if file_path is provided
        if file_path:
            missing_report_df.to_csv(file_path, mode='w', header=not pd.io.common.file_exists(file_path), index=False, sep=';')

        return missing_report_df

