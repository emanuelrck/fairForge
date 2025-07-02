from itertools import combinations
from sklearn.metrics import confusion_matrix

class FairnessEvaluator:
    def __init__(self, df_test, sensitive_columns):
        self.df_test = df_test
        self.sensitive_columns = sensitive_columns

    def evaluate_fairness(self, y_true, y_pred, selected_fairness,model, file_path=None):
        report_lines = ["\n===== Fairness Evaluation Report "+model+"====="]
        dicio_all_fair = {}
        for column in self.sensitive_columns:
            if column not in self.df_test.columns:
                print(column)
                continue
            dicio_all_fair[column] = {}

            report_lines.append(f"\n🔹 Avaliando fairness para: {column}")
            groups = self.df_test[column].unique()

            metrics = {}
            for group in groups:
                group_mask = self.df_test[column] == group
                group_true = y_true[group_mask.to_numpy()]
                group_pred = y_pred[group_mask.to_numpy()]
                
                if len(group_true) == 0:
                    continue
                
                cm = confusion_matrix(group_true, group_pred, labels=[0, 1])
                
                tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, 0)
                
                n = len(group_true)
                metrics[group] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "n": n}
            
            # Gerar todas as combinações de protected e unprotected em ambas as ordens
            for protected, unprotected in combinations(groups, 2):
                for protected_group, unprotected_group in [(protected, unprotected), (unprotected, protected)]:
                    dicio_all_fair[column][(protected_group, unprotected_group)] = {}
                    if protected_group in metrics and unprotected_group in metrics:
                        p, up = metrics[protected_group], metrics[unprotected_group]
                        if "equal_opportunity" in selected_fairness :
                            
                            equal_opportunity = (p["TP"] / (p["FN"] + p["TP"]) if (p["FN"] + p["TP"]) > 0 else 0) - \
                                           (up["TP"] / (up["FN"] + up["TP"]) if (up["FN"] + up["TP"]) > 0 else 0)
                        
                        if "predictive_equality" in selected_fairness :
                            predictive_equality = (p["FP"] / (p["FP"] + p["TN"]) if (p["FP"] + p["TN"]) > 0 else 0) - \
                                              (up["FP"] / (up["FP"] + up["TN"]) if (up["FP"] + up["TN"]) > 0 else 0)
                        if "positive_predictive_parity" in selected_fairness :
                            positive_predictive_parity = (p["TP"] / (p["FP"] + p["TP"]) if (p["FP"] + p["TP"]) > 0 else 0) - \
                                                    (up["TP"] / (up["FP"] + up["TP"]) if (up["FP"] + up["TP"]) > 0 else 0)
                        if "negative_predictive_parity" in selected_fairness :
                            negative_predictive_parity = (p["TN"] / (p["FN"] + p["TN"]) if (p["FN"] + p["TN"]) > 0 else 0) - \
                                                    (up["TN"] / (up["FN"] + up["TN"]) if (up["FN"] + up["TN"]) > 0 else 0)
                        if "accuracy_equality" in selected_fairness :
                            accuracy_equality = ((p["TP"] + p["TN"]) / p["n"]) - ((up["TP"] + up["TN"]) / up["n"])
                        if "statistical_parity" in selected_fairness :
                            statistical_parity = ((p["TP"] + p["FP"]) / p["n"]) - ((up["TP"] + up["FP"]) / up["n"])
                        
                        report_lines.append(f"  - Comparação {protected_group} (protegido) vs {unprotected_group} (não protegido):")
                        if "equal_opportunity" in selected_fairness:
                            report_lines.append(f"    - Equal Opportunity: {equal_opportunity:.4f}")
                            dicio_all_fair[column][(protected_group, unprotected_group)]["equal_opportunity"] = equal_opportunity

                        if "predictive_equality" in selected_fairness:
                            report_lines.append(f"    - Predictive Equality: {predictive_equality:.4f}")
                            dicio_all_fair[column][(protected_group, unprotected_group)]["predictive_equality"] = predictive_equality

                        if "positive_predictive_parity" in selected_fairness:
                            report_lines.append(f"    - Positive Predictive Parity: {positive_predictive_parity:.4f}")
                            dicio_all_fair[column][(protected_group, unprotected_group)]["positive_predictive_parity"] = positive_predictive_parity

                        if "negative_predictive_parity" in selected_fairness:
                            report_lines.append(f"    - Negative Predictive Parity: {negative_predictive_parity:.4f}")
                            dicio_all_fair[column][(protected_group, unprotected_group)]["negative_predictive_parity"] = negative_predictive_parity

                        if "accuracy_equality" in selected_fairness:
                            report_lines.append(f"    - Accuracy Equality: {accuracy_equality:.4f}")
                            dicio_all_fair[column][(protected_group, unprotected_group)]["accuracy_equality"] = accuracy_equality

                        if "statistical_parity" in selected_fairness:
                            report_lines.append(f"    - Statistical Parity: {statistical_parity:.4f}")
                            dicio_all_fair[column][(protected_group, unprotected_group)]["statistical_parity"] = statistical_parity


        final_report = "\n".join(report_lines)
        #print("/n /n /n fairness")
        #print(final_report)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(final_report + "\n")
        
        return final_report, dicio_all_fair
