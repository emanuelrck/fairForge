import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.preprocessing import LabelEncoder
from framework.fairness_evaluator import FairnessEvaluator
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.datasets import StandardDataset
from aif360.algorithms.postprocessing import (
    RejectOptionClassification,
    EqOddsPostprocessing,
    CalibratedEqOddsPostprocessing
)
from sklearn.calibration import CalibratedClassifierCV
warnings.simplefilter(action='ignore', category=FutureWarning)
from fairlearn.postprocessing import ThresholdOptimizer
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class ModelTrainer:
    def __init__(self, df, target_column, sensitive_columns, test_size=0.3, random_state=42, selected_models=None, train_columns = None, sample_weight = [], sensitive_attr = None, favorable_classes_target = None, inprocessing_models = None):

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {df.columns.tolist()}")
        self.df = df.dropna()
        if len(self.df) == 0:
            raise ValueError("The dataset is empty after removing rows with missing values. Please impute missing values first.")
        self.feature_names = df.columns.tolist()
        self.feature_names = [f for f in self.feature_names if f != target_column]
        self.fairness_method = None
        self.fairness_params = {}
        self.sensitive_attr = sensitive_attr
        self.inprocessing_models = inprocessing_models
        self.sample_weight = sample_weight
        self.target_column = target_column
        self.favorable_classes_target = favorable_classes_target
        self.sensitive_columns = sensitive_columns
        self.test_size = test_size
        self.random_state = random_state
        self.predictions = {}
        df = df.reset_index(drop=True)
        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]
        self.train_columns = train_columns if train_columns is not None else []
        if target_column in self.train_columns:
            self.train_columns.remove(target_column)
       
        self.label_encoders = {}
        for column in self.X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            self.X[column] = le.fit_transform(self.X[column])
            self.label_encoders[column] = le

        if self.y.dtype == "object":
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)
            self.label_encoders[target_column] = le
        
        if self.train_columns:
            valid_cols = [c for c in self.train_columns if c in self.X.columns]
            if valid_cols:
                self.X = self.X[valid_cols]

        if len(self.sample_weight) != 0:
          if len(self.sample_weight) != len(self.X):
            self.sample_weight = []
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
            self.sample_weight_train = []
          else:
            self.X_train, self.X_test, self.y_train, self.y_test, self.sample_weight_train, self.sample_weight_test = train_test_split(
            self.X, self.y, self.sample_weight, test_size=self.test_size, random_state=self.random_state
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
            self.sample_weight_train = []


        self.df_test = df.loc[self.X_test.index].reset_index(drop=True)
        self.fairness_evaluator = FairnessEvaluator(self.df_test, sensitive_columns)

        available_models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(),
            "SVM": SVC(),
            "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1, learning_rate_init=0.001)
        }

        self.models = {name: model for name, model in available_models.items() if name in selected_models}

        if not self.models:
            raise ValueError(" Nenhum modelo foi selecionado para treinamento.")

    def train_and_evaluate(self, file_path=None, fairness_file_path=None, selected_fairness = None):
        
        report_lines = ["===== Model Training & Evaluation Report ====="]
        report_fair = []
        final_dicio_fairness = {}
        bld = None
        dicio_model_implement = {"grid_search_reduction": "all",
        "Exponentiated Gradient Reduction": "all",
        "adversarial_debiasing": "MLP (Neural Network)",
        "gerry_fair_classifier": "Logistic Regression",
        "meta_fair_classifier": "Logistic Regression",
        "prejudice_remover":"Logistic Regression",
        
        }
        for name, model in self.models.items():
            if self.fairness_method and ((dicio_model_implement[self.fairness_method] == "all"  and name in self.inprocessing_models)or name == dicio_model_implement[self.fairness_method] ):
                trained_model, bld = apply_fair_training(self,
                    self.X_train,
                    self.y_train,
                    self.X_test,
                    self.y_test,
                    model,
                    sensitive_features = self.sensitive_attr,  
                    fairness_method = self.fairness_method,
                    fairness_params = self.fairness_params,
                    sensitive_attr = self.sensitive_attr
                
                )
                model = trained_model
                method_info = "testar"
                report_lines.append(f"{name} - Treinado com fairness method: {method_info}")

               
                if bld == None:
                    predictions = model.predict(self.X_test)
                else:
                    predictions = model.predict(bld).labels.ravel()
                
            else:
                # Treino tradicional
                if name == "MLP (Neural Network)":
                    model.fit(self.X_train, self.y_train)
                elif len(self.sample_weight_train) != 0:
                    model.fit(self.X_train, self.y_train, sample_weight=self.sample_weight_train)
                else:
                    model.fit(self.X_train, self.y_train)
                predictions = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            classification_rep = classification_report(self.y_test, predictions)
            self.predictions[name] = predictions
            result = f"\n{name} - Accuracy: {accuracy:.4f}\n{classification_rep}"
            
            report_lines.append(result)
            final_dicio_fairness[name] = experiment_fairness( predictions, name, self.sensitive_columns, self.target_column, self.favorable_classes_target, selected_fairness, self.df_test)
            self.models[name] = model

        final_report = "\n".join(report_lines)
        rows = []

        # Iterando pelo dicionário e criando as linhas
        for model, model_data in final_dicio_fairness.items():
            for attribute, attribute_data in model_data.items():
                for combination, metrics in attribute_data.items():
                    row = {
                        'model': model,
                        'attribute': attribute,
                        'combination': str(combination),
                        'equal_opportunity': metrics['equal_opportunity'],
                        'predictive_equality': metrics['predictive_equality'],
                        'positive_predictive_parity': metrics['positive_predictive_parity'],
                        'false_omission_rate': metrics['false_omission_rate'],
                        'disparate_impact': metrics['disparate_impact'],
                        'statistical_parity': metrics['statistical_parity'],
                    }
                    rows.append(row)
        df = pd.DataFrame(rows)
        return final_report, final_dicio_fairness

def experiment_fairness(predictions, name, sensitive_columns, target, positive_target, selected_fairness, test_dataset):
    predictions = np.asarray(predictions)
    dicio_all_fair = {}
    for sense_att in sensitive_columns:
        if sense_att not in test_dataset.columns:
                continue
        dicio_all_fair[sense_att] = {}
        groups = test_dataset[sense_att].unique()
        if len(groups) > 5:
            groups = groups [:5]
        metrics = {}
        
        for priveledge in groups:
            priveledge = str(priveledge)
            dicio_all_fair[sense_att][(priveledge, "not_"+ str(priveledge))] = {}
            df_test_c = test_dataset.copy()
            df_test_c[sense_att] = df_test_c[sense_att].apply(lambda val: priveledge if str(val) == priveledge else "not_"+priveledge)

            label_encoders = {}
            df_test_c[target] = df_test_c[target].astype('object')
            for col in df_test_c.select_dtypes(include='object').columns:
                le = LabelEncoder()
                df_test_c[col] = le.fit_transform(df_test_c[col])
                label_encoders[col] = le
            sex_mapping = dict(zip(label_encoders[sense_att].classes_, label_encoders[sense_att].transform(label_encoders[sense_att].classes_)))
            if target not in label_encoders:
                continue
            income_mapping = dict(zip(label_encoders[target].classes_, label_encoders[target].transform(label_encoders[target].classes_)))
            positive_target_str = str(positive_target)
            if positive_target_str not in income_mapping and positive_target not in income_mapping:
                continue
            fav_class_val = income_mapping.get(positive_target_str, income_mapping.get(positive_target))
            test_exp = StandardDataset(df_test_c,
                            label_name=target,
                            favorable_classes=[fav_class_val],
                            protected_attribute_names=[sense_att],
                            privileged_classes=[[sex_mapping[priveledge]]])

            test_lfr_pred = test_exp.copy()
            test_lfr_pred.labels = predictions.reshape(-1, 1)

            metric = ClassificationMetric(test_exp, test_lfr_pred,
                                        unprivileged_groups=[{sense_att: 1 - sex_mapping[priveledge]}],
                                        privileged_groups=[{sense_att: sex_mapping[priveledge]}])
            
            if "equal_opportunity" in selected_fairness :
                dicio_all_fair[sense_att][(priveledge, "not_"+priveledge)]["equal_opportunity"] = metric.equal_opportunity_difference()
            if "predictive_equality" in selected_fairness :
                dicio_all_fair[sense_att][(priveledge, "not_"+priveledge)]["predictive_equality"] = metric.false_positive_rate_difference()
            if "positive_predictive_parity" in selected_fairness :
                dicio_all_fair[sense_att][(priveledge, "not_"+priveledge)]["positive_predictive_parity"] = metric.average_predictive_value_difference()
            if "false_omission_rate" in selected_fairness :
                dicio_all_fair[sense_att][(priveledge, "not_"+priveledge)]["false_omission_rate"] = metric.false_omission_rate_difference()
            if "statistical_parity" in selected_fairness :
                dicio_all_fair[sense_att][(priveledge, "not_"+priveledge)]["statistical_parity"] = metric.statistical_parity_difference()
            if "disparate_impact" in selected_fairness :
                dicio_all_fair[sense_att][(priveledge, "not_"+priveledge)]["disparate_impact"] = metric.disparate_impact()
    return dicio_all_fair




def print_fairness_metrics(metric, name):
    print(f"\n=== {name} ===")
    print("Statistical Parity Difference:", metric.statistical_parity_difference())
    print("Disparate Impact:", metric.disparate_impact())
    print("Equal Opportunity Difference:", metric.equal_opportunity_difference())
    print("False Positive Rate Difference (Predictive Equality):", metric.false_positive_rate_difference())
    print("Positive Predictive Parity Difference:", metric.average_predictive_value_difference())
    print("false_omission_rate:", metric.false_omission_rate_difference())
   

def apply_fair_training(self,X, y, x_test, y_test, model, sensitive_features,
                        fairness_method=None,
                        fairness_params=None,
                        sensitive_attr = None,
                        ):
    if fairness_params is None:
        fairness_params = {}
    info = f"Fairness Method: {fairness_method or 'None'}"
    if fairness_method is None:
        model.fit(X, y)
        return model, info

    if isinstance(y, np.ndarray):
        y = pd.Series(y, name=self.target_column)
        y_test = pd.Series(y_test, name=self.target_column)

    df_all = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df_all_test = pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    if fairness_method == "prejudice_remover":
        from aif360.algorithms.inprocessing import PrejudiceRemover
        from aif360.datasets import BinaryLabelDataset

        bld = BinaryLabelDataset(df=df_all, 
                                 label_names=[y.name], 
                                 protected_attribute_names=[sensitive_features])

        bld_test = BinaryLabelDataset(df=df_all_test, 
                                 label_names=[y.name], 
                                 protected_attribute_names=[sensitive_features])

        fair_model = PrejudiceRemover(sensitive_attr=sensitive_features, eta=fairness_params.get('eta', 25.0))
        fair_model.fit(bld)
        info += f" | Eta: {fairness_params.get('eta', 25.0)}"
        return fair_model, bld_test


    elif fairness_method == "adversarial_debiasing":
        from aif360.algorithms.inprocessing import AdversarialDebiasing
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
        from aif360.datasets import BinaryLabelDataset
        tf.reset_default_graph()
        sess = tf.Session()
        data = pd.concat([X, y], axis=1)
        data = data.dropna()
        X = data[X.columns]
        y = data[y.name]
        bld = BinaryLabelDataset(df=pd.concat([X, y], axis=1), 
                                 label_names=[y.name], 
                                 protected_attribute_names=[sensitive_features])

        bld_test = BinaryLabelDataset(df=df_all_test, 
                                 label_names=[y.name], 
                                 protected_attribute_names=[sensitive_features])

        fair_model = AdversarialDebiasing(
            privileged_groups=[{sensitive_features: 1}],
            unprivileged_groups=[{sensitive_features: 0}],
            scope_name='debias_classifier',
            sess=sess,
            num_epochs=fairness_params.get('num_epochs', 50),
            batch_size=fairness_params.get('batch_size', 128),
            debias=True
        )
        fair_model.fit(bld)
        info += f" | Adversarial Debiasing (epochs: {fairness_params.get('num_epochs', 50)})"
        return fair_model,  bld_test

    elif fairness_method == "Exponentiated Gradient Reduction":
        from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
        from sklearn.linear_model import LogisticRegression
        constraint = EqualizedOdds()
        mitigator = ExponentiatedGradient(model, constraints=constraint)
        mitigator.fit(self.X_train, self.y_train, sensitive_features=self.X_train[sensitive_attr])

        return mitigator,  None

    elif fairness_method == "grid_search_reduction":
        from fairlearn.reductions import GridSearch, EqualizedOdds
        from sklearn.linear_model import LogisticRegression

        constraint = EqualizedOdds()
        mitigator = GridSearch(model, constraints=constraint, grid_size=20)
        mitigator.fit(X, y, sensitive_features=X[sensitive_attr])
        info += " | Grid Search Reduction"
        return mitigator, None

    elif fairness_method == "meta_fair_classifier":
        from aif360.algorithms.inprocessing import MetaFairClassifier
        from aif360.datasets import BinaryLabelDataset

        bld = BinaryLabelDataset(df=df_all, label_names=[y.name], protected_attribute_names=[sensitive_features])
        bld_test = BinaryLabelDataset(df=df_all_test, label_names=[y.name], protected_attribute_names=[sensitive_features])

        tau = fairness_params.get("tau", 0.8)

        fair_model = MetaFairClassifier(sensitive_attr=sensitive_attr, tau=tau)
        fair_model.fit(bld)
        info += f" | MetaFair (tau: {tau})"
        return fair_model, bld_test

    elif fairness_method == "gerry_fair_classifier":
        from aif360.algorithms.inprocessing import GerryFairClassifier
        from aif360.datasets import BinaryLabelDataset

        bld = BinaryLabelDataset(df=df_all, label_names=[y.name], protected_attribute_names=[sensitive_features])
        bld_test = BinaryLabelDataset(df=df_all_test, label_names=[y.name], protected_attribute_names=[sensitive_features])

        fair_model = GerryFairClassifier(C=fairness_params.get("C", 500))
        fair_model.fit(bld)
        info += f" | GerryFair (C: {fairness_params.get('C', 500)})"
        return fair_model, bld_test

    else:
        raise ValueError(f"Unknown fairness_method: {fairness_method}")


from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from fairlearn.postprocessing import ThresholdOptimizer
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import (
    RejectOptionClassification,
    EqOddsPostprocessing,
    CalibratedEqOddsPostprocessing,
)
from sklearn.preprocessing import LabelEncoder
import numpy as np

def _is_sklearn_estimator(model):
    return hasattr(model, "get_params") and hasattr(model, "set_params")

def postProcessing(method, predictions, df_test, model, sensitive, priveledge, target):
    pred = None

    unprivileged = [{sensitive: 0}]
    privileged = [{sensitive: 1}]

    df_encoded = df_test.copy()
    categorical_cols = df_encoded.select_dtypes(include='object').columns.tolist()
    if target in categorical_cols:
        categorical_cols.remove(target)
    if sensitive in categorical_cols:
        categorical_cols.remove(sensitive)

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    classes = sorted(df_encoded[target].unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"The target '{target}' is not binary (values: {classes}). Post-processing requires a binary target.")
    positive_class = classes[1]
    df_encoded[target] = (df_encoded[target] == positive_class).astype(int)
    df_encoded[sensitive] = (df_encoded[sensitive] == priveledge).astype(int)

    assert df_encoded.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), \
        "Existem colunas não numéricas após o encoding."

    bld = BinaryLabelDataset(
        df=df_encoded,
        label_names=[target],
        protected_attribute_names=[sensitive]
    )
    pred_bld = bld.copy()
    pred_bld.labels = np.asarray(predictions).reshape(-1, 1)
    features = df_encoded.drop(columns=[target])
    labels = df_encoded[target]
    s_feat = df_encoded[sensitive]

    if method == "Reject Option Classification":
        roc = RejectOptionClassification(
            unprivileged_groups=unprivileged,
            privileged_groups=privileged,
            metric_name="Statistical parity difference",
            metric_ub=0.05,
            metric_lb=-0.05
        )
        roc.fit(bld, pred_bld)
        pred = roc.predict(pred_bld).labels.flatten()

    elif method == "Equalized Odds":
        eq_odds = EqOddsPostprocessing(
            privileged_groups=privileged,
            unprivileged_groups=unprivileged
        )
        eq_odds.fit(bld, pred_bld)
        pred = eq_odds.predict(pred_bld).labels.flatten()

    elif method == "Calibrated Equalized Odds":
        if not _is_sklearn_estimator(model):
            return predictions
           
        cal_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        cal_model.fit(features, labels)
        proba = cal_model.predict_proba(features)
        scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        bld.scores = scores.reshape(-1, 1)

        cal_eq_odds = CalibratedEqOddsPostprocessing(
            privileged_groups=privileged,
            unprivileged_groups=unprivileged,
            cost_constraint='fnr',
            seed=42
        )
        cal_eq_odds.fit(bld, pred_bld)
        pred = cal_eq_odds.predict(pred_bld).labels.flatten()

    elif method == "Threshold Optimizer":
        if not _is_sklearn_estimator(model):    
            return predictions
           
        threshold_optimizer = ThresholdOptimizer(
            estimator=model,
            constraints="equalized_odds",     
            predict_method="predict_proba",
            prefit=True                       
        )
        threshold_optimizer.fit(
            X=features,
            y=labels,
            sensitive_features=s_feat
        )
        pred = threshold_optimizer.predict(
            features, sensitive_features=s_feat
        )

    else:
        raise ValueError(f"Método de pós-processamento desconhecido: {method}")

    return np.asarray(pred)
