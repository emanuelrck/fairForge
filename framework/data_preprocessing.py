import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE

from sklearn.preprocessing import LabelEncoder
from fairlearn.preprocessing import CorrelationRemover
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing, LFR, OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult

from aif360.algorithms.preprocessing import DisparateImpactRemover
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric

def dataset_bias_metrics(data, sensitive_attr, privileged_classes, target_name, favorable_target ):
    results = {}
    df = data.copy()
    label_encoders = {}

    for col in range(len(sensitive_attr)):
        if len(df[sensitive_attr[col]].unique()) > 2:
            df[sensitive_attr[col]] = df[sensitive_attr[col]].apply(lambda x: str(x) if x == privileged_classes[col] else f"not_{privileged_classes[col]}")
            privileged_classes[col] = str(privileged_classes[col])
        df[sensitive_attr[col]] = df[sensitive_attr[col]].astype('object')
    df[target_name] = df[target_name].astype('object')
    
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    for index in range(len(sensitive_attr)):
        sense_mapping = dict(zip(label_encoders[sensitive_attr[index]].classes_, label_encoders[sensitive_attr[index]].transform(label_encoders[sensitive_attr[index]].classes_)))
        target_mapping = dict(zip(label_encoders[target_name].classes_, label_encoders[target_name].transform(label_encoders[target_name].classes_)))
        
        dataset = StandardDataset(df,
                                label_name=target_name,
                                favorable_classes=[target_mapping[favorable_target]],
                                protected_attribute_names=[sensitive_attr[index]],
                                privileged_classes=[[sense_mapping[privileged_classes[index]]]])
        metric = BinaryLabelDatasetMetric(dataset,
                                  privileged_groups=[{sensitive_attr[index]: sense_mapping[privileged_classes[index]]}],
                                  unprivileged_groups=[{sensitive_attr[index]: 1 - sense_mapping[privileged_classes[index]]}])
        results[sensitive_attr[index]] = {
    "Statistical Parity Difference": (metric.statistical_parity_difference(), "±0.1", "bias if |value| > 0.1"),
    "Disparate Impact": (metric.disparate_impact(), "0.8–1.25", "bias if < 0.8 (EEOC rule)"),
    "Mean Difference": (metric.mean_difference(), "±0.1", "bias if |value| > 0.1"),
    "Consistency": (metric.consistency(), "≥0.9", "bias if < 0.9")
    }

    for category in results.keys():
        st.markdown(f"**Metrics by {category.capitalize()}**")
        df_raca = metricas_para_df(results[category])
        st.dataframe(df_raca)


def detectar_bias(metrica, valor):
    v = valor[0] if isinstance(valor, np.ndarray) else float(valor)
    if metrica in ["Statistical Parity Difference", "Mean Difference"]:
        return "Bias Detected" if abs(v) > 0.1 else "No Bias"
    elif metrica == "Disparate Impact":
        return "Bias Detected" if v < 0.8 or v > 1.25 else "No Bias"
    elif metrica == "Consistency":
        return "Bias Detected" if v < 0.9 else "No Bias"
    else:
        return "Unknown"
def metricas_para_df(metricas_dict):
    rows = []
    for metrica, (valor, referencia, _) in metricas_dict.items():
        valor_float = valor[0] if isinstance(valor, np.ndarray) else float(valor)
        bias = detectar_bias(metrica, valor)
        rows.append({
            "Metric": metrica,
            "Value": round(valor_float, 4),
            "Reference": referencia,
            "Bias detected": bias
        })
    return pd.DataFrame(rows)

def impute_missing_values(df, numeric_strategy="mean", categorical_strategy="mode", custom_value=None, 
                          use_knn=False, use_iterative=False, use_rf=False, use_autoencoder=False):
    method = ""
    df_imputed = df.copy()
    
    # Numeric Columns
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    method += "Numeric: "
    if numeric_cols.any():
        if use_knn:
            imputer_knn = KNNImputer(n_neighbors=5)
            df_imputed[numeric_cols] = imputer_knn.fit_transform(df_imputed[numeric_cols])
            method += "Knn |"
        elif use_iterative:
            method += "Mice |"
            imputer_iterative = IterativeImputer(max_iter=10, random_state=0)
            df_imputed[numeric_cols] = imputer_iterative.fit_transform(df_imputed[numeric_cols])
            
        elif use_rf:
            method += "Random Forest |"
            rf = RandomForestRegressor(n_estimators=100, random_state=0)
            for col in numeric_cols:
                train_data = df_imputed.dropna(subset=[col])
                X_train = train_data.drop(col, axis=1)
                y_train = train_data[col]
                rf.fit(X_train, y_train)
                missing_data = df_imputed[df_imputed[col].isnull()]
                X_missing = missing_data.drop(col, axis=1)
                df_imputed.loc[df_imputed[col].isnull(), col] = rf.predict(X_missing)
        else:
            if numeric_strategy == "constant" and custom_value is not None:
                method += f"constant = {custom_value} |"
                imputer_num = SimpleImputer(strategy="constant", fill_value=custom_value)
            else:
                method += f"{numeric_strategy} |"
                imputer_num = SimpleImputer(strategy=numeric_strategy)
            df_imputed[numeric_cols] = imputer_num.fit_transform(df_imputed[numeric_cols])
    
    # Categorical Columns
    categorical_cols = df_imputed.select_dtypes(include=["object"]).columns
    if categorical_strategy == "constant" and custom_value is not None:
        method += f" Categorical: constant = {custom_value}"
    else:
        method += f" Categorical: {categorical_strategy}"

    for col in categorical_cols:
        if categorical_strategy == "mode":
            
            mode = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else "Unknown"
            df_imputed[col] = df_imputed[col].fillna(mode)
        elif categorical_strategy == "constant" and custom_value is not None:
           
            df_imputed[col] = df_imputed[col].fillna(custom_value)
        elif categorical_strategy == "unknown":
           
            df_imputed[col] = df_imputed[col].fillna("Unknown")
    
    return df_imputed, method


def oversampling(df, target_column, sensitive_columns, method="None", random_state=42, clusters = 10):
    if method == "None":
        return df
    
    df_encoded = df.copy()  
    label_encoders = {}
    for column in df_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le  

    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]

    sampler = None
    if method == "Smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "FairSmote":
        sampler = SMOTE(random_state=random_state)
    elif method == "Random":
        sampler =  RandomOverSampler(random_state=random_state)
    elif method == "ADASYN":
        sampler = ADASYN(random_state=random_state)
    elif method == "Borderline Smote":
        sampler = BorderlineSMOTE(random_state=random_state)
    elif method == "Kmeans Smote":
        sampler = KMeansSMOTE(
        kmeans_estimator=MiniBatchKMeans(n_clusters=clusters, n_init=1, random_state=random_state),
        random_state=random_state,
    )
    elif method == "SVM Smote":
        sampler = SVMSMOTE(random_state=0)
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled
    for column in label_encoders:
        df_resampled[column] = label_encoders[column].inverse_transform(df_resampled[column])

    return df_resampled


def augment_minority_group(df, target_column, sensitive_column, group_value, N=100):
    if target_column not in df or sensitive_column not in df or N <= 0:
        return df 
    
    df_minority = df[df[sensitive_column] == group_value]
    df_synthetic = resample(df_minority, replace=True, n_samples=N, random_state=42)
    df_augmented = pd.concat([df, df_synthetic], axis=0).reset_index(drop=True)
    return df_augmented

def reweigh(df, target, favorable_classes, protected_attribute_name, privileged_classes):
    
    label_encoders = {}
    df = df.dropna()
    df = df.reset_index(drop=True)
   
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
  
    if protected_attribute_name in label_encoders:
        value_privileged_classes = label_encoders[protected_attribute_name].transform([privileged_classes])[0]
        privileged_groups = [{protected_attribute_name: value_privileged_classes}]
        unprivileged_groups = [{protected_attribute_name: 0 if value_privileged_classes != 0 else 1}]

    else:
        value_privileged_classes = privileged_classes
        privileged_groups = [{protected_attribute_name: privileged_classes}]
        unprivileged_groups = [{protected_attribute_name: next(val for val in df[protected_attribute_name].unique().tolist() if val != value_privileged_classes)}]
    
    if target in label_encoders:
        favorable_classes = label_encoders[target].transform([favorable_classes])[0]
    else:
        favorable_classes = favorable_classes

    dataset = StandardDataset(
        df,
        label_name=target,
        favorable_classes=[favorable_classes],
        protected_attribute_names=[protected_attribute_name],
        privileged_classes=[[value_privileged_classes]],
        features_to_drop=[]
    )

    rw = Reweighing(unprivileged_groups, privileged_groups)
    rw_dataset = rw.fit_transform(dataset)
    df_transf = rw_dataset.convert_to_dataframe()[0]
    instance_weights = rw_dataset.instance_weights
    df_transf = revert_label_encoding(df_transf, label_encoders)
    df_transf = df_transf.reset_index(drop=True)
    return instance_weights

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def Dir(
    df,
    protected_attribute,
    target,
    repair_level=1.0,
    drop_columns=None
):
    df = df.copy()
    original_df = df.copy()
    if protected_attribute in drop_columns:
        drop_columns.remove(protected_attribute)
    if drop_columns is not None:
        dropped_data = df[drop_columns].copy()
        df = df.drop(columns=drop_columns)

    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col != protected_attribute and col != target:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    repaired_df = df.copy()
    groups = df[protected_attribute].unique()

    for col in numeric_cols:
        if col == protected_attribute:
            continue

        cdfs = {}
        for g in groups:
            col_values = df.loc[df[protected_attribute] == g, col].sort_values()
            probs = np.linspace(0, 1, len(col_values), endpoint=False)
            cdfs[g] = (col_values.values, probs)

        all_quantiles = []
        for g in groups:
            all_quantiles.append(np.quantile(df[df[protected_attribute] == g][col], np.linspace(0, 1, 100)))
        median_quantile = np.median(np.stack(all_quantiles), axis=0)
        q_levels = np.linspace(0, 1, 100)

        for g in groups:
            mask = df[protected_attribute] == g
            vals = df.loc[mask, col].values
            ranks = rankdata(vals, method='average') / len(vals)
            orig_q = np.quantile(vals, ranks)
            interp_vals = np.interp(ranks, q_levels, median_quantile)
            repaired_vals = (1 - repair_level) * vals + repair_level * interp_vals
            repaired_df.loc[mask, col] = repaired_vals

    repaired_df = pd.concat([dropped_data.reset_index(drop=True), repaired_df], axis=1)
    return repaired_df



def Learning_fair_representations(
    df,
    target,
    favorable_classes,
    protected_attribute_name,
    privileged_classes,
    drop_columns=None
):
    from sklearn.preprocessing import LabelEncoder
    from aif360.datasets import StandardDataset
    from aif360.algorithms.preprocessing import LFR
    import random
    import tensorflow as tf
    random.seed(70)
    tf.random.set_seed(70)
    label_encoders = {}
    original_df = df.copy()
    target_data = original_df[target]
    if protected_attribute_name in drop_columns:
        drop_columns.remove(protected_attribute_name)

    if drop_columns is not None:
        dropped_data = df[drop_columns].copy()
        df = df.drop(columns=drop_columns)
    else:
        dropped_data = pd.DataFrame(index=df.index)  

    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    if protected_attribute_name in label_encoders:
        value_privileged_classes = label_encoders[protected_attribute_name].transform([privileged_classes])[0]
    else:
        value_privileged_classes = privileged_classes
    privileged_groups = [{protected_attribute_name: value_privileged_classes}]
    unprivileged_val = next(val for val in df[protected_attribute_name].unique()
                            if val != value_privileged_classes)
    unprivileged_groups = [{protected_attribute_name: unprivileged_val}]

    if target in label_encoders:
        favorable_classes_transformed = label_encoders[target].transform([favorable_classes])[0]
    else:
        favorable_classes_transformed = favorable_classes
    dataset = StandardDataset(
        df,
        label_name=target,
        favorable_classes=[favorable_classes_transformed],
        protected_attribute_names=[protected_attribute_name],
        privileged_classes=[[value_privileged_classes]],
        features_to_drop=[]
    )

    lfr = LFR(unprivileged_groups, privileged_groups, k=5, Ax=0.01, Ay=1.0, Az=2.0, verbose=1)
    lfr.fit(dataset)
    lfr_dataset = lfr.transform(dataset)

    df_transf = lfr_dataset.convert_to_dataframe()[0].reset_index(drop=True)

    df_transf = pd.concat([dropped_data.reset_index(drop=True), df_transf], axis=1)

    for col in [protected_attribute_name, target]:
        if col in label_encoders and col in df_transf:
            le = label_encoders[col]
            valid_values = df_transf[col].round().astype(int).clip(0, len(le.classes_) - 1)
            df_transf[col] = le.inverse_transform(valid_values)
    df_transf[protected_attribute_name] = original_df[protected_attribute_name]
    if len(df_transf[target].unique()) == 1:
        df_transf[target] = target_data
    return df_transf, original_df, label_encoders

def revert_label_encodinglfr(df, label_encoders):
    for column, le in label_encoders.items():
        if column in df:
            original_classes = np.arange(len(le.classes_))
            col_values = df[column].values
            mapped_values = np.array([
                original_classes[np.abs(original_classes - val).argmin()]
                for val in col_values
            ])

            df[column] = le.inverse_transform(mapped_values)

    return df


def revert_label_encoding(df, label_encoders):
    for column, le in label_encoders.items():
        if column in df.columns:
            df[column] = le.inverse_transform(df[column].astype(int))
    return df

def change_labels(df, target_column, sensitive_column, sensitive_group_to_replace, N=100):
    filtered_df = df[df[sensitive_column] == sensitive_group_to_replace]
    N = min(N, len(filtered_df))
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])  
    filtered_df[target_column] = label_encoder.transform(filtered_df[target_column])

    X = filtered_df.drop(columns=[target_column, sensitive_column]) 
    y = filtered_df[target_column]  
    X = pd.get_dummies(X)  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    probas = model.predict_proba(X)[:, 1]  
    
    filtered_df['prob_target'] = probas
    sorted_df = filtered_df.sort_values(by='prob_target', ascending=False)
    top_n_df = sorted_df.head(N)
    indices_to_replace = top_n_df.index

    for indx in indices_to_replace:
        aux = set(df[target_column])  
        aux.discard(df.loc[indx, target_column])  
        df.loc[indx, target_column] = aux.pop()  
    
    df[target_column] = label_encoder.inverse_transform(df[target_column])
    return df