import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
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
            print("-----------C-------------")
            print(sensitive_attr[col])
            print(df[sensitive_attr[col]].unique())
            print("\n\n\n\Binarizei\n\n\n")
            df[sensitive_attr[col]] = df[sensitive_attr[col]].apply(lambda x: str(x) if x == privileged_classes[col] else f"not_{privileged_classes[col]}")
            privileged_classes[col] = str(privileged_classes[col])
        df[sensitive_attr[col]] = df[sensitive_attr[col]].astype('object')
        print("col to objetc ", sensitive_attr[col])
    print(privileged_classes)
    df[target_name] = df[target_name].astype('object')
    
    for col in df.select_dtypes(include='object').columns:
        print("-----------B-------------")
        print(col)
        print(df[col].unique())
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

        print(label_encoders)
    for index in range(len(sensitive_attr)):
        print("----------A-----------")
        print(sensitive_attr[index])
        print(df[sensitive_attr[index]].unique())
        sense_mapping = dict(zip(label_encoders[sensitive_attr[index]].classes_, label_encoders[sensitive_attr[index]].transform(label_encoders[sensitive_attr[index]].classes_)))

        print("sense_mapping#:", sense_mapping)
        target_mapping = dict(zip(label_encoders[target_name].classes_, label_encoders[target_name].transform(label_encoders[target_name].classes_)))
        
        # === Step 3: Create AIF360 dataset ===
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
        st.subheader("Metrics by  "+ category.capitalize())
        df_raca = metricas_para_df(results[category])
        st.dataframe(df_raca)

    
    print(results)


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
            "Métrica": metrica,
            "Valor": round(valor_float, 4),
            "Referência": referencia,
            "Bias Detectado": bias
        })
    return pd.DataFrame(rows)

def impute_missing_values(df, numeric_strategy="mean", categorical_strategy="mode", custom_value=None, 
                          use_knn=False, use_iterative=False, use_rf=False, use_autoencoder=False):
    """
    Impute missing values based on selected strategies:
    
    - Numeric columns: 'mean', 'median', 'most_frequent', 'constant' (custom value).
    - Categorical columns: 'mode', 'constant' (custom value), 'unknown'.
    - KNN Imputation (K-Nearest Neighbors) and Iterative Imputation (MICE) can be enabled.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        numeric_strategy (str): Strategy for numerical imputation ('mean', 'median', 'most_frequent', 'constant').
        categorical_strategy (str): Strategy for categorical imputation ('mode', 'constant', 'unknown').
        custom_value (optional): Custom value to use when 'constant' is selected.
        use_knn (bool): Whether to apply KNN imputation.
        use_iterative (bool): Whether to apply Iterative Imputation (MICE).
        use_rf (bool): Whether to apply Random Forest for imputation.
        use_autoencoder (bool): Whether to use an Autoencoder for deep learning-based imputation.
    
    Returns:
        pd.DataFrame: A new dataframe with imputed values.
    """
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

    """
    Implementação do FairSMOTE para balanceamento de classes considerando atributos sensíveis.

    Args:
        df (pd.DataFrame): O dataset original.
        target_column (str): Nome da coluna-alvo (classe).
        sensitive_columns (list): Lista de colunas sensíveis que devem ser preservadas.
        sampling_strategy (str or dict, optional): Estratégia de amostragem para SMOTE. Default: "auto".
        random_state (int, optional): Seed para reprodutibilidade.

    Returns:
        pd.DataFrame: O dataset balanceado pelo FairSMOTE.
    """

    if method == "None":
        return df
    
    df_encoded = df.copy()  # Criar cópia para preservar os dados originais

    # Aplicar Label Encoding para colunas categóricas
    label_encoders = {}
    for column in df_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le  # Salvar encoder para decodificar depois, se necessário

    # Separar features (X) e target (y)
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
    
    
    # Aplicar sampler apenas nas colunas não sensíveis
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Criar DataFrame com os dados balanceados
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled

    # Reverter Label Encoding para os atributos categóricos
    for column in label_encoders:
        df_resampled[column] = label_encoders[column].inverse_transform(df_resampled[column])

    # Ajustar atributos sensíveis preservando a distribuição original
    #for column in sensitive_columns:
    #    if column in df.columns:
    #        # Manter as proporções originais do atributo sensível
    #        original_distribution = df[column].value_counts(normalize=True)
    #        resampled_size = len(df_resampled)
    #        sampled_values = np.random.choice(original_distribution.index, size=resampled_size, p=original_distribution.values)
    #        df_resampled[column] = sampled_values

    return df_resampled


def augment_minority_group(df, target_column, sensitive_column, group_value, N=100):
    """
    Generates synthetic samples for the underrepresented group.

    Args:
        df (pd.DataFrame): Original dataset.
        target_column (str): Label column.
        sensitive_column (str): The sensitive attribute.
        group_value: The value of the underrepresented group.
        N (int, optional): Number of synthetic samples.

    Returns:
        df_augmented (pd.DataFrame): The augmented dataset.
    """
    if target_column not in df or sensitive_column not in df or N <= 0:
        return df 
    
    df_minority = df[df[sensitive_column] == group_value]
    
    # Generate synthetic samples
    df_synthetic = resample(df_minority, replace=True, n_samples=N, random_state=42)
    
    df_augmented = pd.concat([df, df_synthetic], axis=0).reset_index(drop=True)
    return df_augmented

def reweigh(df, target, favorable_classes, protected_attribute_name, privileged_classes):
    
    label_encoders = {}
    df = df.dropna()
    # Salvar o dataset no session_state
    df = df.reset_index(drop=True)
    print(df[target].value_counts())
    # Codifica colunas do tipo 'object'
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        print(column)
        label_encoders[column] = le

    print("Income classes:", label_encoders)
    
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
    print("Transformed dataset weights:", rw_dataset.instance_weights[:5])
    instance_weights = rw_dataset.instance_weights
    # Reverte os valores de encoding
    df_transf = revert_label_encoding(df_transf, label_encoders)
    df_transf = df_transf.reset_index(drop=True)
    print(df_transf[target].value_counts())
    return df_transf, instance_weights

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
    """
    Implements the Disparate Impact Remover (DIR) algorithm
    from Feldman et al. (2015).

    Parameters:
        df (pd.DataFrame): Input dataframe.
        protected_attribute (str): Name of the protected attribute column.
        repair_level (float): Degree of repair [0.0 (none) to 1.0 (full)].
        drop_columns (list): Optional list of columns to drop before processing.

    Returns:
        repaired_df (pd.DataFrame): Transformed dataframe with disparate impact removed.
        original_df (pd.DataFrame): Original input dataframe.
    """

    df = df.copy()
    original_df = df.copy()
    if protected_attribute in drop_columns:
        drop_columns.remove(protected_attribute)
    # Drop specified columns
    if drop_columns is not None:
        dropped_data = df[drop_columns].copy()
        df = df.drop(columns=drop_columns)

    # Label encode categorical features (except the protected attribute)
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col != protected_attribute and col != target:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Normalize numeric columns to [0, 1] (required for geometric repair)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    repaired_df = df.copy()
    groups = df[protected_attribute].unique()

    # Apply geometric repair to each numeric attribute
    for col in numeric_cols:
        if col == protected_attribute:
            continue

        # Get the CDF for each group
        cdfs = {}
        for g in groups:
            col_values = df.loc[df[protected_attribute] == g, col].sort_values()
            probs = np.linspace(0, 1, len(col_values), endpoint=False)
            cdfs[g] = (col_values.values, probs)

        # Build median quantile function
        all_quantiles = []
        for g in groups:
            all_quantiles.append(np.quantile(df[df[protected_attribute] == g][col], np.linspace(0, 1, 100)))
        median_quantile = np.median(np.stack(all_quantiles), axis=0)
        q_levels = np.linspace(0, 1, 100)

        # Apply geometric interpolation repair
        for g in groups:
            mask = df[protected_attribute] == g
            vals = df.loc[mask, col].values
            ranks = rankdata(vals, method='average') / len(vals)
            orig_q = np.quantile(vals, ranks)
            interp_vals = np.interp(ranks, q_levels, median_quantile)
            repaired_vals = (1 - repair_level) * vals + repair_level * interp_vals
            repaired_df.loc[mask, col] = repaired_vals

    repaired_df = pd.concat([dropped_data.reset_index(drop=True), repaired_df], axis=1)
    for col in repaired_df:
        print("---------")
        print(col)
        print(repaired_df[col].unique())
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

    # 1. Save original dataframe and removed columns
    original_df = df.copy()
    target_data = original_df[target]
    if protected_attribute_name in drop_columns:
        drop_columns.remove(protected_attribute_name)
    print(drop_columns)

    if drop_columns is not None:
        dropped_data = df[drop_columns].copy()
        df = df.drop(columns=drop_columns)
    else:
        dropped_data = pd.DataFrame(index=df.index)  # empty frame for consistency

    # 2. Encode categorical columns
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # 3. Prepare privileged/unprivileged groups
    if protected_attribute_name in label_encoders:
        value_privileged_classes = label_encoders[protected_attribute_name].transform([privileged_classes])[0]
    else:
        value_privileged_classes = privileged_classes
    print("value_privileged_classes:", value_privileged_classes)
    privileged_groups = [{protected_attribute_name: value_privileged_classes}]
    unprivileged_val = next(val for val in df[protected_attribute_name].unique()
                            if val != value_privileged_classes)
    unprivileged_groups = [{protected_attribute_name: unprivileged_val}]

    # 4. Transform favorable class label if needed
    if target in label_encoders:
        favorable_classes_transformed = label_encoders[target].transform([favorable_classes])[0]
    else:
        favorable_classes_transformed = favorable_classes
    print("favorable_classes_transformed:", favorable_classes_transformed)
    # 5. Create AIF360 dataset
    dataset = StandardDataset(
        df,
        label_name=target,
        favorable_classes=[favorable_classes_transformed],
        protected_attribute_names=[protected_attribute_name],
        privileged_classes=[[value_privileged_classes]],
        features_to_drop=[]
    )

    # 6. Apply LFR
    lfr = LFR(unprivileged_groups, privileged_groups, k=5, Ax=0.01, Ay=1.0, Az=2.0, verbose=1)
    lfr.fit(dataset)
    lfr_dataset = lfr.transform(dataset)

    # 7. Get transformed dataframe
    df_transf = lfr_dataset.convert_to_dataframe()[0].reset_index(drop=True)

    # 8. Reinsert dropped columns
    df_transf = pd.concat([dropped_data.reset_index(drop=True), df_transf], axis=1)

    # 9. Recover human-readable labels for protected attribute and target
    for col in [protected_attribute_name, target]:
        if col in label_encoders and col in df_transf:
            le = label_encoders[col]
            # Safely inverse_transform the valid label values
            valid_values = df_transf[col].round().astype(int).clip(0, len(le.classes_) - 1)
            df_transf[col] = le.inverse_transform(valid_values)

    for col in df_transf:
        print("---------")
        print(col)
        print(df_transf[col].unique())
    if len(df_transf[target].unique()) == 1:
        df_transf[target] = target_data
    return df_transf, original_df, label_encoders

def revert_label_encodinglfr(df, label_encoders):
    for column, le in label_encoders.items():
        if column in df:
            # Valores originais codificados vistos no fit do LabelEncoder
            original_classes = np.arange(len(le.classes_))  # ex: [0, 1, 2]
            print(original_classes)

            # Coluna atual após LFR (com valores contínuos ou fora do esperado)
            col_values = df[column].values

            # Mapeia cada valor para o valor original mais próximo
            mapped_values = np.array([
                original_classes[np.abs(original_classes - val).argmin()]
                for val in col_values
            ])

            # Agora sim, faz o inverse_transform com segurança
            df[column] = le.inverse_transform(mapped_values)
            print("---------------")
            print(column)
            print(df[column].unique())

    return df


def revert_label_encoding(df, label_encoders):
    """
    Reverte as colunas de um DataFrame que foram transformadas por LabelEncoder.
    """
    for column, le in label_encoders.items():
        if column in df.columns:
            print(column,"->",df[column].unique) 
            df[column] = le.inverse_transform(df[column].astype(int))
    return df

def change_labels(df, target_column, sensitive_column, sensitive_group_to_replace, N=100):
    # Filtra o dataframe para manter apenas as instâncias do grupo sensível
    filtered_df = df[df[sensitive_column] == sensitive_group_to_replace]
    
    N = min(N, len(filtered_df))
    
    # Codificar a variável alvo (target_column) se for categórica
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])  # Codificando os valores de string no df original
    filtered_df[target_column] = label_encoder.transform(filtered_df[target_column])  # Codificando os valores no filtered_df
    
    # Separar as features e o target
    X = filtered_df.drop(columns=[target_column, sensitive_column])  # features
    y = filtered_df[target_column]  # target
    
    # Codificar variáveis categóricas nas features, se existirem
    X = pd.get_dummies(X)  # Usando a codificação one-hot para variáveis categóricas
    
    # Dividir os dados em treino e teste (apenas para treinar o modelo de maneira adequada)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Usando RandomForestClassifier como exemplo de modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Gerar as probabilidades para cada instância no conjunto filtrado
    probas = model.predict_proba(X)[:, 1]  # Probabilidade da classe positiva (se for binário)
    
    # Adiciona a coluna de probabilidades no dataframe filtrado
    filtered_df['prob_target'] = probas
    
    # Ordena as instâncias pela probabilidade da classe positiva (descrescente)
    sorted_df = filtered_df.sort_values(by='prob_target', ascending=False)
    
    # Seleciona as top N instâncias com maior probabilidade
    top_n_df = sorted_df.head(N)
    
    # Substitui o valor de target_column nessas top N instâncias
    indices_to_replace = top_n_df.index

    for indx in indices_to_replace:
        aux = set(df[target_column])  # Recupera todos os valores únicos de target_column no df original
        aux.discard(df.loc[indx, target_column])  # Remove o valor atual do índice
        df.loc[indx, target_column] = aux.pop()  # Substitui o valor pelo outro valor disponível
    
    # Descodificar a coluna target_column de volta para os valores originais
    df[target_column] = label_encoder.inverse_transform(df[target_column])
    
    return df