import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from matplotlib.patches import Patch
available_models = [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "SVM"
        ]
available_models2 = [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM"
        ]
def show_fairness_test_model(fairness_dict):
    # Transformar dicionário em DataFrame
    rows = []
    for feature, pairs in fairness_dict.items():
        for group_pair, metrics in pairs.items():
            row = {
                'Feature': feature,
                'Group': group_pair[0],
            }
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Exibir como tabela no Streamlit
    st.subheader("Fairness Metrics Table")
    st.dataframe(df)
def automatic_fairness(df):
    st.sidebar.markdown("""<h2>Fairness Dynamics</h2>""", unsafe_allow_html=True)
    with st.sidebar.expander(" ", expanded=False):
        # Seleção dinâmica do fairness metric e do grupo protegido
        #selected_fairness = st.selectbox("Choose fairness metric:", st.session_state["selected_metrics_fairness"])
        sensitive_attribute = st.selectbox("Choose sensitive attribute:", st.session_state["sensitive_columns"] )
        priveledge_group = st.selectbox("Priveledge Group:", st.session_state["df"][sensitive_attribute].unique().tolist())
        #unprotected_group = st.selectbox("Unprotected Group:", [val for val in st.session_state["df"][sensitive_attribute].unique().tolist() if val != protected_group])

        show_fairness = False
        if st.button("Show Fairness Plot"):
            show_fairness = True
        
    
    return sensitive_attribute, priveledge_group, show_fairness

def ola (df):
    st.sidebar.markdown("""<h2>Fairness Dynamics</h2>""", unsafe_allow_html=True)
    with st.sidebar.expander(" ", expanded=False):
        # Seleção dinâmica do fairness metric e do grupo protegido
        
        sensitive_attribute = st.selectbox("Choose sensitive attribute:", st.session_state["sensitive_columns"] )
        protected_group = st.selectbox("Protected Group:", st.session_state["df"][sensitive_attribute].unique().tolist())
        #unprotected_group = st.selectbox("Unprotected Group:", [val for val in st.session_state["df"][sensitive_attribute].unique().tolist() if val != protected_group])

        show_fairness = False
        if st.button("Show Fairness Plot"):
            show_fairness = True
        
    st.sidebar.markdown("""<h2>Metrics</h2>""", unsafe_allow_html=True)
    with st.sidebar.expander(" ", expanded=False):
        sensitive_attribute_scatter, protected_group_scatter, selected_metric_1_scatter, selected_metric_2_scatter, show_scatter = select_metrics_and_plot(st.session_state["selected_metrics_fairness"], st.session_state["sensitive_columns"])
    
        
    return sensitive_attribute, protected_group , show_fairness, sensitive_attribute_scatter, protected_group_scatter, selected_metric_1_scatter, selected_metric_2_scatter, show_scatter


def postProcessing_caracteristics (df):
    st.sidebar.markdown("""<h2>PostProcessing attributes</h2>""", unsafe_allow_html=True)
    with st.sidebar.expander(" ", expanded=False):
        # Seleção dinâmica do fairness metric e do grupo protegido
        
        sensitive_post = st.selectbox("Choose sensitive attribute:", st.session_state["sensitive_columns"] )
        priveledge_post = st.selectbox("Priveledge Group:", st.session_state["df"][sensitive_post].unique().tolist())
        #unprotected_group = st.selectbox("Unprotected Group:", [val for val in st.session_state["df"][sensitive_attribute].unique().tolist() if val != protected_group])

        print(sensitive_post )
    return sensitive_post, priveledge_post

def show_fairness_plot_single_model(
    automatic_results, column, protected_group, selected_fairness, 
    ax=None, max_y=1
):
    unprotected_group = "not_" + str(protected_group)
    keys = list(automatic_results.keys())
    indices = np.arange(len(keys))
    bar_width = 0.5

    # Prepare data
    fairness_values = []
    bar_colors = []

    for key in keys:
        _, fairness_dict = automatic_results[key]  # get fairness dict
        value = None
        if column in fairness_dict["XGBoost"]:
            value = fairness_dict["XGBoost"][column].get((str(protected_group), unprotected_group), {}).get(selected_fairness, None)
        
        fairness_values.append(abs(value) if value is not None else 0)

        if value is None:
            bar_colors.append("gray")
        else:
            bar_colors.append("darkblue" if value < 0 else "skyblue")

    # Create figure/axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Plot bars
    ax.bar(indices, fairness_values, bar_width, color=bar_colors, label="Current")

    # X axis
    ax.set_xticks(indices)
    ax.set_xticklabels(keys, rotation=45, ha="right")

    # Y axis
    ax.set_ylabel(selected_fairness)
    ax.set_title(f"Fairness: {selected_fairness} - {column}")

    # Y axis limits
    if selected_fairness == "disparate_impact":
        ax.set_ylim(bottom=0, top=1)
    else:
        max_val = max(filter(None, fairness_values), default=1)
        ax.set_ylim(bottom=0, top=(max_y if max_y is not None else max_val) * 1.05)

    # Legend (consistent with style of show_plots_fairness)
    legend_elements = [
        Patch(facecolor='skyblue', label='Fairness ≥ 0'),
        Patch(facecolor='darkblue', label='Fairness < 0'),
        Patch(facecolor='gray', label='Missing value')
    ]
    ax.legend(handles=legend_elements)

    return fig



def inprocessing_categorys(df):
    st.sidebar.markdown("""<h2>Inprocessing options</h2>""", unsafe_allow_html=True)
    with st.sidebar.expander(" ", expanded=False):
        # Seleção dinâmica do fairness metric e do grupo protegido
        inprocessing_eta = st.number_input("Enter the eta value", value=25)
        sensitive_attribute = st.selectbox("Choose sensitive attribute to influence with inprocessing methods:", st.session_state["sensitive_columns"] )
        models_aplly_inprocessing = st.multiselect("Choose the models to be used with the inprocessing methods:", available_models, default =available_models2 )

        return inprocessing_eta, sensitive_attribute, models_aplly_inprocessing

def display_categorys(df):
    tipical_sensitive_information = {
            "gender", "sex", "biological_sex", "assigned_sex", "sex_at_birth", "gender_identity",
            "gender_expression", "gender_category", "race", "ethnicity", "ethnic_group",
            "racial_identity", "cultural_background", "dob",
            "year_of_birth", "religion", "belief", "faith", "spirituality", "religious_affiliation", "sex9", "marital", "marital_status"
        }
    #------------------------------------atributos
    


    

    #--------------------------------------------------------------------correçoes de data
    st.sidebar.markdown("""<h2>Missing Data</h2>""", unsafe_allow_html=True)
    with st.sidebar.expander("", expanded=False):
        # Seleção de estratégia para imputação numérica
            numeric_strategy = st.selectbox(
                "Choose numeric imputation strategy",
                ["mean", "median", "most_frequent", "constant"]
            )
            
            # Caso o usuário escolha 'constant', pedir o valor customizado
            custom_value = None
            if numeric_strategy == "constant":
                custom_value = st.number_input("Enter constant value", value=0)
            
            # Seleção de estratégia para imputação categórica
            categorical_strategy = st.selectbox(
                "Choose categorical imputation strategy",
                ["mode", "constant", "unknown"]
            )
            
            # Caso o usuário escolha 'constant' para categóricos, pedir o valor customizado
            custom_value_cat = None
            if categorical_strategy == "constant":
                custom_value_cat = st.text_input("Enter constant value for categorical", value="Unknown")
            
            # Adicionar checkboxes para opções avançadas
            use_knn = st.checkbox("Use KNN Imputation (K-Nearest Neighbors)", False)
            use_iterative = st.checkbox("Use Iterative Imputation (MICE)", False)
            use_rf = st.checkbox("Use Random Forest Imputation", False)
        

    st.sidebar.markdown("""<h2>Data Correction Options</h2>""", unsafe_allow_html=True)
    with st.sidebar.expander("", expanded=False):
        tab3, tab4, tab5,tab6, tab7 = st.tabs([ "Bliding", "Massaging", "Reweigh","LFR" ,"DisparateImpactRemover"])
            
        

            

        with tab3:
            include_columns = st.multiselect(
            "Select columns to be included during the training phase:", 
            df.columns.tolist(), 
            default= df.columns.tolist()
        )
        with tab4:
            sensitive_change = st.selectbox(
            "Enter the sensitive attribute to change target value", 
            df.columns.tolist(), 
            index=0)
            group_change = st.selectbox("Enter the category to chage target value:", st.session_state["df"][sensitive_change].unique().tolist(), key="category")
            number_change = st.number_input("Enter the amount of instaces to change", value=100)
    
        with tab5:
            protected_attribute_name_reweigh = st.selectbox(
            "Enter the sensitive attribute to reweigh", 
            df.columns.tolist(), 
            index=0)
            privileged_classes_reweigh = st.selectbox("Enter the privileged category:", st.session_state["df"][protected_attribute_name_reweigh].unique().tolist(), key="privileged category")
        
        with tab6:
            protected_attribute_name_lfr = st.selectbox(
            "Enter the sensitive attribute to LFR", 
            df.columns.tolist(), 
            index=0)
            privileged_classes_lfr = st.selectbox("Enter the privileged category:", st.session_state["df"][protected_attribute_name_lfr].unique().tolist(), key="privileged category lfr")
        
        with tab7:
            protected_attribute_name_dir = st.selectbox(
            "Enter the sensitive attribute to DisparateImpactRemover", 
            df.columns.tolist(), 
            index=0)
            privileged_classes_dir = st.selectbox("Enter the privileged category:", st.session_state["df"][protected_attribute_name_dir].unique().tolist(), key="privileged category dir")
            repair_level_dir = st.number_input("Enter the repair value", min_value=0.0, max_value=1.0, value=1.0)

    st.sidebar.markdown("""<h2>Data Resampling methods</h2>""", unsafe_allow_html=True)
    with st.sidebar.expander(""):
        tabr1, tabr2 = st.tabs([ "Resampling methods","Synthetic Data"])

        with tabr1:
            resampling_method = st.selectbox("Fair Resampling Method", ["None","Smote" ,"FairSmote", "Random", "ADASYN", "Borderline Smote", "Kmeans Smote", "SVM Smote"])

            cluster = 10
            if resampling_method == "Kmeans Smote":
                clusters = st.number_input("Enter number of clusters: ", value=10)
        with tabr2:
            sensitive_synt = st.selectbox(
            "Enter the sensitive attribute to generate syntetic samples", 
            df.columns.tolist(), 
            index=0)
            group_synt = st.selectbox("Enter the Unrepresented category:", st.session_state["df"][sensitive_synt].unique().tolist(), key="Unrepresented category")
            number_synt = st.number_input("Enter constant value", value=100)
            
        

    default_sensitive_columns = [colunas_lower[col] for col in colunas_lower if col in tipical_sensitive_information]



    return default_sensitive_columns,  numeric_strategy, custom_value, categorical_strategy, custom_value_cat, use_knn, use_iterative, use_rf,resampling_method, cluster, sensitive_synt, group_synt, number_synt, include_columns, sensitive_change, group_change, number_change, protected_attribute_name_reweigh, privileged_classes_reweigh, repair_level_dir, protected_attribute_name_dir, privileged_classes_dir, protected_attribute_name_lfr, privileged_classes_lfr

def extract_metrics(report):
            """Processa o relatório para extrair métricas de performance e fairness"""
            metrics = {}
            for line in report.split("\n"):
                if "Accuracy" in line:
                    model_name = line.split(" - ")[0].strip()
                    acc = float(line.split(":")[1].strip())
                    metrics[model_name] = acc
            return metrics

import streamlit as st

def display_class_distribution(data):
    """
    Exibe a distribuição de classes e destaca classes desbalanceadas no Streamlit.

    Args:
        data (list of dicts): Lista de dicionários com informações sobre a distribuição de classes.
    Returns:
        str: Resumo em string da distribuição e análise.
    """
    st.markdown("##### Class Distribution and Imbalance Check")

    info = "##### Class Distribution and Imbalance Check\n"
    for item in data:
        target = item["target_column"]
        st.write(f"###### Target Column: {target}")
        info += f"###### Target Column: {target}\n"

        # Exibir distribuição das classes
        st.write("**Class Distribution:**")
        info += "**Class Distribution:**\n"
        
        for class_name, percentage in item["class_distribution"].items():
            st.write(f"- {class_name.strip()}: **{percentage:.2f}%**")
            info += f"- {class_name.strip()}: {percentage:.2f}%\n"

        # Exibir classes desbalanceadas (se houver)
        if item["imbalanced_classes"]:
            st.write("**Imbalanced Classes:**")
            st.write(", ".join([cls.strip() for cls in item["imbalanced_classes"]]))
            info += "**Imbalanced Classes:**\n"
            info += ", ".join([cls.strip() for cls in item["imbalanced_classes"]]) + "\n"
        else:
            st.write("No imbalanced classes detected.")
            info += "No imbalanced classes detected.\n"

        st.markdown("---")  # Separador visual entre targets (se houver mais de um)

    return info

    
def show_plots(num_models, accuracy_anterior, accuracy_atual, accuracy_orig, column):
    import numpy as np
    import matplotlib.pyplot as plt

    # Determina índices comuns entre os DataFrames/Series
    common_indices = set(accuracy_atual.index) & set(accuracy_orig.index)
    if accuracy_anterior is not None:
        common_indices &= set(accuracy_anterior.index)
    
    # Ordena para manter consistência visual
    common_indices = sorted(common_indices)

    # Filtra os dados
    accuracy_atual = accuracy_atual.loc[common_indices]
    accuracy_orig = accuracy_orig.loc[common_indices]
    if accuracy_anterior is not None:
        accuracy_anterior = accuracy_anterior.loc[common_indices]

    num_models = len(common_indices)
    indices = np.arange(num_models)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars0 = ax.bar(indices - bar_width, accuracy_orig[column], bar_width, label="Original", color="gray")

    if accuracy_anterior is not None:
        bars1 = ax.bar(indices, accuracy_anterior[column], bar_width, label="Previous", color="blue")
        bars2 = ax.bar(indices + bar_width, accuracy_atual[column], bar_width, label="Current", color="orange")
    else:
        bars2 = ax.bar(indices, accuracy_atual[column], bar_width, label="Current", color="orange")

    ax.set_xticks(indices)
    ax.set_xticklabels(common_indices, rotation=45, ha="right")
    plt.ylabel(column)
    plt.title("Model Performance Comparison (Original vs. Previous vs. Current)")
    plt.ylim(0, 1)
    plt.legend()

    return fig


def show_plots_fairness(report_before, report_after, report_orig, column, protected_group, selected_fairness, ax=None, max_y = 1):
    unprotected_group = "not_"+str(protected_group)
    protected_group = str(protected_group)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    print(max_y)
    # Última linha da função (em vez de ax.set_ylim(0, 1)):
    if selected_fairness == "disparate_impact":
        ax.set_ylim(0, 1)
    else:
        if max_y is not None:
            print("aqui")
            ax.set_ylim(0, max_y * 1.05)  # leve margem visual

    models = list(report_after.keys())
    num_models = len(models)
    indices = np.arange(num_models)
    bar_width = 0.25

    #fig, ax = plt.subplots(figsize=(10, 6))

    has_previous = bool(report_before)

    orig_values, previous_values, current_values = [], [], []
    orig_colors, previous_colors, current_colors = [], [], []

    for model in models:
        orig_value = None
        prev_value = None
        curr_value = None

        if model in report_orig and column in report_orig[model]:
            orig_value = report_orig[model][column].get((protected_group, unprotected_group), {}).get(selected_fairness, None)

        if has_previous and model in report_before and column in report_before[model]:
            prev_value = report_before[model][column].get((protected_group, unprotected_group), {}).get(selected_fairness, None)

        if model in report_after and column in report_after[model]:
            curr_value = report_after[model][column].get((protected_group, unprotected_group), {}).get(selected_fairness, None)

        orig_values.append(abs(orig_value) if orig_value is not None else 0)
        previous_values.append(abs(prev_value) if prev_value is not None else 0)
        current_values.append(abs(curr_value) if curr_value is not None else 0)

        orig_colors.append("gray")
        previous_colors.append("darkblue" if prev_value is not None and prev_value < 0 else "blue")
        current_colors.append("darkorange" if curr_value is not None and curr_value < 0 else "orange")

    # Plotando as barras com offset
    ax.bar(indices - bar_width, orig_values, bar_width, color=orig_colors, label="Original")
    
    if has_previous and any(v > 0 for v in previous_values):
        ax.bar(indices, previous_values, bar_width, color=previous_colors, label="Previous")
        ax.bar(indices + bar_width, current_values, bar_width, color=current_colors, label="Current")
        legend_elements = [
            Patch(facecolor='gray', label='Original'),
            Patch(facecolor='blue', label='Previous Positive'),
            Patch(facecolor='darkblue', label='Previous Negative'),
            Patch(facecolor='orange', label='Current Positive'),
            Patch(facecolor='darkorange', label='Current Negative')
        ]
    else:
        ax.bar(indices, current_values, bar_width, color=current_colors, label="Current")
        legend_elements = [
            Patch(facecolor='gray', label='Original'),
            Patch(facecolor='orange', label='Current Positive'),
            Patch(facecolor='darkorange', label='Current Negative')
        ]

    ax.set_xticks(indices)
    ax.set_xticklabels(models, rotation=45, ha="right")
    plt.ylabel(selected_fairness)
    plt.title(f"Fairness Comparison: {selected_fairness} ({column})")
    #ax.set_ylim(0, 1)
    ax.legend(handles=legend_elements)

    return fig

def show_comparison_scatter_plot(accuracy_anterior, accuracy_atual, report_before, report_after, sensitive_attribute, protected_group, metric_1, metric_2):
    unprotected_group = "not_"+ protected_group
    models = list(report_after.keys())
    model_colors = plt.cm.get_cmap('tab10', len(models))  # Usando uma colormap para atribuir cores distintas

    fig, ax = plt.subplots(figsize=(10, 6))

    # Listas para armazenar os valores das métricas e as cores
    metric_1_values = []
    metric_2_values = []
    colors = []

    for idx, model in enumerate(models):
        # Para Accuracy
        if "accuracy" in metric_1.lower():
            # Acessando accuracy usando accuracy_anterior e accuracy_atual
            
            curr_value_1 = accuracy_atual.get(metric_1, None).get(model, {})
        else:
            curr_value_1 = report_after.get(model, {}).get(sensitive_attribute, {}).get((protected_group, unprotected_group), {}).get(metric_1, None)

        if "accuracy" in metric_2.lower():
            # Acessando accuracy usando accuracy_anterior e accuracy_atual
            curr_value_2 = accuracy_atual.get(metric_2, None).get(model, {})
        else:
            curr_value_2 = report_after.get(model, {}).get(sensitive_attribute, {}).get((protected_group, unprotected_group), {}).get(metric_2, None)

        # Usando o valor "current" (atual) e garantindo que seja absoluto
        metric_1_values.append(abs(curr_value_1) if curr_value_1 is not None else 0)
        metric_2_values.append(abs(curr_value_2) if curr_value_2 is not None else 0)
        
        # Atribuindo uma cor única para cada modelo
        colors.append(model_colors(idx))  # Atribuindo uma cor do colormap

    # Plotando os pontos no gráfico de dispersão
    scatter = ax.scatter(metric_1_values, metric_2_values, c=colors, label=models, s=100)  # 's' é o tamanho dos pontos

    # Labels e título
    plt.xlabel(metric_1)
    plt.ylabel(metric_2)
    plt.title(f"Comparison of Metrics in the current Model: {metric_1} vs {metric_2}")

    # Adicionando a legenda
    legend_elements = [Patch(facecolor=model_colors(i), label=model) for i, model in enumerate(models)]
    ax.legend(handles=legend_elements)

    return fig


def display_final_report(report):
    #FINAL REPORT
    if len(report["dataset_car"]) > 0 : 
        st.write("##### Dataset caracteristics")
        for i in  report["dataset_car"]:
            st.write(i+"\n\n\n")
    if len(report["preprocessing"]) > 0 : 
        st.write("\n\n\n\n\n##### Preprocessing methods applied")
        for change in range(len(report["preprocessing"])):
                st.markdown(f"{change + 1}º: {report['preprocessing'][change]}")
    else:
         st.write("\n\n\n\n\n##### Original Dataset")
    st.write("\n\n\n\n\n##### Inprocessing method applied")
    st.write("- "+report["inprocessing"])
    

# Função para selecionar as métricas e gerar o gráfico
def select_metrics_and_plot(selected_metrics_fairness, sensitive_columns):
    # Seleção das métricas (accuracy ou fairness)
    selected_metric_1 = st.selectbox("Choose first metric (accuracy or fairness):", ["Accuracy"] + selected_metrics_fairness, key="metric_1")
    selected_metric_2 = st.selectbox("Choose second metric (accuracy or fairness):", ["Accuracy"] + selected_metrics_fairness, key="metric_2")

    # Garantir que as métricas selecionadas são diferentes
    if selected_metric_1 == selected_metric_2:
        st.error("Please select two different metrics.")
        

    # Seleção do atributo sensível e dos grupos
    sensitive_attribute = st.selectbox("Choose sensitive attribute:", sensitive_columns, key="sensitive_attribute")
    protected_group = st.selectbox("Choose Protected Group:", st.session_state["df"][sensitive_attribute].unique().tolist(), key="protected_group")
    #unprotected_group = st.selectbox("Choose Unprotected Group:", [val for val in st.session_state["df"][sensitive_attribute].unique().tolist() if val != protected_group], key="unprotected_group")

    show_scatter = False
    # Plotar o gráfico quando o botão for clicado
    if st.button("Show Metrics Comparison Scatter Plot"):
        show_scatter = True
        # Chama a função de plotagem de gráficos de fairness e accuracy
       
    return sensitive_attribute, protected_group, selected_metric_1, selected_metric_2, show_scatter