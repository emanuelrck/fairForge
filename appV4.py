import streamlit as st
import pickle
import io
import joblib
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import accuracy_score
import seaborn as sns
from framework.reader import DataReader
from framework.analyzer import DataAnalyzer
from framework.model_trainer import ModelTrainer, experiment_fairness, postProcessing
import aux_func
from framework.data_preprocessing import impute_missing_values, oversampling, augment_minority_group, change_labels, reweigh, Dir, Learning_fair_representations, dataset_bias_metrics
import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval
import webbrowser
#from frontend import main_frontend, style_b1, style_b2, style_b3, style_b4, style_b5,style_b6
from sklearn.metrics import accuracy_score, classification_report



def main_frontend():
    st.markdown("""
    <style>

        button:hover {
           background-color: #F5F5F5 !important;
           color: rgb(50 93 121)  !important;
           border-color: rgb(50 93 121)  !important;
        }
        
        button:active {
           background-color: #F5F5F5 !important;
           color: rgb(50 93 121)  !important;
           border-color: rgb(50 93 121)  !important;
        }

        button:focus:not(:active) {
           background-color: #F5F5F5 !important;
           color: rgb(50 93 121)  !important;
           border-color: rgb(50 93 121)  !important;
        }

        .st-emotion-cache-1tokvoz {
            display: inline !important;
            transition: left 300ms;
            color: rgb(128, 132, 149);
            line-height: 0;
        }

        .st-emotion-cache-1h9usn1 {
            margin-bottom: 0px;
            margin-top: 0px;
            width: 100%;
            border-style: solid;
            border-width: 1px;
            border-color: rgba(49, 51, 63, 0.2);
            border-radius: 0.5rem;
            border: none;
        }

        .st-bs {
            background-color: rgb(225 238 244);
            border-color: #325d79;
        }


        .st-cg {
            background-color: #e1eef4 !important;
        }
        
        .st-d9 {
            background-color: rgb(50 93 121);
        }

        .st-d2 {
            background-color: #e1eef4 !important;
        }

        span.st-ar.st-cn.st-co.st-cp.st-cq {
            color: #325d79 !important;
        }
        .st-bn {
            color: rgb(50 93 121) !important;
        }

        .st-d3 {
            background-color: #e1eef4 !important;
        }

        span.st-emotion-cache-9ycgxx.e1blfcsg3 {
            color: #ffffff !important;
        }

        .st-emotion-cache-1aehpvj {
            color: rgb(255 255 255);
        }

        .st-emotion-cache-133trn5 {
            vertical-align: middle;
            overflow: hidden;
            fill: rgb(255 255 255);
            display: inline-flex;
            -webkit-box-align: center;
            align-items: center;
            font-size: 2.3rem;
            width: 2.3rem;
            height: 2.3rem;
            flex-shrink: 0;
        }

        .st-e5 {
            color: #325d79 !important;
        }

        .st-eb {
            color: rgb(50 93 121);
        }









        .st-emotion-cache-fsammq {
            font-family: "Source Sans Pro", sans-serif;
            font-size: 14px;
            color: #325d79 !important;
        }

        /* Remove Menu e Footer do Streamlit */
        #MainMenu {visibility: hidden; display: none;}
        footer {visibility: hidden; display: none;}
        header {visibility: hidden; display: none; position: fixed; padding:0;}

        /* Fundo da página */
        .stApp {
            background-color: #F5F5F5; /* Mesmo fundo do seu cabeçalho */
        }

        .st-emotion-cache-mtjnbi{
            width: 100%;
            padding: 0;
            max-width: 736px;
        }

        .st-emotion-cache-a6qe2i {
            background: #F5F5F5;
            border:none;
        }
        .st-emotion-cache-kgpedg{
            background: #F5F5F5;
            box-shadow: 4px 4px 10px rgba(50, 93, 121, 0.4);
        }
        section.stSidebar.st-emotion-cache-rpj0dg.e1c29vlm0 {
            background: #F5F5F5;
            box-shadow: 4px 4px 10px rgba(50, 93, 121, 0.4);
        }

        .st-emotion-cache-1y9tyez {
            z-index: 999990;
            color: rgb(250, 250, 250);
            margin-top: 0.25rem;
            visibility: visible;
        }

        button {
            background-color: white;
            color: #325D79;
            border: none;
            padding: 12px 24px;
            margin: 0 10px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 25px;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background 0.3s, color 0.3s, box-shadow 0.3s;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); 
        }

        section.st-emotion-cache-xwtqgq.e1blfcsg0 {
            background: #F5F5F5;
            padding: #F5F5F5;
            box-shadow: 4px 4px 10px rgba(50, 93, 121, 0.4);
        }

        .st-emotion-cache-19el2lr {
            display: inline-flex;
            -webkit-box-align: center;
            align-items: center;
            -webkit-box-pack: center;
            justify-content: center;
            font-weight: 400;
            border-radius: 0.5rem;
            margin: 0px 0.125rem;
            text-transform: none;
            font-family: inherit;
            color: inherit;
            width: auto;
            cursor: pointer;
            user-select: none;
            background-color: #325D79;
            border: none;
            font-size: 14px;
            line-height: 1;
            min-width: 2rem;
            min-height: 2rem;
            padding: 0px;
        }

        .st-emotion-cache-11lmpti {
            width: 100%;
            position: relative;
            display: flex;
            flex: 1 1 0%;
            flex-direction: column;
            gap: 0;}

        .st-emotion-cache-17m6xoq {
            display: inline;
            transition: left 300ms;
            color: rgb(250, 250, 250);
            line-height: 0;
            visibility: visible;
        }

        .st-emotion-cache-7z085q {
            color-scheme: normal;
            border: none;
            padding: 0px;
            margin: 0px;
            overflow: hidden;
            width:200%;
            margin-left:-50%

        }



        /* Estilo para os widgets (ex: sliders, inputs) */
        .stTextInput, .stNumberInput, .stSelectbox, .stMultiselect {
            border: 2px solid #325D79 !important;
            border-radius: 10px !important;
        }

        /* Títulos e textos */
        h1, h2, h3, h4, h5, h6 {
            color: #325D79 !important;
        }
        
        span {
            color: #325D79;
        }
        p{
            color: #325D79 !;
        }





        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            padding-top: 80px;  /* espaço para o header fixo */
            text-align: center;
        }}

        .fixed-header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #F5F5F5;
            display: flex;
            justify-content: center;
            padding: 15px 0;
            z-index: 1000;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        }}

        .nav-btn {{
            background-color: white;
            color: #325D79;
            border: none;
            padding: 12px 24px;
            margin: 0 10px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 25px;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background 0.3s, color 0.3s, box-shadow 0.3s;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }}

        .nav-btn:hover {{
            background-color: #E0E0E0;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        }}

        .active {{
            background-color: #325D79;
            color: white;
        }}

    </style>
    """, unsafe_allow_html=True)


    st.markdown("""
    <style>
    .st-emotion-cache-ocqkz7 {
        position: fixed;
        align-items: center;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #F5F5F5;
        display: flex;
        justify-content: center;
        padding: 20px 20%;
        z-index: 1000;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
    }

    .stButton>button {
        background-color: white;
        color: #325D79;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 25px;
        transition: all 0.3s ease;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        width: 90%; 
        margin-bottom: 2%;
        margin-top: 2%;
    }

    .stButton>button:hover {
        background-color: #E0E0E0;
        color:white
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button:active {
        background-color: #325D79;
        color: white;
        border-color:#325D79;
    }

    .conteudo {
        margin-top: -10px;
    }

    .st-emotion-cache-1s4qa0f {
        width: calc(15% - 1rem);
        flex: 10%;
    }

    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 100px;
    }

    .button-container .stButton {
        flex: 1;
        display: flex;
        justify-content: center;
    }
    
    .st-emotion-cache-9eid05 {
    width: 105.174px;
    position: relative;
    display: flex;
    flex: 1 1 0%;
    flex-direction: row;
    gap: 1rem;
    }

    .st-emotion-cache-1qrd9al p, .st-emotion-cache-1qrd9al ol, .st-emotion-cache-1qrd9al ul, .st-emotion-cache-1qrd9al dl, .st-emotion-cache-1qrd9al li {
    font-size: inherit;
    color: #325D79;
    }


    .st-emotion-cache-b0y9n5:focus:not(:active) {
    color: #325D79 !important;
    background-color: #E1EEF4;
    }
    
    .st-key-download_button button {
        padding-left: 37.5%;
        padding-right: 37.5%;
    }




    %Sidebar
    
    .st-dv {
        background-color: #E0E0E0;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        border-color: #325D79;
    }   
    span {
        color: white !important;
    }
    .st-emotion-cache-s1invk:hover {
        color: #325D79;
    }
    .st-emotion-cache-b0y9n5:focus:not(:active) {
        border-color: #325D79;
        color:#325D79;
    }
    .st-emotion-cache-b0y9n5:hover {
    border-color: #325D79;
    color: #325D79;
    }
    .st-d1 {
        color: #325D79;
    }
    
    .st-ga {
        color: #325D79;
    }
    .st-emotion-cache-124cvek {
        display: flex;
        -webkit-box-align: center;
        align-items: center;
        padding: 0px 1rem;
        background: #a0a2a2;
    }
    .st-emotion-cache-124cvek:hover {
        background: #a0a2a2;
    }
    .stTextInput, .stNumberInput, .stSelectbox, .stMultiselect {
        border: 2px solid rgba(0, 0, 0, 0) !important;
        border-radius: 10px !important;
    }
    .st-f4 {
        background-color: #a0a2a2;
    }

    .st-emotion-cache-1b2ybts {
        vertical-align: middle;
        overflow: visible;
        fill: currentcolor;
        display: inline-flex;
        -webkit-box-align: center;
        align-items: center;
        font-size: 1.25rem;
        width: 1.25rem;
        height: 1.25rem;
        flex-shrink: 0;
    }
    .st-emotion-cache-s1invk {
        color: #325D79;
        visibility: visible;
    }
    .st-emotion-cache-s1invk:hover {
        color: #325D79;
        visibility: visible;
    }
    .st-emotion-cache-1l9kk8e {
        background-color: #e1eef4;
        border-radius: 1px;
    }
    .st-emotion-cache-7oyrr6 {
        color: #325D79;
        font-size: 14px;
        line-height: 1.25;
    }


    .st-emotion-cache-1l9kk8e:hover:enabled, .st-emotion-cache-1l9kk8e:focus:enabled {
        color: rgb(255, 255, 255);
        background-color: #325D79;
        transition: none;
        outline: none;
    }
    .st-emotion-cache-s1invk:hover svg {
        fill: #325D79;
    }
    .st-cx {
        white-space: nowrap;
        padding-left: 0px;
        padding-right: 0px;
    }

    .st-emotion-cache-13na8ym {
        margin-bottom: 10px;
        margin-top: -40px;
        padding: 
        width: 100%;
        border-style: solid;
        border-width: 1px;
        border-color: rgba(250, 250, 250, 0.2);
        border-radius: 0.5rem;
    }
    
    .st-emotion-cache-1clstc5 {
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        
    }
    
    .st-emotion-cache-1espb9k h2 {
        font-size: 1.25rem;
        padding: 0;
    }
    
    
    %Upload
    .st-emotion-cache-fis6aj {
        left: 0px;
        right: 0px;
        line-height: 1.25;
        padding-top: 0.75rem;
        padding-left: 1rem;
        padding-right: 1rem;
        visibility: hidden;
        background: red;
    }
    .st-emotion-cache-1erivf3 {
        display: flex;
        -webkit-box-align: center;
        align-items: center;
        padding: 1rem;
        background-color: #325D79;
        border-radius: 0.5rem;
        color: rgb(250, 250, 250);
    }

    .st-emotion-cache-b0y9n5 {
        display: inline-flex;
        -webkit-box-align: center;
        align-items: center;
        -webkit-box-pack: center;
        justify-content: center;
        font-weight: 400;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        min-height: 2.5rem;
        margin: 0px;
        line-height: 1.6;
        text-transform: none;
        font-size: inherit;
        font-family: inherit;
        color: #325D79;
        width: auto;
        cursor: pointer;
        user-select: none;
        background-color: white;
        border: 1px solid rgba(250, 250, 250, 0.2);
    }
    
    .stButton {
        display: flex;
        justify-content: center;
    }
    
    .st-emotion-cache-14m29r0 {
        list-style-type: none;
        margin: 0px;
        padding: 0px;
        visibility: collapse;
    }
    .st-emotion-cache-14m29r0 {
        list-style-type: none;
        margin: 0px;
        padding: 0px;
        /* visibility: collapse; */
        /* height: 2px; */
        color: #325D79;
        background-color: #E1EEF4;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        border-radius: 25px;
        padding-right: 30px;
        padding-left: 30px;
    }
    .st-ds{
        background-color:#e1eef4;
    }
    .st-emotion-cache-9gx57n.focused{
    border-color:black ;
    }
    svg.st-dr.st-e6.st-es.st-et.st-eu {
        background-color: #325d79;
    }
    .stAlertContainer.st-c3.st-cs.st-c5.st-c6.st-c7.st-c8.st-g0.st-g1.st-dt.st-g2.st-cp.st-g3.st-g4.st-g5.st-ai.st-aj.st-g6.st-cu.st-cw.st-cx.st-cv.st-ds.st-g9.st-fo.st-bu.st-g7.st-af.st-cy.st-ah.st-ag.st-g8{
        background-color: #E1EEF4 !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
        
    }
    .stAlertContainer.st-c3.st-cs.st-c5.st-c6.st-c7.st-c8.st-g3.st-g4.st-dt.st-gf.st-cp.st-g6.st-g7.st-g8.st-ai.st-aj.st-g9.st-cu.st-cw.st-cx.st-cv.st-ds.st-gc.st-fo.st-bu.st-ga.st-af.st-cy.st-ah.st-ag.st-gb{
        background-color: #E1EEF4 !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
    }

    

    .st-emotion-cache-14m29r0 {
        list-style-type: none;
        margin: 0px;
        padding: 0px;
        visibility: visible;
    }
    
    .st-emotion-cache-12xsiil {
        display: flex;
        -webkit-box-align: center;
        align-items: center;
        margin-bottom: 0.25rem;
        padding-right: 30px;
        padding-left: 30px;
    }

    .st-emotion-cache-clky9d {
        vertical-align: middle;
        overflow: hidden;
        fill: #325D79;
        display: inline-flex;
        -webkit-box-align: center;
        align-items: center;
        font-size: 1.8rem;
        width: 1.8rem;
        height: 1.8rem;
        flex-shrink: 0;
    }
    
    
    .st-cg {
        background-color: #325D79;
    }
    
    %Processamento
    .st-g9 {
        background-color: #E1EEF4 !important;
    }
    .st-emotion-cache-1cvow4s {
        font-family: "Source Sans Pro", sans-serif;
        font-size: 1rem;
        margin-bottom: -1rem;
        color: inherit;
        padding-left: 3%;
    }

    .st-emotion-cache-1qg05tj {
        font-size: 14px;
        color: #325d79;
        display: flex;
        visibility: visible;
        margin-bottom: 0.25rem;
        height: auto;
        min-height: 1.5rem;
        vertical-align: middle;
        flex-direction: row;
        -webkit-box-align: center;
        align-items: center;
    }
    .st-cx {
        white-space: nowrap;
        padding-left: 0px;
        padding-right: 0px;
        background-color: #e1eef4 !important;
        color :#325D79;
    }
    .st-gd {
        color: #325D79;
        border-color: #325D79 !important;
    }
    .st-fx {
        fill: #325D79;
    }
    .st-gq {
        color:  #325D79;
    }
    .st-e5 {
        color: #E1EEF4;
    }
    svg.st-dq.st-e5.st-fq.st-fr.st-fs {
        background-color: #325D79;
    }
    .st-cf {
        color: #325D79; !important;
        fill: #325D79; !important;
        border-color: #325D79 !important;
    }
    .st-dr {
        background-color: #e1eef4;
    }
    .st-fy{
        color:#325D79 !important;
        border-color: #325D79 !important;
    }
    .st-emotion-cache-ah6jdd {
        font-family: "Source Sans Pro", sans-serif;
        color: #325D79;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .st-emotion-cache-1cvow4s p, .st-emotion-cache-1cvow4s ol, .st-emotion-cache-1cvow4s ul, .st-emotion-cache-1cvow4s dl, .st-emotion-cache-1cvow4s li {
        font-size: inherit;
        color: #325D79;
    }

    .st-cb {
        white-space: nowrap;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .st-emotion-cache-8lz9yt {
        vertical-align: middle;
        overflow: hidden;
        fill: #325D79;
        display: inline-flex;
        -webkit-box-align: center;
        align-items: center;
        font-size: 0.5rem;
        width: 0.5rem;
        height: 0.5rem;
        flex-shrink: 0;
    }
    
    
    
    
    
    % Results
    .custom-history-button > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .custom-history-button > button:hover {
        background-color: #45a049;
    }


    .st-emotion-cache-jh76sn:focus:not(:active) {
        border-color: #325D79;
        color:#325D79;
    }
    

    .st-emotion-cache-1y1yxp6 {
        width: 138.213px;
        position: relative;
        display: flex;
        flex: 1 1 0%;
        flex-direction: column;
        gap: 0rem;
    }

    .st-emotion-cache-1aehpvj {
        color: rgb(50 93 121);
    }

    .st-emotion-cache-xxjg8 {
        width: 217.594px;
        position: relative;
        display: flex;
        flex: 1 1 0%;
        flex-direction: column;
        gap: 0rem;
    }

    .st-emotion-cache-1gulkj5{
    background-color: rgb(50 93 121);
    }

    .st-emotion-cache-1vsah7k p, .st-emotion-cache-1vsah7k ol, .st-emotion-cache-1vsah7k ul, .st-emotion-cache-1vsah7k dl, .st-emotion-cache-1vsah7k li {
        font-size: inherit;
        color: #325d79;
    }
    st-emotion-cache-9gx57n:focused{
        border-color: #325d79
    }
    .st-emotion-cache-1espb9k p{
        font-size: inherit;
        margin-left: 4%;
    }
    
    
    </style>
    """, unsafe_allow_html=True)

    # Layout dos botões dentro da div
    st.markdown('<div class="button-container">', unsafe_allow_html=True)

    if "b1" not in st.session_state:
        st.session_state["b1"] = True
    if "b2" not in st.session_state:
        st.session_state["b2"] = False
    if "b3" not in st.session_state:
        st.session_state["b3"] = False
    if "b4" not in st.session_state:
        st.session_state["b4"] = False
    if "b5" not in st.session_state:
        st.session_state["b5"] = False
    if "b6" not in st.session_state:
        st.session_state["b6"] = False

    col1, col2, col5 ,col3, col6, col4  = st.columns(6)

    with col1:
        if st.button('✔ Upload', key='botao1', on_click = style_b1):
            st.session_state["b1"] = True
            st.session_state["b2"] = False
            st.session_state["b3"] = False
            st.session_state["b4"] = False
            st.session_state["b5"] = False
            st.session_state["b6"] = False
            st.markdown("""
        <style>
        
        }
        </style>
            """, unsafe_allow_html=True)

    with col2:
        if st.button('✔ Preprocessing', key='botao2', on_click = style_b2):
            st.session_state["b2"] = True
            st.session_state["b1"] = False
            st.session_state["b3"] = False
            st.session_state["b4"] = False
            st.session_state["b5"] = False
            st.session_state["b6"] = False
    with col5:
        if st.button('✔ Inprocessing', key='botao5', on_click = style_b5):
            st.session_state["b5"] = True   
            st.session_state["b1"] = False
            st.session_state["b3"] = False
            st.session_state["b2"] = False
            st.session_state["b4"] = False
            st.session_state["b6"] = False
            
    with col3:
        if st.button('✔ Training', key='botao3', on_click = style_b3):
            st.session_state["b3"] = True
            st.session_state["b1"] = False
            st.session_state["b2"] = False
            st.session_state["b4"] = False
            st.session_state["b5"] = False
            st.session_state["b6"] = False
    with col4:
        if st.button('✔ Results', key='botao4', on_click = style_b4):
            st.session_state["b4"] = True
            st.session_state["b1"] = False
            st.session_state["b3"] = False
            st.session_state["b2"] = False
            st.session_state["b5"] = False
            st.session_state["b6"] = False
    with col6:
        if st.button('✔ PostProcessing', key='botao6', on_click = style_b6):
            st.session_state["b4"] = False
            st.session_state["b1"] = False
            st.session_state["b3"] = False
            st.session_state["b2"] = False
            st.session_state["b5"] = False
            st.session_state["b6"] = True

def style_b1():
    st.markdown("""
    <style>
    
    div.st-key-botao1 button {
        background-color: #325D79 !important;
        color: white !important;

    }
    .st-key-botao1 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }
    
    div.st-key-continue0 button {
        background-color: #325D79 !important;
        color: white !important;
    }
    .st-key-continue0 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }

    </style>
    """, unsafe_allow_html=True)
def style_b2():
    st.markdown("""
    <style>
    .st-key-botao1 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao2 button {
        background-color: #325D79 !important; 
        color: white !important;
    }
    .st-key-botao2 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }

    div.st-key-continue0 button {
        background-color: #325D79 !important;
        color: white !important;
    }
    .st-key-continue0 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
def style_b3():
    st.markdown("""
    <style>
    .st-key-botao1 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao2 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao3 button {
        background-color: #325D79 !important; 
        color: white !important;
    }
    .st-key-botao3 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }
      .st-key-botao5 button {
        background-color:  #e1eef4 !important;
    }
    div.st-key-continue0 button {
        background-color: #325D79 !important;
        color: white !important;
    }
    .st-key-continue0 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
def style_b4():
    st.markdown("""
    <style>
    
    .st-key-botao1 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao2 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao3 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao6 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao4 button {
        background-color: #325D79 !important; 
        color: white !important;
    }
    .st-key-botao4 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }
    .st-key-botao5 button {
        background-color:  #e1eef4 !important;
    }
    div.st-key-continue0 button {
        background-color: #325D79 !important;
        color: white !important;
    }
    .st-key-continue0 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
def style_b5():
    st.markdown("""
    <style>
    .st-key-botao1 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao2 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao5 button {
        background-color: #325D79 !important; 
        color: white !important;
    }
    .st-key-botao5 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }
    div.st-key-continue0 button {
        background-color: #325D79 !important;
        color: white !important;
    }
    .st-key-continue0 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


def style_b6():
    st.markdown("""
    <style>
    
    .st-key-botao1 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao2 button {
        background-color:  #e1eef4 !important;
    }
    .st-key-botao3 button {
        background-color:  #e1eef4 !important;
    }
    
    .st-key-botao6 button {
        background-color: #325D79 !important; 
        color: white !important;
    }
    .st-key-botao6 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }
    .st-key-botao5 button {
        background-color:  #e1eef4 !important;
    }
    div.st-key-continue0 button {
        background-color: #325D79 !important;
        color: white !important;
    }
    .st-key-continue0 button:focus:not(:active) {
        background-color: #325D79 !important; 
        color: white !important;
    }

    .st-em:hove{
        color: white !important;}

    .st-emotion-cache-ocsh0s {
        border-color: rgb(50 93 121);
        color: rgb(50 93 121);
    }

    .st-emotion-cache-ocsh0s:hover {
        border-color: rgb(50 93 121)  !important;
        color: rgb(50 93 121)  !important;
    }
    .st-emotion-cache-ocsh0s:focus:not(:active) {
        border-color: rgb(50 93 121) !important;
        color: rgb(50 93 121) !important;
    }

    .st-emotion-cache-ocsh0s:active {
        border-color: rgb(50 93 121);
        color: rgb(50 93 121);
        background-color: rgb(225 238 244);
    }

    .st-emotion-cache-1hyd1ho:hover {
        color: rgb(50 93 121)  !important;
    }
    

    </style>
    """, unsafe_allow_html=True)




# File paths for saving reports
imbalance_report_file = "reports/imbalance_report.csv"
missing_values_report_file = "reports/missing_values_report.csv"
model_report_file = "reports/model_performance.csv"
fairness_report_file = "reports/fairness_report.csv"

#-----------------------------------------------------

st.set_page_config(page_title="JS-Python Page Switcher")

# Initialize the session state
if "pagina" not in st.session_state:
    st.session_state.pagina = "upload"

selected_page = streamlit_js_eval(js_expressions="window.selectedPage", key="eval_page")

if selected_page and selected_page != st.session_state.pagina:
    st.session_state.pagina = selected_page

#-----------------------------------------------------


# Função para abrir o link em uma nova aba
def abrir_link(url):
    webbrowser.open(url, new=2)

# Configuração do Streamlit
#st.set_page_config(page_title="ML Framework", layout="wide")

main_frontend()



st.markdown('</div>', unsafe_allow_html=True)

def b0():
    st.session_state["b2"] = True
    st.session_state["b1"] = False
    st.session_state["b3"] = False
    st.session_state["b4"] = False
    st.session_state["b5"] = False
    st.session_state["b6"] = False
def b1():
    st.session_state["b5"] = True   
    st.session_state["b1"] = False
    st.session_state["b3"] = False
    st.session_state["b2"] = False
    st.session_state["b4"] = False
    st.session_state["b6"] = False
def b2():
    st.session_state["b3"] = True
    st.session_state["b1"] = False
    st.session_state["b2"] = False
    st.session_state["b4"] = False
    st.session_state["b5"] = False
    st.session_state["b6"] = False
def b3():
    st.session_state["b4"] = False
    st.session_state["b1"] = False
    st.session_state["b3"] = False
    st.session_state["b2"] = False
    st.session_state["b5"] = False
    st.session_state["b6"] = True
def b4():
    st.session_state["b4"] = True
    st.session_state["b1"] = False
    st.session_state["b3"] = False
    st.session_state["b2"] = False
    st.session_state["b5"] = False
    st.session_state["b6"] = False
 

def tipical_sensitive_information():
    return {
        "gender", "sex", "biological_sex", "assigned_sex", "sex_at_birth", "gender_identity",
        "gender_expression", "gender_category", "race", "ethnicity", "ethnic_group",
        "racial_identity", "cultural_background", "dob",
        "year_of_birth", "religion", "belief", "faith", "spirituality", "religious_affiliation", "sex9", "marital", "marital_status"
    }

def compute_max_fairness_value(automatic_results, selected_measures, column, protected_group):
        unprotected_group = "not_" + str(protected_group)
        max_value = 0
        for fairness_measure in selected_measures:
            if fairness_measure == "disparate_impact":
                continue
            for key in automatic_results:
                _, fairness_dict = automatic_results[key]
                if column in fairness_dict["XGBoost"]:

                    value = fairness_dict["XGBoost"][column].get((protected_group, unprotected_group), {}).get(fairness_measure, None)
                    if value is not None:
                        max_value = max(max_value, abs(value))
        return max_value

available_fairness_metrics = ["equal_opportunity", "predictive_equality", "positive_predictive_parity", "true_positive_rate", "statistical_parity" , "disparate_impact"]     
available_models = [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "SVM",
            "MLP (Neural Network)"
        ]

def automatic():
    df = st.session_state["df"]
    st.session_state["automatic"] = True
    missing_data = DataAnalyzer.check_missing_values(st.session_state["df"], file_path=missing_values_report_file)
    colunas_lower = {col.lower(): col for col in st.session_state["df"].columns}
    sensitive_col = [colunas_lower[col] for col in colunas_lower if col in tipical_sensitive_information()]
    print(sensitive_col)
    clusters = 1
    for col in sensitive_col:
        clusters *= len(st.session_state["df"][col].unique())
    print("\n\n\n---------Clust-------")
    print(clusters)
    progress = st.progress(0)
    #se existir missing values 
    if missing_data.iloc[0, 1] > 0:
        
            
        
        dados_automaticos = st.session_state["df"]
        missing_possibilities_names = ["Original","MICE"]
        automatic_combinations = ["Original", "MICE","Smote", "MICE & Smote"," LFR", "MICE & Smote & LFR"]
        for i in range(len(automatic_combinations)):
            progress.progress((i + 1) / len(automatic_combinations))
            dados_automaticos = st.session_state["df"]
            if i == 0:
                dados_automaticos = dados_automaticos
            if i == 1 or i == 3 or i == 5:
                dados_automaticos, method = impute_missing_values(
                        dados_automaticos,  # O dataframe com os valores ausentes
                        numeric_strategy="",
                        categorical_strategy="",
                        custom_value="",
                        use_knn=False,
                        use_iterative=True,
                        use_rf=False
                    )
            if i == 2 or i == 3 or i == 5:
                dados_automaticos = oversampling(dados_automaticos, 
                target_column=df.columns.tolist()[-1], 
                sensitive_columns=sensitive_col, 
                method="Smote",
                clusters=clusters)
            
            if i == 4 or i == 5:
                dados_automaticos, _ , _ = Learning_fair_representations(dados_automaticos,
                target = df.columns.tolist()[-1],
                favorable_classes = dados_automaticos[df.columns.tolist()[-1]].unique().tolist()[0],
                protected_attribute_name  = sensitive_col[0], 
                privileged_classes = dados_automaticos[sensitive_col[0]].unique().tolist()[0],
                drop_columns =  sensitive_col.copy())

                
            st.session_state["sensitive_columns"] = sensitive_col
            print("\n\n\n-----------Sensitive")
            print(st.session_state["sensitive_columns"])
            st.session_state["selected_metrics_fairness"] = available_fairness_metrics
            
            modelo_automatico = ModelTrainer(dados_automaticos, df.columns.tolist()[-1], sensitive_col, test_size=0.3, random_state=42, selected_models="XGBoost", train_columns = df.columns.to_list(), favorable_classes_target = st.session_state["favorable_classes_target"], )
            st.session_state["current_model"] = modelo_automatico
            #modelo_automatico.fairness_method = "adversarial_debiasing"
            #modelo_automatico.fairness_params = {"eta": 25}
            #modelo_automatico.sensitive_attr = sensitive_col[0]

            print(sensitive_col)
            perfom_aut, fair_aut  = modelo_automatico.train_and_evaluate(selected_fairness=available_fairness_metrics, file_path = model_report_file, fairness_file_path = fairness_report_file)

            perfom_aut_write = pd.DataFrame.from_dict(aux_func.extract_metrics(perfom_aut), orient="index", columns=["Accuracy"])
            
            st.session_state["automatic_results"][automatic_combinations[i]] = [perfom_aut_write, fair_aut]
            
            
            st.session_state["b4"] = True
            st.session_state["b1"] = False
        
    else:
            automatic_combinations = ["Original","Smote"," LFR", "Smote & LFR"]
            for i in range(len(automatic_combinations)):
                progress.progress((i + 1) / len(automatic_combinations))
                dados_automaticos = st.session_state["df"]

                if i == 1 or i == 3:
                    dados_automaticos = oversampling(dados_automaticos, 
                    target_column=df.columns.tolist()[-1], 
                    sensitive_columns=sensitive_col, 
                    method="Smote",
                    clusters=clusters)
                
                if i == 2 or i == 3:
                    dados_automaticos, _ , _ = Learning_fair_representations(dados_automaticos,
                    target = df.columns.tolist()[-1],
                    favorable_classes = dados_automaticos[df.columns.tolist()[-1]].unique().tolist()[0],
                    protected_attribute_name  = sensitive_col[0], 
                    privileged_classes = dados_automaticos[sensitive_col[0]].unique().tolist()[0],
                    drop_columns =  sensitive_col.copy())

           

                st.session_state["sensitive_columns"] = sensitive_col
                st.session_state["selected_metrics_fairness"] = available_fairness_metrics
                
                modelo_automatico = ModelTrainer(dados_automaticos, df.columns.tolist()[-1], sensitive_col, test_size=0.2, random_state=42, selected_models="XGBoost", train_columns = df.columns.to_list(),  favorable_classes_target = st.session_state["favorable_classes_target"], sample_weight = st.session_state["instance_weights"])

        

                perfom_aut, fair_aut  = modelo_automatico.train_and_evaluate(selected_fairness=available_fairness_metrics, file_path = model_report_file, fairness_file_path = fairness_report_file)

                perfom_aut_write = pd.DataFrame.from_dict(aux_func.extract_metrics(perfom_aut), orient="index", columns=["Accuracy"])
            
                st.session_state["automatic_results"][automatic_combinations[i]] = [perfom_aut_write, fair_aut]
            
            
            st.session_state["b4"] = True
            st.session_state["b1"] = False
    st.session_state.current_model = modelo_automatico
    return



st.markdown('<div class="conteudo">', unsafe_allow_html=True)
if st.session_state["b1"]:
    st.subheader("Improve Data")
    style_b1()
    # 📂 Upload dataset
    uploaded_file = st.file_uploader(
        "Upload a CSV, JSON, Excel, or DATA file", 
        type=["csv", "json", "xlsx", "data"], key="uploader1"
    )
    shown = False


    
    if uploaded_file:
        #if st.button("Load Dataset"):
            with st.spinner("Loading dataset..."):
                print("inicio")
                df = DataReader.read_data(uploaded_file)
                
                # Salvar o dataset no session_state
                st.session_state["df"] = df
                #st.session_state["df_previus"] = None  # Resetar histórico de edições
                st.session_state["df_original"] = df.copy()  # Manter a versão original
                if "saved_models" not in st.session_state:
                    st.session_state["saved_models"] = {}

                colunas_lower = {col.lower(): col for col in df.columns}
                st.session_state["changes"] = []
                st.session_state["current_model"] = None
                st.session_state["columns_train"] = df.columns.to_list()
                st.session_state["default_target_column"] = df.columns.tolist()[-1]
                st.session_state["favorable_classes_target"] = df[st.session_state["default_target_column"]].unique()[1]

                print("\n\n\n-----------AQUI----")
                print(st.session_state["favorable_classes_target"])
                st.session_state["sensitive_columns"] = [colunas_lower[col] for col in colunas_lower if col in tipical_sensitive_information()]
                st.session_state["selected_models"] = available_models
                st.session_state["selected_metrics_fairness"] = available_fairness_metrics
                st.session_state["inprocessing_method"] = None
                st.session_state["inprocessing_params"] = None
                st.session_state["inprocessing_sensitive_attr"] = None
                st.session_state["inprocessing_models"] = None
                st.session_state["inprocessing_eta"] = None
                st.session_state["postprocessing_method"] = None
                st.session_state["final_report"] = {}
                st.session_state["instance_weights"] = []
                st.session_state["final_report"]["dataset_car"] = []
                st.session_state["final_report"]["preprocessing"] = []
                st.session_state["final_report"]["inprocessing"] = "None"
                st.session_state["final_report"]["results"] = []
                st.session_state["automatic"] = False
                st.session_state["automatic_results"] = {}
                if "atual_final_report" not in st.session_state:
                    st.session_state["atual_final_report"]={} 
                    st.session_state["previus_final_report"]={}


            #st.success("Dataset uploaded successfully!")
            st.write("### Dataset Preview:")
            st.write(df.head())


            shown = True
            #st.session_state["b1"] = False
            #st.session_state["b2"] = True
            #st.session_state["b3"] = False
            #st.session_state["b4"] = False
            #st.session_state["b5"] = False
            st.button('Automatic', on_click = automatic)
            #st.button('Continue', key='continue0', on_click = b0)



            st.sidebar.markdown("""<h2>Attribute Specifications</h2>""", unsafe_allow_html=True)
            with st.sidebar.expander("", expanded=False):
                    st.session_state["default_target_column"] = st.selectbox(
                        "Choose target column:", 
                        df.columns.tolist(), 
                        index=len(df.columns) - 1
                    )
                    st.session_state["favorable_classes_target"] = st.selectbox("Enter the privileged category:",st.session_state["df"][st.session_state["default_target_column"]].unique().tolist(), index =1, key="favorable_classes")
                    colunas_lower = {col.lower(): col for col in df.columns}
                    default_sensitive_columns = [colunas_lower[col] for col in colunas_lower if col in tipical_sensitive_information()]

                    st.session_state["sensitive_columns"] = st.multiselect(
                        "Select sensitive attributes:", 
                        df.columns.tolist(), 
                        default=default_sensitive_columns
                    )
                    st.session_state["priveleged_classes"] = []
                    for col in st.session_state["sensitive_columns"]:
                        st.session_state["priveleged_classes"].append(st.selectbox("Enter the priveleged class of "+col, st.session_state["df"][col].unique().tolist()))
        




    #------------------------------------ MODEL
    st.subheader("Test Model")
    prediction_dataset = None
    prediction_dataset_raw = st.file_uploader(
            "Upload a CSV, JSON, Excel, or DATA file", 
            type=["csv", "json", "xlsx", "data"], key="uploader2"
        )
    #st.sidebar.markdown("""<h2>Test Model</h2>""", unsafe_allow_html=True)
    if prediction_dataset_raw:
        prediction_dataset = DataReader.read_data(prediction_dataset_raw)
    with st.sidebar.expander("**Test Model**", expanded=False):
        # 📂 Upload dataset
        if prediction_dataset is not None:
            TestModel_predictions = st.selectbox(
                    "Choose target column:", 
                    prediction_dataset.columns.tolist(), 
                    index=len(prediction_dataset.columns) - 1
                )
            TestModel_target = st.selectbox(
                    "Choose target column:", 
                    prediction_dataset.columns.tolist(), 
                    index=len(prediction_dataset.columns) - 2
                )
            TestModel_target_positive = st.selectbox("Enter the privileged category:",prediction_dataset[TestModel_target].unique().tolist())

            TestModel_sens = st.multiselect(
                    "Select sensitive attributes:", 
                    prediction_dataset.columns.tolist(), 
                    default=[prediction_dataset.columns.tolist()[0]]
                )
            TestModel_priveledge = []
            for col in TestModel_sens:
                    TestModel_priveledge.append(st.selectbox("Enter the priveleged class of "+col, prediction_dataset[col].unique().tolist()))

       # else: 
        #    st.sidebar.markdown("""<h3>Load prediction dataset</h3>""", unsafe_allow_html=True)

    if prediction_dataset is not None and st.button("Check Fairness"):
        with st.spinner("Calculating fairness metrics..."):
            dicio_all_fair = experiment_fairness(
                predictions=prediction_dataset.iloc[:, -1].to_numpy(),
                name="",
                sensitive_columns=TestModel_sens,
                target=TestModel_target,
                positive_target=TestModel_target_positive,
                selected_fairness=available_fairness_metrics,
                test_dataset=prediction_dataset
            )
            print(dicio_all_fair)
            aux_func.show_fairness_test_model(dicio_all_fair)
        st.success("Fairness check completed.")







    if "df" in st.session_state and not shown:
        st.write("### Dataset Preview:")
        st.write(st.session_state["df"].head())


    st.button('Continue', key='continue0', on_click = b0)
   
            

elif st.session_state["b2"]:
    style_b2()
    st.session_state["b1"] = False
    st.session_state["b2"] = True
    st.session_state["b3"] = False
    st.session_state["b4"] = False
    st.session_state["b5"] = False
    st.session_state["b6"] = False
    
    
    if "df" in st.session_state:
    
    #---------------------------------Especificações do utilizador
    #   Valores selecionados pelo user
        default_sensitive_columns, numeric_strategy, custom_value, categorical_strategy, custom_value_cat, use_knn, use_iterative, use_rf, resampling_method, clusters, sensitive_synt, group_synt, number_synt, include_columns, sensitive_change, group_change, number_change, protected_attribute_name_reweigh, privileged_classes_reweigh, repair_level_dir, protected_attribute_name_dir, privileged_classes_dir, protected_attribute_name_lfr, privileged_classes_lfr = aux_func.display_categorys(st.session_state["df"])


        priveleged_classes = st.session_state["priveleged_classes"]
        favorable_classes_target = st.session_state["favorable_classes_target"]
        default_target_column = st.session_state["default_target_column"]
        sensitive_columns = st.session_state["sensitive_columns"]
        selected_models = st.session_state["selected_models"]
        selected_metrics_fairness = st.session_state["selected_metrics_fairness"]

        if st.button("Check Bias"):
            dataset_bias_metrics(st.session_state["df"],sensitive_columns,priveleged_classes,default_target_column ,favorable_classes_target)
            
        st.subheader("Missing Data Handling")
        if st.button("Check Missing Values"):
            with st.spinner("Checking for missing values..."):
                missing_data = DataAnalyzer.check_missing_values(st.session_state["df"], file_path=missing_values_report_file)
            st.text(missing_data)
            aux_info = ["#### Missing Values:\n"]
            for _, row in missing_data.iterrows():
                aux_info.append(f" - {row['column']}: **{row['missing_percentage']:.2f}%** \n")
            missing_info_str = "".join(aux_info)
            st.session_state["final_report"]["dataset_car"].append(missing_info_str)
            st.success("Missing values report saved!")

        if st.button("Impute Missing Values"):
            with st.spinner("Imputing missing values..."):
                if "df" in st.session_state:
                    st.session_state["df_previous"] = st.session_state["df"]
                st.session_state["df"], method = impute_missing_values(
                    st.session_state["df"],
                    numeric_strategy=numeric_strategy,
                    categorical_strategy=categorical_strategy,
                    custom_value=custom_value,
                    use_knn=use_knn,
                    use_iterative=use_iterative,
                    use_rf=use_rf
                )
                st.session_state["changes"].append("Imputed missing values with: " + method)
            st.success("Missing values imputed!")

        st.subheader("Class Imbalance Handling")
        if st.button("Analyze Imbalance"):
            with st.spinner("Running class imbalance analysis..."):
                report = DataAnalyzer.analyze_multiple_targets(st.session_state["df"], sensitive_columns, threshold=0.05, file_path=imbalance_report_file)
                st.session_state["final_report"]["dataset_car"].append(aux_func.display_class_distribution(report))
            st.success("Imbalance report saved!")

        if st.button("Resample"):
            st.session_state["df_previus"] = st.session_state["df"]
            st.session_state["changes"].append(f"Remsampled with {resampling_method}")
            with st.spinner(f"Applying {resampling_method}..."):
                st.session_state["df"] = oversampling(st.session_state["df"], 
                                                        target_column=default_target_column, 
                                                        sensitive_columns=sensitive_columns, 
                                                        method=resampling_method, clusters=clusters)
            st.success(f"{resampling_method} applied!")

        if st.button("Generate syntetic data"):
            st.session_state["df_previus"] = st.session_state["df"]
            st.session_state["changes"].append(f"Generated {number_synt} syntetic data for  {sensitive_synt} -> {group_synt}")
            with st.spinner(f"Generating syntetic data for  {sensitive_synt} -> {group_synt}..."):
                st.session_state["df"] = augment_minority_group(st.session_state["df"], 
                                                    target_column=default_target_column, 
                                                    sensitive_column=sensitive_synt, 
                                                    group_value = group_synt,
                                                    N = number_synt)
            st.success(f"Generated {number_synt} syntetic  data for  {sensitive_synt} -> {group_synt} applied!")

        st.subheader("Bias & Fairness Preprocessing")
        if st.button("Bliding"):
            removed_att = ", ".join([item for item in st.session_state["df_original"].columns.to_list() if item not in include_columns])
            st.session_state["columns_train"] = include_columns
            st.session_state["changes"].append(f"Blinding attributes:{removed_att}")
            st.success(f"Blinded attributes:{removed_att} applied!")

        if st.button("Massaging"):
            st.session_state["df_previus"] = st.session_state["df"]
            st.session_state["changes"].append(f"Changed {number_change} targets from {sensitive_change} with value {group_change}")
            with st.spinner(f"Changing {number_change} targets from {sensitive_change} with value {group_change}..."):
                st.session_state["df"] = change_labels(st.session_state["df"],target_column=default_target_column, 
                sensitive_column=sensitive_change, 
                sensitive_group_to_replace = group_change,
                N = number_change)
            st.success(f"Changed {number_change} targets from {sensitive_change} with value {group_change}")

        if st.button("reweigh"):
            st.session_state["df_previus"] = st.session_state["df"]
            st.session_state["changes"].append(f"reweigh {protected_attribute_name_reweigh} with privileged group {privileged_classes_reweigh}")
            with st.spinner(f"reweighing {protected_attribute_name_reweigh} with privileged group {privileged_classes_reweigh}..."):
                st.session_state["df"], st.session_state["instance_weights"] = reweigh(st.session_state["df"],
                target = default_target_column,
                favorable_classes = favorable_classes_target,
                protected_attribute_name  = protected_attribute_name_reweigh, 
                privileged_classes = privileged_classes_reweigh)
            st.success(f" reweighed {protected_attribute_name_reweigh} with privileged group {privileged_classes_reweigh}")

        if st.button("LFR"):
            st.session_state["df_previus"] = st.session_state["df"]
            st.session_state["changes"].append(f"LFR {protected_attribute_name_lfr} with privileged group {privileged_classes_lfr}")
            with st.spinner(f"LFR {protected_attribute_name_lfr} with privileged group {privileged_classes_lfr}..."):
                st.session_state["df"], df_legivel, encoders = Learning_fair_representations(st.session_state["df"],
                target = default_target_column,
                favorable_classes = favorable_classes_target,
                protected_attribute_name  = protected_attribute_name_lfr, 
                privileged_classes = privileged_classes_lfr,
                drop_columns =  st.session_state["sensitive_columns"])
            st.success(f" LFR {protected_attribute_name_lfr} with privileged group {privileged_classes_lfr}")

        dir = """if st.button("DisparateImpactRemover"):
            st.session_state["df_previus"] = st.session_state["df"]
            st.session_state["changes"].append(f"Removing disparate impact from {protected_attribute_name_dir} with privileged group {privileged_classes_dir} using a repair level of {repair_level_dir}")
            with st.spinner(f"Removing disparate impact from {protected_attribute_name_dir} with privileged group {privileged_classes_dir} using a repair level of {repair_level_dir}..."):
                st.session_state["df"] = Dir(df = st.session_state["df"],
                target = default_target_column,
                
                protected_attribute  = protected_attribute_name_dir, 
                
                repair_level = repair_level_dir,
                drop_columns = st.session_state["sensitive_columns"])
            st.success(f" Removed disparate impact from {protected_attribute_name_dir} with privileged group {privileged_classes_dir} using a repair level of {repair_level_dir}")
        """
        st.subheader("Modifications applied to the current dataset")
        st.session_state["final_report"]["preprocessing"] = st.session_state["changes"]
        for change in range(len(st.session_state["changes"])):
            st.markdown(f"{change + 1}º: {st.session_state['changes'][change]}")
        st.button("Continue", key='continue0', on_click = b1)


    else:
        st.subheader("LMissing the dataset")

elif st.session_state["b5"]:
    style_b5()
    st.session_state["b1"] = False
    st.session_state["b2"] = False
    st.session_state["b3"] = False
    st.session_state["b4"] = False
    st.session_state["b5"] = True
    st.session_state["b6"] = False
    st.session_state["inprocessing_eta"], st.session_state["inprocessing_sensitive_attr"], st.session_state["inprocessing_models"] = aux_func.inprocessing_categorys(st.session_state["df"])

    st.write("### Inprocessing Methods")

    if st.button("None"):
        st.session_state["inprocessing_method"] = None
        st.session_state["inprocessing_params"] = None
        st.success("During training it will not be applied any inprocessing method")
        st.session_state["final_report"]["inprocessing"] = "None"

    # ----------------------------------
    st.subheader("Logistic Regression Classifier")
    if st.button("Prejudice Remover"):
        st.session_state["inprocessing_method"] = "prejudice_remover"
        st.session_state["inprocessing_params"] = {"eta": st.session_state["inprocessing_eta"]}
        st.success("Using Prejudice Remover")
        st.session_state["final_report"]["inprocessing"] = "Prejudice Remover"

    if st.button("Meta Fair Classifier"):
        st.session_state["inprocessing_method"] = "meta_fair_classifier"
        st.session_state["inprocessing_params"] = {"eta": st.session_state["inprocessing_eta"]}
        st.success("Using Meta Fair Classifier")
        st.session_state["final_report"]["inprocessing"] = "Meta Fair Classifier"

    if st.button("Gerry Fair Classifier"):
        st.session_state["inprocessing_method"] = "gerry_fair_classifier"
        st.session_state["inprocessing_params"] = {"eta": st.session_state["inprocessing_eta"]}
        st.success("Using Gerry Fair Classifier")
        st.session_state["final_report"]["inprocessing"] = "Gerry Fair Classifier"


    st.subheader("Neural Net")  
    if st.button("Adversarial Debiasing"):
        st.session_state["inprocessing_method"] = "adversarial_debiasing"
        st.session_state["inprocessing_params"] = {"eta": st.session_state["inprocessing_eta"]}
        st.success("Using Adversarial Debiasing")
        st.session_state["final_report"]["inprocessing"] = "Adversarial Debiasing"

    st.subheader("Choosen Models")
    

    if st.button("Exponentiated Gradient Reduction"):
        st.session_state["inprocessing_method"] = "Exponentiated Gradient Reduction"
        st.session_state["inprocessing_params"] = {"eta": st.session_state["inprocessing_eta"]}
        st.success("Using Exponentiated Gradient Reduction")
        st.session_state["final_report"]["inprocessing"] = "Exponentiated Gradient Reduction"

    if st.button("Grid Search Reduction"):
        st.session_state["inprocessing_method"] = "grid_search_reduction"
        st.session_state["inprocessing_params"] = {"eta": st.session_state["inprocessing_eta"]}
        st.success("Using Grid Search Reduction")
        st.session_state["final_report"]["inprocessing"] = "Grid Search Reduction"


    
    st.button("Continue",  key='continue0',on_click = b2)
       
elif st.session_state["b3"]:
    style_b3()
    st.session_state["b1"] = False
    st.session_state["b2"] = False
    st.session_state["b3"] = True
    st.session_state["b4"] = False
    st.session_state["b5"] = False
    st.session_state["b6"] = False
    
    def sidebar():
        default_models = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "SVM", "MLP (Neural Network)"]
        available_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        available_fairness_metrics = ["equal_opportunity", "predictive_equality", "positive_predictive_parity", "true_positive_rate", "statistical_parity", "disparate_impact"]

        # Models selection inside an expander
        st.sidebar.markdown("""<h2>Models</h2>""", unsafe_allow_html=True)
        with st.sidebar.expander("", expanded=False):
            selected_models = st.multiselect(
                "Select models:", 
                default_models, 
                default=default_models
            )

        # Fairness measures selection inside an expander
        st.sidebar.markdown("""<h2>Fairness Measures</h2>""", unsafe_allow_html=True)
        with st.sidebar.expander("", expanded=False):
            selected_metrics_fairness = st.multiselect(
                "Select fairness metrics:", 
                available_fairness_metrics, 
                default=available_fairness_metrics
            )
        
        return selected_models, selected_metrics_fairness
    st.session_state["selected_models"], st.session_state["selected_metrics_fairness"] = sidebar() 
    st.subheader("Train Models")

    

    if st.button("Train Models and Compare Performance & Fairness"): 

        if st.session_state["atual_final_report"] == {}:
        
            st.session_state["atual_final_report"] = deepcopy(st.session_state["final_report"])
            
            
        else:
            print("\n\n\nAQUII\n\n\n")
            st.session_state["previus_final_report"] = deepcopy(st.session_state["atual_final_report"])
            st.session_state["atual_final_report"] = deepcopy(st.session_state["final_report"])

        if "report_after_performance" not in st.session_state:
            st.session_state["accuracy_atual"] = None
            st.session_state["accuracy_anterior"] = None
            st.session_state["report_after_performance"] = None

            fair_atual = None
            fair_anterior = None
            st.session_state["report_after_fairness"] = None
            
        else:
            
            st.session_state["report_before_performance"] = st.session_state["report_after_performance"]
            st.session_state["accuracy_anterior"] = pd.DataFrame.from_dict(aux_func.extract_metrics(st.session_state["report_after_performance"]), orient="index", columns=["Accuracy"])

            st.session_state["report_before_fairness"] = st.session_state["report_after_fairness"]

        with st.spinner("Training models on corrected data..."):
            
            trainer_after = ModelTrainer(st.session_state["df"], st.session_state["default_target_column"], st.session_state["sensitive_columns"], test_size=0.3, random_state=42, selected_models=st.session_state["selected_models"], train_columns = st.session_state["columns_train"], favorable_classes_target =  st.session_state["favorable_classes_target"], sample_weight = st.session_state["instance_weights"])

            #TODO: COLOCAR ISTO EM PARAMETROS
            trainer_after.fairness_method = st.session_state["inprocessing_method"]
            trainer_after.fairness_params = st.session_state["inprocessing_params"]
            trainer_after.sensitive_attr = st.session_state["inprocessing_sensitive_attr"]
            trainer_after.inprocessing_models = st.session_state["inprocessing_models"]

            st.session_state["current_model"] = trainer_after
            
            st.session_state["report_after_performance"], st.session_state["report_after_fairness"]  = trainer_after.train_and_evaluate(selected_fairness=st.session_state["selected_metrics_fairness"], file_path = model_report_file, fairness_file_path = fairness_report_file)
            st.session_state["accuracy_atual"] = pd.DataFrame.from_dict(aux_func.extract_metrics(st.session_state["report_after_performance"]), orient="index", columns=["Accuracy"])



            trainer_orig = ModelTrainer(st.session_state["df_original"], st.session_state["default_target_column"], st.session_state["sensitive_columns"], test_size=0.3, random_state=42, selected_models=st.session_state["selected_models"], train_columns = st.session_state["df_original"].columns.to_list(), favorable_classes_target = st.session_state["favorable_classes_target"])
            trainer_orig.fairness_method = None
            st.session_state["report_orig_performance"], st.session_state["report_orig_fairness"]  = trainer_orig.train_and_evaluate(selected_fairness=st.session_state["selected_metrics_fairness"], file_path = model_report_file, fairness_file_path = fairness_report_file)

            st.session_state["accuracy_orig"] = pd.DataFrame.from_dict(aux_func.extract_metrics(st.session_state["report_orig_performance"]), orient="index", columns=["Accuracy"])
            

        st.success("Models trained on both datasets! Comparing results...")
    if st.button("Save current Model"): 
        st.session_state["saved_models"]["_".join(st.session_state["changes"])] = st.session_state["current_model"]
    
    st.button("Continue",  key='continue0',on_click = b3)

elif st.session_state["b6"]:
    style_b6()
    st.session_state["b1"] = False
    st.session_state["b2"] = False
    st.session_state["b3"] = False
    st.session_state["b4"] = False
    st.session_state["b5"] = False
    st.session_state["b6"] = True
    
    sensitive_post, priveledge_post = aux_func.postProcessing_caracteristics(st.session_state["df"])
    st.write("### PostProcessing Methods")

    if st.button("None"):
        st.session_state["postprocessing_method"] = None
        
        st.success("It will not be applied a postprocessing method")
        st.session_state["final_report"]["postprocessing"] = "None"

    # ----------------------------------
    #st.subheader("Logistic Regression Classifier")
    if st.button("Equalized Odds"):
        st.session_state["postprocessing_method"] = "Equalized Odds"
        st.success("Using Equalized Odds")
        st.session_state["final_report"]["postprocessing"] = "Equalized Odds"

    if st.button("Calibrated Equalized Odds"):
        st.session_state["postprocessing_method"] = "Calibrated Equalized Odds"
        st.success("Using Calibrated Equalized Odds")
        st.session_state["final_report"]["postprocessing"] = "Calibrated Equalized Odds"

    if st.button("Threshold Optimizer"):
        st.session_state["postprocessing_method"] = "Threshold Optimizer"
        st.success("Using Threshold Optimizer")
        st.session_state["final_report"]["postprocessing"] = "Threshold Optimizer"

    if st.button("Reject Option Classification"):
        st.session_state["postprocessing_method"] = "Reject Option Classification"
        st.success("Using Reject Option Classification")
        st.session_state["final_report"]["postprocessing"] = "Reject Option Classification"

    if st.session_state["postprocessing_method"] != None:
        dicio_all_fair = {}
        report_lines = ["===== Model Training & Evaluation Report ====="]
        for model in st.session_state["selected_models"]:
            if model == "SVM" and st.session_state["postprocessing_method"] == "Threshold Optimizer":
                continue
            new_pred = postProcessing(st.session_state["postprocessing_method"], st.session_state["current_model"].predictions[model], st.session_state["current_model"].df_test, st.session_state["current_model"].models[model], sensitive_post, priveledge_post, st.session_state["default_target_column"])
            
            dicio_all_fair[model] = experiment_fairness(
                        predictions=new_pred,
                        name=model,
                        sensitive_columns=st.session_state["sensitive_columns"],
                        target=st.session_state["default_target_column"],
                        positive_target=st.session_state["favorable_classes_target"],
                        selected_fairness=available_fairness_metrics,
                        test_dataset=st.session_state["current_model"].df_test
                    )
            print(model)
            print(st.session_state["current_model"].y_test)
            print(new_pred)
            accuracy = accuracy_score(st.session_state["current_model"].y_test, new_pred)

            classification_rep = classification_report(st.session_state["current_model"].y_test, new_pred)
            result = f"\n{model} - Accuracy: {accuracy:.4f}\n{classification_rep}"
            report_lines.append(result)
        st.session_state["report_after_fairness"] = dicio_all_fair

        st.session_state["report_after_performance"] = "\n".join(report_lines)
        st.session_state["accuracy_atual"] = pd.DataFrame.from_dict(aux_func.extract_metrics(st.session_state["report_after_performance"]), orient="index", columns=["Accuracy"])
        print(st.session_state["accuracy_atual"])
    st.button("Continue",  key='continue0',on_click = b4)


elif st.session_state["b4"]:
    style_b4()
    st.session_state["b1"] = False
    st.session_state["b2"] = False
    st.session_state["b3"] = False
    st.session_state["b4"] = True
    st.session_state["b5"] = False
    st.session_state["b6"] = False
    if  st.session_state["automatic"]:
        st.subheader("\n\n\n Accuracy")
        x_labels = []
        performance_values = []
        for key, value in st.session_state["automatic_results"].items():
            perfom_aut_write, fair_aut = value  # unpack
            x_labels.append(key)
            
            # Supondo que perfom_aut_write seja um DF com uma coluna chamada 'accuracy'
            # Você pode ajustar isso para sua métrica (ex: f1, precision...)
            performance = perfom_aut_write["Accuracy"].mean()  # ou algum outro agregador
            performance_values.append(performance)

        # Plotando
        fig = plt.figure(figsize=(10, 6))
        plt.bar(x_labels, performance_values, color='skyblue')
        plt.xlabel("Scenario")
        plt.ylabel("Performance (accuracy)")
        plt.title("Model Performance by Scenario")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
        selected_measures = st.session_state["selected_metrics_fairness"]
        num_measures = len(selected_measures)
        cols = 3
        rows = math.ceil(num_measures / cols)
        sensitive_attribute, protected_group, show_fairness = aux_func.automatic_fairness(st.session_state["df"])
        
        if show_fairness:

            max_fairness_value = compute_max_fairness_value(
                st.session_state["automatic_results"],
                selected_measures,
                column=sensitive_attribute,
                protected_group=protected_group
            )
            st.markdown('<div style="margin-bottom: 50px;">', unsafe_allow_html=True)
            st.subheader("\n\n\n Fairness")
            fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
            axs = axs.flatten()

            for i, fairness_measure in enumerate(selected_measures):
                ax = axs[i]

                aux_func.show_fairness_plot_single_model(
                    st.session_state["automatic_results"],
                    column=sensitive_attribute,
                    protected_group=protected_group,
                    selected_fairness=fairness_measure,
                    ax=ax,
                    max_y=max_fairness_value
                )
                ax.set_title(fairness_measure.replace('_', ' ').title())

            # Clean empty axes if needed
            for j in range(len(selected_measures), len(axs)):
                fig.delaxes(axs[j])

            plt.tight_layout()
            st.pyplot(fig)




            
    else:
        if st.session_state["previus_final_report"] == {}:
            st.subheader("Current:")
            aux_func.display_final_report(st.session_state["atual_final_report"])
        else:
            st.subheader("Previus:")
            aux_func.display_final_report(st.session_state["previus_final_report"])
            st.subheader("Current:")
            aux_func.display_final_report(st.session_state["atual_final_report"])
        sensitive_attribute, protected_group, show_fairness, sensitive_attribute_scatter, protected_group_scatter, selected_metric_1_scatter, selected_metric_2_scatter, show_scatter = aux_func.ola (1)
        
        #------------------------------------------------Graficos
        if "report_after_performance"  in st.session_state:
            st.subheader("\n\n\n Accuracy")
            st.pyplot(aux_func.show_plots(len(st.session_state["selected_models"]),st.session_state["accuracy_anterior"], st.session_state["accuracy_atual"],st.session_state["accuracy_orig"] , "Accuracy" ))
        #-----------------------------------------------------Fairness

            
            # Plotando o gráfico
            if show_fairness:
                st.subheader("\n\n\n Fairness")
                selected_measures = st.session_state["selected_metrics_fairness"]
                num_measures = len(selected_measures)
                cols = 3
                rows = math.ceil(num_measures / cols)


                # Coleta os valores máximos para definir a escala
                max_fairness_value = 0

                for fairness_measure in selected_measures:
                    if fairness_measure == "disparate_impact":
                        continue  # Pula disparate impact

                    for model in st.session_state["report_after_fairness"]:
                        if "report_before_fairness" in st.session_state:
                            for report in [st.session_state["report_orig_fairness"],
                                        st.session_state["report_before_fairness"],
                                        st.session_state["report_after_fairness"]]:
                                if model in report and sensitive_attribute in report[model]:
                                    value = report[model][sensitive_attribute].get(
                                        (protected_group, "not_" + protected_group), {}).get(fairness_measure, None)
                                    if value is not None:
                                        max_fairness_value = max(max_fairness_value, abs(value))
                        else:
                            for report in [st.session_state["report_orig_fairness"],
                                        st.session_state["report_after_fairness"]]:
                                if model in report and sensitive_attribute in report[model]:
                                    value = report[model][sensitive_attribute].get(
                                        (str(protected_group), "not_" + str(protected_group)), {}).get(fairness_measure, None)
                                    if value is not None:
                                        max_fairness_value = max(max_fairness_value, abs(value))

                fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
                axs = axs.flatten()

                for i, fairness_measure in enumerate(selected_measures):
                    ax = axs[i]
                    
                    aux_func.show_plots_fairness(
                        st.session_state.get("report_before_fairness", {}),
                        st.session_state.get("report_after_fairness", {}),
                        st.session_state.get("report_orig_fairness", {}),
                        sensitive_attribute,
                        protected_group,
                        
                        fairness_measure,
                        ax=ax,
                        max_y=max_fairness_value  # <-- novo parâmetro
                    )
                    ax.set_title(fairness_measure.replace('_', ' ').title())

                # Apagar os eixos vazios, se houver
                for j in range(len(selected_measures), len(axs)):
                    fig.delaxes(axs[j])

                plt.tight_layout()
                st.pyplot(fig)
            
            if show_scatter:
                st.markdown('<div style="margin-bottom: 50px;">', unsafe_allow_html=True)
                st.subheader("\n\n\n Fairness Scatter")
                fig = aux_func.show_comparison_scatter_plot(
                st.session_state.get("accuracy_anterior", {}),
                st.session_state.get("accuracy_atual", {}),
                st.session_state.get("report_before_fairness", {}),
                st.session_state.get("report_after_fairness", {}),
                sensitive_attribute_scatter,
                protected_group_scatter,
                selected_metric_1_scatter,
                selected_metric_2_scatter
                )
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------Show models saved

        #if st.button("Report"):
            

        
        if st.session_state["saved_models"] != None:
            if st.markdown('<div class="custom-history-button">', unsafe_allow_html=True):
                        if st.button("History"):
                            Nmodelo = 1
                            for nome, versao in st.session_state["saved_models"].items():
                                st.markdown(f"### 🔹 {Nmodelo}º Modelo")
                                Nmodelo += 1
                                partes = nome.split('_')
                                if nome == "":
                                    st.markdown("Original")
                                else:
                                    for i, parte in enumerate(partes, start=1):
                                        st.markdown(f" {i}º {parte}")
                        st.markdown('</div>', unsafe_allow_html=True)

    

    # Example model_trainer usage
    # trainer = ModelTrainer(...)
    # final_report, fairness_dict = trainer.train_and_evaluate(...)
    # df_test = trainer.df_test

    # Simulate output
    # Replace these with actual results
    # model = trainer.models["Random Forest"]
    # df_fairness = pd.read_csv("fairness.csv", sep=";")

    def sidebar_model_download():
        st.sidebar.markdown("""<h2>Model Download</h2>""", unsafe_allow_html=True)
        with st.sidebar.expander(" ", expanded=False):
            model_name = st.selectbox("Model Name:", st.session_state.current_model.models.keys())
        return model_name

    


    def download_dataframe(df, filename="data.csv"):
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, sep=";")
        st.download_button(
            label="Download DataFrame as CSV",
            data=buffer.getvalue(),
            file_name=filename,
            mime="text/csv",
            key="csvdownload"
        )

    
    def download_model(model, filename="model.pkl"):
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        st.download_button(
            label="Download Trained Model",
            data=buffer,
            file_name=filename,
            mime="application/octet-stream",
            key="modeldownload"
        )
    # UI Usage
    
    st.markdown("""
    <style>
    .st-key-csvdownload button:active {
        background-color: #325D79;
        color: white;
        border-color:#325D79;
    }
    .st-key-modeldownload button:active {
        background-color: #325D79;
        color: white;
        border-color:#325D79;
    }
    </style>
    """, unsafe_allow_html=True)

    

    st.subheader("Dataset Downloads")
    download_dataframe(st.session_state.df, filename="data.csv")
    if "current_model" in st.session_state:
        model_name = sidebar_model_download()
        model = st.session_state.current_model.models[model_name]


        st.subheader(f"Model: {model_name}")
        download_model(model, filename=f"{model_name.replace(' ', '_').lower()}.pkl")
    else:
        st.warning("Train a model first to enable downloads.")


st.markdown('</div>', unsafe_allow_html=True)
















#--------------------------------------last one
#st.title("📊 Comparação de Performance & Fairness Antes e Depois")

#-----------------------------------------------------
a =''' st.session_state structure
df -> dataset atualizado (com as mudanças selecionadas)
df_previus -> dataset com todas as mudanças excepto a ultima
df_original -> dataset original
report_before_performance -> performance  no dataset anterior
report_after_performance -> performance  no dataset atual
report_before_fairness -> fairness  no dataset anterior
report_after_fairness -> fairness  no dataset atual
accuracy_atual 
accuracy_anterior
changes -> lista de string com as alteraçoes feita no modelo
saved_models -> dicionario em que key é string com alteraçoes que contem e valor é a class do modelo  ( o modelo esta em self.models["tipo modelo"])
current_model -> class do modelo
columns_train -> colunas que vao ser usadas no treino ( no teste sao todas usadas)
b1, b2,b3,b4 -> butoes de sessao

'''
a="""with st.sidebar.expander("📂 Upload Your Dataset", expanded=True):
    print("sdff")"""

d= """
# Adicionar um botão para carregar o dataset
if uploaded_file and st.button("🔄 Load Dataset"):
    with st.spinner("📂 Loading dataset..."):
        print("inicio")
        df = DataReader.read_data(uploaded_file)
        
        # Salvar o dataset no session_state
        st.session_state["df"] = df
        #st.session_state["df_previus"] = None  # Resetar histórico de edições
        st.session_state["df_original"] = df.copy()  # Manter a versão original
        st.session_state["saved_models"] = {}
        st.session_state["changes"] = []
        st.session_state["current_model"] = None
        st.session_state["columns_train"] = df.columns.to_list()

    st.success("✅ Dataset uploaded successfully!")
    st.write("### 📌 Dataset Preview:")
    st.write(df.head())


"""

#-----------------------------------------------------
# Configuração de correção de dados
c= """if "df" in st.session_state:
    
#---------------------------------Especificações do utilizador
#   Valores selecionados pelo user
    default_target_column, default_sensitive_columns,sensitive_columns, selected_models, selected_metrics_fairness, numeric_strategy, custom_value, categorical_strategy, custom_value_cat, use_knn, use_iterative, use_rf, resampling_method, clusters, sensitive_synt, group_synt, number_synt, include_columns, sensitive_change, group_change, number_change = aux_func.display_categorys(st.session_state["df"])

    st.subheader("📉 Missing Values Analysis")
    if st.button("Check Missing Values"):
        with st.spinner("🔎 Checking for missing values..."):
            missing_data = DataAnalyzer.check_missing_values(st.session_state["df"], file_path=missing_values_report_file)
        st.text(missing_data)
        st.success("✅ Missing values report saved!")

    # Se o botão 'Impute' for pressionado, aplicar imputação
    if st.button("Impute Missing Values"):
        
        with st.spinner("🔧 Imputing missing values..."):
            # Save previous dataset
            if "df" in st.session_state:
                st.session_state["df_previous"] = st.session_state["df"]
            st.session_state["df"], method= impute_missing_values(
                st.session_state["df"],  # O dataframe com os valores ausentes
                numeric_strategy=numeric_strategy,
                categorical_strategy=categorical_strategy,
                custom_value=custom_value,
                use_knn=use_knn,
                use_iterative=use_iterative,
                use_rf=use_rf
            )
            st.session_state["changes"].append("Imputed missing values with: " + method)
        st.success("✅ Missing values imputed!")
        st.subheader("Imputed DataFrame")
        #st.write(st.session_state["df"])

    st.subheader("⚖️ Class Imbalance Analysis")
    if st.button("Analyze Imbalance"):
        with st.spinner("📊 Running class imbalance analysis..."):
            report = DataAnalyzer.analyze_multiple_targets(st.session_state["df"], sensitive_columns, threshold=0.05, file_path=imbalance_report_file)
            aux_func.display_class_distribution(report)
        st.success("✅ Imbalance report saved!")
    
    if st.button("Resample"):
        st.session_state["df_previus"] = st.session_state["df"]
        st.session_state["changes"].append(f"Remsampled with {resampling_method}")
        with st.spinner(f"Applying {resampling_method}..."):
            st.session_state["df"] = oversampling(st.session_state["df"], 
                                                target_column=default_target_column, 
                                                sensitive_columns=sensitive_columns, 
                                                method=resampling_method, clusters=clusters)
        st.success(f"✅ {resampling_method} applied!")



        #present changes in the current dataset 
    
    
    if st.button("Generate syntetic data"):
        st.session_state["df_previus"] = st.session_state["df"]
        st.session_state["changes"].append(f"Generated {number_synt} syntetic data for  {sensitive_synt} -> {group_synt}")
        with st.spinner(f"Generating syntetic data for  {sensitive_synt} -> {group_synt}..."):
            st.session_state["df"] = augment_minority_group(st.session_state["df"], 
                                                target_column=default_target_column, 
                                                sensitive_column=sensitive_synt, 
                                                group_value = group_synt,
                                                N = number_synt)
        st.success(f"✅ Generated {number_synt} syntetic  data for  {sensitive_synt} -> {group_synt} applied!")
    
    if st.button("Bliding"):
        removed_att = ", ".join([item for item in st.session_state["df_original"].columns.to_list() if item not in include_columns])
        st.session_state["columns_train"] = include_columns
        st.session_state["changes"].append(f"Blinding attributes:{removed_att}")

        st.success(f"✅ Blinded attributes:{removed_att} applied!")


    if st.button("Change Lables"):
        st.session_state["df_previus"] = st.session_state["df"]
        st.session_state["changes"].append(f"Changed {number_change} targets from {sensitive_change} with value {group_change}")
        with st.spinner(f"Changing {number_change} targets from {sensitive_change} with value {group_change}..."):
            st.session_state["df"] = change_labels(st.session_state["df"],target_column=default_target_column, 
            sensitive_column=sensitive_change, 
            sensitive_group_to_replace = group_change,
            N = number_change)
        st.success(f"✅ Changed {number_change} targets from {sensitive_change} with value {group_change}")

    
    st.subheader("Modifications applied to the current dataset")

    for change in range(len(st.session_state["changes"])):
        st.markdown(f"{change + 1}º: {st.session_state['changes'][change]}")"""






#----------------------------------Treinar Modelo
    
a= """    st.subheader("🚀 Train Models")


    if st.button("Train Models and Compare Performance & Fairness"): 
        if "report_after_performance" not in st.session_state:
            st.session_state["accuracy_atual"] = None
            st.session_state["accuracy_anterior"] = None
            st.session_state["report_after_performance"] = None

            fair_atual = None
            fair_anterior = None
            st.session_state["report_after_fairness"] = None
            
        else:     
            st.session_state["report_before_performance"] = st.session_state["report_after_performance"]
            st.session_state["accuracy_anterior"] = pd.DataFrame.from_dict(aux_func.extract_metrics(st.session_state["report_after_performance"]), orient="index", columns=["Accuracy"])

            st.session_state["report_before_fairness"] = st.session_state["report_after_fairness"]

            #print("\n\n\n\n\n-----------------Aqui-------------")
            #print(st.session_state["report_before_fairness"]["XGBoost"]["sex"][(" Male", " Female")]["Statistical Parity"])
            #print("\n\n\n\n\n")

        with st.spinner("🚀 Training models on corrected data..."):
            
            trainer_after = ModelTrainer(st.session_state["df"], default_target_column, sensitive_columns, test_size=0.2, random_state=42, selected_models=selected_models, train_columns = st.session_state["columns_train"])
            st.session_state["current_model"] = trainer_after
            
            st.session_state["report_after_performance"], st.session_state["report_after_fairness"]  = trainer_after.train_and_evaluate(selected_fairness=selected_metrics_fairness, file_path = model_report_file, fairness_file_path = fairness_report_file)
            st.session_state["accuracy_atual"] = pd.DataFrame.from_dict(aux_func.extract_metrics(st.session_state["report_after_performance"]), orient="index", columns=["Accuracy"])

        st.success("✅ Models trained on both datasets! Comparing results...")
    if st.button("Save current Model"): 
        st.session_state["saved_models"]["_".join(st.session_state["changes"])] = st.session_state["current_model"]"""



#Results
c= """#------------------------------------------------Graficos
    if "report_after_performance"  in st.session_state:
        st.pyplot(aux_func.show_plots(len(selected_models),st.session_state["accuracy_anterior"], st.session_state["accuracy_atual"], "Accuracy" ))



#-----------------------------------------------------Fairness
        # Seleção dinâmica do fairness metric e do grupo protegido
        selected_fairness = st.selectbox("Choose fairness metric:", selected_metrics_fairness)
        sensitive_attribute = st.selectbox("Choose sensitive attribute:", sensitive_columns )
        protected_group = st.selectbox("Protected Group:", st.session_state["df"][sensitive_attribute].unique().tolist())
        unprotected_group = st.selectbox("Unprotected Group:", [val for val in st.session_state["df"][sensitive_attribute].unique().tolist() if val != protected_group])
        
        # Plotando o gráfico
        if st.button("Show Fairness Plot"):
            fig = aux_func.show_plots_fairness(
                st.session_state.get("report_before_fairness", {}),
                st.session_state.get("report_after_fairness", {}),
                sensitive_attribute,  # Coluna fixa como exemplo, pode ser alterada dinamicamente
                protected_group,
                unprotected_group,
                selected_fairness
            )
            st.pyplot(fig)
        

        aux_func.select_metrics_and_plot(selected_metrics_fairness, sensitive_columns)
    # ----------------------------------------Show models saved

    with st.expander("📂 Show saved models"):
        st.write("# 📋 Models List")
        if st.session_state["saved_models"] != None:
            Nmodelo = 1
            for nome, versao in st.session_state["saved_models"].items():
                st.markdown(f"### 🔹 {Nmodelo}º Modelo")
                Nmodelo += 1
                partes = nome.split('_')
                if nome == "":
                    st.markdown("Original")
                else:
                    for i, parte in enumerate(partes, start=1):
                        st.markdown(f" {i}º {parte}")"""
