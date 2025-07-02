import streamlit as st
def main_frontend():
    st.markdown("""
    <style>
    
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
        h1, h2, h3, h4, h5, h6, span {
            color: #325D79 !important;
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
    .st-d2 {
        background-color: #325D79;
    }
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
    .st-d2 {
        background-color: #325D79 !important;
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
    .st-d3 {
        background-color: #325D79;
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

    .st-emotion-cache-xxjg8 {
        width: 217.594px;
        position: relative;
        display: flex;
        flex: 1 1 0%;
        flex-direction: column;
        gap: 0rem;
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
    </style>
    """, unsafe_allow_html=True)

