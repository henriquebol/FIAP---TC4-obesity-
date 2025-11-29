import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

def run_previsao_page():
    st.set_page_config(layout= 'wide')
    # --- Carregar modelo e artefatos ---
    with open('model/obesity_model.pkl', 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    scaler = data['scaler']
    label_encoders = data['label_encoders']
    columns = data['columns']

    # --- Sidebar de navegação ---
    #st.sidebar.title("Navegação")
    #page = st.sidebar.radio("Ir para:", ["Previsão de Obesidade", "Insights e Métricas"])

    # --- Página 1: Previsão ---
    #if page == "Previsão de Obesidade":
        #st.title("🏥 Preditor de Nível de Obesidade")
    st.markdown("Responda as perguntas abaixo para estimar o nível de obesidade:")

    user_input = {}
    with st.container(border=True):
        col1, col2 = st.columns(2, gap="large")
        # Perguntas categóricas
        with col1:
            user_input["Sexo"] = st.selectbox("Gênero:", ["Masculino", "Feminino"])
            user_input["Idade"] = st.slider("Idade (anos):", 10, 80, 25)
            user_input["Altura(m)"] = st.number_input("Altura (m):", min_value=1.20, max_value=2.10, value=1.70, step=0.01)
        with col2:
            user_input["Peso(kg)"] = st.number_input("Peso (kg):", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            user_input["Historico_familiar"] = st.selectbox("Algum membro da família sofre ou sofreu de obesidade?", ["nao", "sim"])

    st.subheader("Hábitos alimentares")
    with st.container(border=True):
        col1, col2 = st.columns(2, gap="large")
        with col1:    
            user_input["FAVC"] = st.selectbox("Você come alimentos altamente calóricos com frequência?", ["nao", "sim"])
            user_input["FCVC"] = st.slider("Você costuma comer vegetais nas refeições? (1=nunca, 3=sempre)", 1, 3, 2)
            user_input["NCP"] = st.slider("Quantas refeições principais você faz por dia?", 1, 4, 3)
        with col2:    
            user_input["CAEC"] = st.selectbox("Você come algo entre as refeições?", ["nao", "As_vezes", "Frequentemente", "Sempre"])
            user_input["Fumante"] = st.selectbox("Você fuma?", ["nao", "sim"])

    st.subheader("Hábitos diários")
    with st.container(border=True):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            user_input["CH2O"] = st.slider("Quanta água você bebe por dia? (1=pouca, 3=muita)", 1, 3, 2)
            user_input["SCC"] = st.selectbox("Você monitora as calorias que ingere?", ["nao", "sim"])
            user_input["FAF"] = st.slider("Com que frequência pratica atividade física? (0=nunca, 3=frequente)", 0, 3, 2)
        with col2:
            user_input["TUE"] = st.slider("Tempo de uso de dispositivos eletrônicos (0=baixo, 2=alto)", 0, 2, 1)
            user_input["CALC"] = st.selectbox("Com que frequência você bebe álcool?", ["nao", "As_vezes", "Frequentemente", "Sempre"])
            user_input["MTRANS"] = st.selectbox("Meio de transporte principal:", ["A_pe", "Bicicleta", "Transporte_publico", "Motocicleta", "Automovel"])

    # Prever
    if st.button("Classificar"):
        df_input = pd.DataFrame([user_input])

        # --- Mapeamentos manuais com fillna ---
        mtrans_mapping = {'A_pe':0, 'Bicicleta':1, 'Transporte_publico':2, 'Motocicleta':3, 'Automovel':4}
        calc_mapping = {'nao':0, 'As_vezes':1, 'Frequentemente':2, 'Sempre':3}
        caec_mapping = {'nao':0, 'As_vezes':1, 'Frequentemente':2, 'Sempre':3}

        df_input["MTRANS"] = df_input["MTRANS"].map(mtrans_mapping).fillna(-1)
        df_input["CALC"] = df_input["CALC"].map(calc_mapping).fillna(-1)
        df_input["CAEC"] = df_input["CAEC"].map(caec_mapping).fillna(-1)

        # --- LabelEncoder para demais colunas ---
        for col, le in label_encoders.items():
            if col in df_input.columns and col != "Obesidade":
                try:
                    df_input[col] = le.transform(df_input[col])
                except ValueError as e:
                    st.error(f"⚠️ Erro ao transformar coluna {col}: {e}")
                    st.stop()

        # --- Checar colunas faltantes ---
        missing_cols = set(columns) - set(df_input.columns)
        if missing_cols:
            st.error(f"⚠️ Colunas faltando: {missing_cols}")
            st.stop()

        # --- Reordenar para bater com 'columns' ---
        df_input = df_input.reindex(columns=columns)

        # --- Checar NaN antes de escalar ---
        if df_input.isnull().any().any():
            st.error("⚠️ Existem valores ausentes nas entradas. Verifique os campos preenchidos.")
            st.write(df_input)
            st.stop()

        # --- Escalar ---
        df_scaled = scaler.transform(df_input)

        # --- Previsão ---
        pred = model.predict(df_scaled)[0]
        inv_pred = list(label_encoders["Obesidade"].inverse_transform([pred]))[0]

        st.success(f"🏷️ Nível de obesidade previsto: **{inv_pred}**")

