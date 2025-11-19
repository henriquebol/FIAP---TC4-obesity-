import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# --- Carregar modelo e artefatos ---
with open('model/obesity_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']
label_encoders = data['label_encoders']
columns = data['columns']

# --- Sidebar de navegação ---
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Previsão de Obesidade", "Insights e Métricas"])

# --- Página 1: Previsão ---
if page == "Previsão de Obesidade":
    st.title("🏥 Preditor de Nível de Obesidade")
    st.markdown("Responda as perguntas abaixo para estimar o nível de obesidade:")

    # Perguntas categóricas
    user_input = {}
    user_input["Gender"] = st.selectbox("Gênero:", ["Male", "Female"])
    user_input["Age"] = st.slider("Idade (anos):", 10, 80, 25)
    user_input["Height"] = st.number_input("Altura (m):", min_value=1.20, max_value=2.10, value=1.70, step=0.01)
    user_input["Weight"] = st.number_input("Peso (kg):", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    user_input["family_history"] = st.selectbox("Algum membro da família sofre ou sofreu de obesidade?", ["no", "yes"])

    st.subheader("Hábitos alimentares")
    user_input["FAVC"] = st.selectbox("Você come alimentos altamente calóricos com frequência?", ["no", "yes"])
    user_input["FCVC"] = st.slider("Você costuma comer vegetais nas refeições? (1=nunca, 3=sempre)", 1, 3, 2)
    user_input["NCP"] = st.slider("Quantas refeições principais você faz por dia?", 1, 4, 3)
    user_input["CAEC"] = st.selectbox("Você come algo entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    user_input["SMOKE"] = st.selectbox("Você fuma?", ["no", "yes"])

    st.subheader("Hábitos diários")
    user_input["CH2O"] = st.slider("Quanta água você bebe por dia? (1=pouca, 3=muita)", 1, 3, 2)
    user_input["SCC"] = st.selectbox("Você monitora as calorias que ingere?", ["no", "yes"])
    user_input["FAF"] = st.slider("Com que frequência pratica atividade física? (0=nunca, 3=frequente)", 0, 3, 2)
    user_input["TUE"] = st.slider("Tempo de uso de dispositivos eletrônicos (0=baixo, 2=alto)", 0, 2, 1)
    user_input["CALC"] = st.selectbox("Com que frequência você bebe álcool?", ["no", "Sometimes", "Frequently", "Always"])
    user_input["MTRANS"] = st.selectbox("Meio de transporte principal:", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

    # Prever
    if st.button("Classificar"):
        df_input = pd.DataFrame([user_input])
        for col, le in label_encoders.items():
            if col in df_input.columns:
                df_input[col] = le.transform(df_input[col])
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)[0]
        inv_pred = list(label_encoders["Obesity"].inverse_transform([pred]))[0]

        st.success(f"🏷️ Nível de obesidade previsto: **{inv_pred}**")

# --- Página 2: Insights e Métricas ---
elif page == "Insights e Métricas":
    st.title("📊 Insights e Desempenho dos Modelos")

    st.markdown("### 🔹 Comparação de Acurácia entre Modelos")
    try:
        img_comp = Image.open("graphs/model_comparison.png")
        st.image(img_comp, caption="Comparação de Acurácia entre os Modelos", use_container_width=True)
    except:
        st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")

    st.markdown("### 🔹 Importância das entradas")

    try:
        img_conf = Image.open("graphs/feature_importance.png")
        st.image(img_conf, caption="Importância das entradas", use_container_width=True)
    except:
        st.warning("Não encontrada. Execute o script de treinamento novamente.")

    st.markdown("### 🔹Matriz de Confusão do Melhor Modelo (Gradient Boosting)")

    try:
        img_conf = Image.open("graphs/confusion_matrix.png")
        st.image(img_conf, caption="Matriz de Confusão do Melhor Modelo", use_container_width=True)
    except:
        st.warning("Matriz de confusão não encontrada. Execute o script de treinamento novamente.")

    st.markdown("### 🔹 Correlação entre Variáveis")
    try:
        img_corr = Image.open("graphs/correlation_heatmap.png")
        st.image(img_corr, caption="Mapa de Correlação entre Variáveis", use_container_width=True)
    except:
        st.warning("Mapa de correlação não encontrado.")

    st.markdown("""
    ### 💡 Insights Principais:
    - **Peso** e **Altura** têm alta correlação inversa.
    - **Atividade física (FAF)** tem correlação negativa com obesidade.
    - **Consumo de alimentos calóricos (FAVC)** aumenta o risco de obesidade.
    - **Tempo em dispositivos eletrônicos (TUE)** tende a elevar o nível de obesidade.
    - **Histórico familiar** influencia diretamente na probabilidade de obesidade.
    """)
