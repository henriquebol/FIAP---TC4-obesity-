# app.py

import streamlit as st
# Importa a função específica do seu outro arquivo
from analise import run_analise_page
from previsao import run_previsao_page 

st.set_page_config(page_title="Sistema de Predição de Obesidade", layout="wide")

# Dicionário que mapeia o nome de exibição no selectbox para a função a ser chamada
PAGES = {
    "Previsão de Obesidade": run_previsao_page,  # None representa a função padrão (home)
    "Insights e Métricas": run_analise_page
}

# --- Lógica da Barra Lateral (Sidebar) ---
st.sidebar.title('Navegação')
selection = st.sidebar.radio("Escolha uma página", list(PAGES.keys()))

# --- Lógica de Exibição Principal ---

# Verifica qual função deve ser executada com base na seleção
if selection == "Previsão de Obesidade":
    # Conteúdo da página inicial
    st.title("🏥 Preditor de Nível de Obesidade")
    #st.write("Use o selectbox na barra lateral para navegar manualmente.")
    run_previsao_page()

elif selection == "Insights e Métricas":
    # Executa a função importada de analise.py
    run_analise_page()

