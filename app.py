# app.py

import streamlit as st
from streamlit_option_menu import option_menu
# Importa a função específica do seu outro arquivo
from analise import run_analise_page
from previsao import run_previsao_page 

#st.set_page_config(page_title="Sistema de Predição de Obesidade", layout="wide")

st.set_page_config(
    page_title='Sistema de Predição de Obesidade',
    page_icon='heart-pulse',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'About': "Esse app foi desenvolvido por."
    }
)

# Ctrl + K + C   → Comentar
# Ctrl + K + U   → Descomentar
# # Dicionário que mapeia o nome de exibição no selectbox para a função a ser chamada
# PAGES = {
#     "Previsão de Obesidade": run_previsao_page,  # None representa a função padrão (home)
#     "Insights e Métricas": run_analise_page
# }

# # --- Lógica da Barra Lateral (Sidebar) ---
# st.sidebar.title('Navegação')
# selection = st.sidebar.radio("Escolha uma página", list(PAGES.keys()))

# # --- Lógica de Exibição Principal ---

# # Verifica qual função deve ser executada com base na seleção
# if selection == "Previsão de Obesidade":
#     # Conteúdo da página inicial
#     st.title("🏥 Preditor de Nível de Obesidade")
#     #st.write("Use o selectbox na barra lateral para navegar manualmente.")
#     run_previsao_page()

# elif selection == "Insights e Métricas":
#     # Executa a função importada de analise.py
#     run_analise_page()


def main():
    with st.sidebar:
        #Configurando o Menu Principal
            selected = option_menu(
            menu_title = "Menu Principal",
            options=['Previsão de Obesidade', 'Insights e Métricas'],
            icons=["clipboard-data","bar-chart-line"],
            menu_icon="file-earmark-bar-graph-fill",
            default_index=0,
            orientation="vertical",
            styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "options": {"color": "black", "font-size": "25px"},
            "icon": {"color": "#2D314BFB", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "font-weight": "bold",
                "color": "#5a5f63",
                "text-align": "left",
                "margin": "10px",
                "--hover-color": "#d9d2e9"
            },
            "nav-link.active": {
                "background-color": "gray"
            }
            })
     

    if selected =='Previsão de Obesidade':
        run_previsao_page()
    if selected == 'Insights e Métricas':
        run_analise_page()

main()