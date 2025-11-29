import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

def run_analise_page():
    # --- Página 2: Insights e Métricas ---
    #if page == "Insights e Métricas":
    st.set_page_config(layout="wide")
    st.title("📊 Insights e Análise de métricas")

    tab1, tab2, tab3, tab4 = st.tabs(["📈**Análise de Correlação**", "🎯**Análise de Boxplots**", "📉**Análise de Distribuição**", "**CONCLUSÃO**"],)
    
    with tab1:
        st.markdown("#### 🔹 Correlação entre as características dos entrevistados (Mapa de Calor)")
        try:
            img_comp = Image.open("graphs/matriz_correlacao.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"
            with st.expander("Análise", expanded=True):
                st.markdown('''
                    ### Análise da Relação entre Variáveis e Obesidade

                    Com base na análise do mapa de calor de correlação e das visualizações (box plots e count plots), podemos extrair os seguintes insights e conclusões sobre os fatores que influenciam o grau de obesidade:

                    *   **Correlações Principais:**
                        *   **FAVC** (Consumo frequente de alimentos hipercalóricos) apresentou a maior correlação absoluta negativa com **Obesidade**, indicando que pessoas que não consomem frequentemente alimentos hipercalóricos tendem a ter menor obesidade ou peso normal. No entanto, o **LabelEncoder** pode ter invertido o sentido original da correlação, então é importante revisar a interpretação dos valores codificados. Se 'não' foi codificado como 0 e 'sim' como 1, então uma correlação negativa significaria que 'não' (0) está associado a níveis mais altos de obesidade, o que seria contraintuitivo. Reavaliar a codificação é crucial aqui.
                        *   **Idade** mostrou uma correlação positiva, sugerindo que, em geral, indivíduos mais velhos tendem a apresentar maiores níveis de obesidade.
                        *   **NCP** (Número de refeições principais) e **FCVC** (Frequência de consumo de vegetais) apresentaram correlações negativas. Para **NCP**, menos refeições principais podem estar associadas a níveis mais altos de obesidade. Para **FCVC**, um menor consumo de vegetais está associado a maiores níveis de obesidade.
                        *   **CALC** (Consumo de álcool) também mostrou uma correlação negativa, o que, dependendo da codificação, pode indicar que um maior consumo de álcool está associado a menores níveis de obesidade, ou vice-versa. A codificação 'nao', 'As_vezes', 'Frequentemente', 'Sempre' (0,1,2,3) foi feita em ordem crescente, então uma correlação negativa sugere que quanto maior o consumo de álcool, menor o nível de obesidade, o que também pode ser um ponto para reflexão sobre a complexidade da relação.
                                ''')
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
##########################################################################################################################
    with tab2:
        st.markdown("#### 🔹 Análise dos Boxplots")
        try:   
            img_comp = Image.open(f"graphs/box_plot_Peso(kg).png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"    
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        try:   
            img_comp = Image.open(f"graphs/box_plot_Idade.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"    
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        with st.expander("Análise", expanded=True):
                st.markdown('''
                **Peso(kg) e Idade:** Claramente, o **Peso(kg)** aumenta progressivamente com o nível de obesidade. 
            A **Idade** também demonstra uma tendência de aumento da mediana à medida que o nível de obesidade se agrava, 
                            especialmente em 'Obesidade_tipo_II' e 'Obesidade_tipo_III'.
                        ''')
        #************************************************************************************************#
        st.markdown('---')
        try:   
            img_comp = Image.open(f"graphs/box_plot_Altura(m).png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"    
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        with st.expander("Análise", expanded=True):
                st.markdown('''
                **Altura(m):** Não parece haver uma relação linear forte ou um padrão claro entre
                             **Altura(m)** e os níveis de obesidade.
                        ''')
        #************************************************************************************************#
        st.markdown('---')
        try:   
            img_comp = Image.open(f"graphs/box_plot_FCVC.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"    
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        try:   
            img_comp = Image.open(f"graphs/box_plot_NCP.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"    
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        with st.expander("Análise", expanded=True):
                st.markdown('''
                        **FCVC (Frequência de consumo de vegetais) e NCP (Número de refeições principais):** As medianas de **FCVC** e **NCP** tendem a diminuir ou 
                            permanecer estáveis em níveis mais altos de obesidade, o que reforça 
                            a ideia de que menor consumo de vegetais e menos refeições principais 
                            podem estar associados à obesidade.
                            ''')
        #************************************************************************************************#
        st.markdown('---')
        try:   
            img_comp = Image.open(f"graphs/box_plot_FAF.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"    
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        try:   
            img_comp = Image.open(f"graphs/box_plot_TUE.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"    
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        with st.expander("Análise", expanded=True):
                st.markdown('''
                        **FAF (Frequência de atividade física) e TUE (Tempo de uso de dispositivos tecnológicos):** 
                        **FAF** tende a diminuir e **TUE** tende a aumentar com o agravamento da obesidade, 
                            o que é um insight esperado e reforça a importância da atividade física e a limitação 
                            do tempo de tela para auxiliar na redução dos níveis de obesidade.
                            ''')         
 ################################################################################################################   
    with tab3:
        st.markdown("#### 🔹 Análise dos gráficos de Distribuição")
        try:
            img_comp = Image.open(f"graphs/dist_plot_Historico_familiar_Original.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")

        with st.expander("Análise", expanded=True):
            st.markdown('''
                *   **Historico familiar:** Indivíduos com histórico familiar de obesidade são predominantes 
                        nos níveis mais altos de obesidade (Obesidade_tipo_I, II, III), 
                        sugerindo uma forte influência genética ou de hábitos familiares.''')

    #######################################################################################################    
        st.markdown('---')
        try:
            img_comp = Image.open(f"graphs/dist_plot_Sexo_Original.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        with st.expander("Análise", expanded=True):
            st.markdown('''
                **Sexo:** Observa-se uma distribuição notável nos níveis de obesidade. Por exemplo, 'Obesidade_tipo_III' parece ser predominantemente feminina,
                         enquanto 'Obesidade_tipo_II' é mais encontrada no sexo masculino.''')
    ####################################################################################################### 
        st.markdown('---')
        try:
            img_comp = Image.open(f"graphs/dist_plot_FAVC_Original.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        with st.expander("Análise", expanded=True):
            st.markdown('''
                 **FAVC:** Podemos inferir que a categoria 'sim' (consumo frequente de alimentos hipercalóricos) prevalece em todos níveis de obesidade, 
                        enquanto 'não' é mais comum em 'Peso Normal'. Indicando que as pessoas que não consomem alimentos hipercalóricos
                        com frequência tendem a ter um peso dentro da normalidade.
                        ''')
    #######################################################################################################
        st.markdown('---')
        try:
            img_comp = Image.open(f"graphs/dist_plot_MTRANS_Original.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        with st.expander("Análise", expanded=True):
            st.markdown('''
                 **MTRANS (Meio de transporte):** 'Transporte_publico' é amplamente utilizado em todos os níveis, 
                        mas 'Automovel' e 'A_pe' podem mostrar variações interessantes. Por exemplo, 'Automovel' 
                        pode ser mais comum em grupos com maior obesidade devido à menor atividade física associada. 
                        A categoria 'A_pe' é mais comum entre os grupos de 'Peso_Normal' e 'Abaixo_do_peso', 
                        indicando que andar pode ser um fator protetor.
                        ''')
    #######################################################################################################
        st.markdown('---')
        try:
            img_comp = Image.open(f"graphs/dist_plot_CALC_Original.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        with st.expander("Análise", expanded=True):
            st.markdown('''
                 **CALC (Consumo de álcool):** 
                        Podemos inferir com relação aos dados apresentados no gráfico que o consumo de álcool 
                        tem relação direta com a obesidade. Ou seja, quanto maior o consumo de álcool maior a probabilidade 
                        do indivíduo ser tornar uma pessoa obesa no futuro.
                        ''')
    #######################################################################################################
        st.markdown('---')
        try:
            img_comp = Image.open(f"graphs/dist_plot_CAEC_Original.png")
            st.image(img_comp, use_container_width=True) #caption="Comparação de Acurácia entre os Modelos"
        except:
            st.warning("Imagem de comparação não encontrada. Execute o script de treinamento novamente.")
        
        with st.expander("Análise", expanded=True):
            st.markdown('''
                 **CAEC (Consumo de alimentos entre as refeições):** 
                        Assim como na análise em relação ao consumo de álccol, o consumo de alimentos entre as refeições
                        de forma exporádica pode ser associado à obesidade.
                        ''')
    #######################################################################################################
    with tab4:
        st.markdown('### 💡 Conclusão:')
        st.markdown('''
            Os fatores mais influentes na obesidade, conforme esta análise exploratória, 
            são o **histórico familiar**, **idade**, **peso**, **frequência de atividade física (FAF)**, 
            **tempo de uso de dispositivos tecnológicos (TUE)**, 
            **consumo frequente de alimentos hipercalóricos (FAVC)**, 
            **frequência de consumo de vegetais (FCVC)** e o **meio de transporte (MTRANS)**. 
            Existe uma clara progressão do peso, idade, e hábitos de vida (como sedentarismo 
            e consumo de alimentos pouco saudáveis) com o aumento do nível de obesidade. 
        ''')