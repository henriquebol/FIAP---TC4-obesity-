# 🧠 Classificação de Obesidade  
Projeto FIAP – Tech Challenge • Machine Learning + Streamlit

Este projeto implementa um modelo de **classificação de níveis de obesidade** com base em dados coletados por questionário e atributos antropométricos.  
Inclui **treinamento do modelo**, **análise exploratória**, **persistência do pipeline** e uma **interface web em Streamlit** para realizar previsões em tempo real.

## 📌 Objetivo  
Desenvolver um sistema capaz de **prever o nível de obesidade** de um indivíduo com base em hábitos, características corporais e comportamento alimentar.

Ele é composto por duas partes:

1. **Treinamento do modelo (train_model.py)**  
2. **Aplicação interativa em Streamlit (app.py)**  

## 📂 Dataset  
O dataset contém atributos relacionados a:

- Idade  
- Peso e altura  
- IMC  
- Número de refeições  
- Tipo de alimentação  
- Consumo calórico  
- Nível de atividade física  
- Histórico familiar  
- Hábitos alimentares  

A classe alvo representa categorias como:  
*Peso abaixo do normal, saudável, sobrepeso, obesidade grau I/II/III.*

## 🔍 Análise Exploratória  
O projeto inclui EDA com gráficos gerados automaticamente, como:

- Distribuição das classes  
- Correlação entre variáveis  
- Boxplots por categoria  
- Relação IMC × Obesidade  

Todos os gráficos são salvos em:

```bash
/graphs
```

## 🧹 Pré-processamento  
O pipeline realiza:

- Limpeza e normalização de dados  
- Codificação de variáveis categóricas  
- Criação de atributos auxiliares (como IMC, caso aplicável)  
- Divisão em treino/teste mantendo distribuição das classes  
- Padronização de colunas numéricas  
- Criação de pipeline completo para inferência

## 🤖 Treinamento do Modelo (train_model.py)

O script testa diversos algoritmos, como:

- Logistic Regression  
- Random Forest  
- KNN  
- SVM  
- XGBoost (se presente no projeto)

São geradas métricas como:

- Acurácia  
- Matriz de confusão  
- Precision / Recall / F1-Score  

O melhor modelo é **salvo automaticamente** em:

```bash
/model/model.pkl
```

Junto com o pipeline de pré-processamento, garantindo que a inferência seja consistente.

## 🖥️ Aplicação Streamlit (app.py)

A interface web permite:

- Inserir informações do indivíduo  
- Visualizar IMC calculado  
- Obter a previsão de nível de obesidade  
- Exibir informações auxiliares  

Para iniciar a aplicação:

```bash
streamlit run app.py
```

Ela abrirá em:

```bash
http://localhost:8501
```

## ▶️ Como Executar o Projeto  

### 1️⃣ Instale as dependências  

```bash
pip install -r requirements.txt
```

### 2️⃣ Execute o treinamento  

```bash
python train_model.py
```

Isso irá:  
- Processar o dataset  
- Treinar os modelos  
- Salvar o melhor pipeline  
- Gerar gráficos exploratórios  

### 3️⃣ Abra a interface  

```bash
streamlit run app.py
```

## 📁 Estrutura do Projeto

```bash
FIAP---TC4-obesity/
│
├── data/ # Dataset original
│ └── obesity.csv
│
├── graphs/ # Gráficos gerados pelo EDA
│ └── *.png
│
├── model/ # Modelo treinado + pipeline de preprocessamento
│ └── model.pkl
│
├── train_model.py # Pipeline de treino e avaliação
├── app.py # Interface Streamlit
├── requirements.txt # Dependências
└── README.md # Este arquivo
```

## 🛠️ Dependências

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- streamlit

## 📜 Licença  
Este projeto é livre para uso acadêmico e estudo.
