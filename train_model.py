import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# --- 1. Carregar os dados ---
df = pd.read_csv('data/Obesity.csv')
df = df.dropna()


#-------Traduzindo colunas para português---------
df_trad =  df.copy()
df_trad['Gender'] = df_trad['Gender'].replace({
    'Female': 'Feminino',
    'Male': 'Masculino'
})

df_trad['family_history'] = df_trad['family_history'].replace({
    'yes': 'sim',
    'no': 'nao'
})

df_trad['FAVC'] = df_trad['FAVC'].replace({
    'yes': 'sim',
    'no': 'nao'
})

df_trad['SMOKE'] = df_trad['SMOKE'].replace({
    'yes': 'sim',
    'no': 'nao'
})

df_trad['SCC'] = df_trad['SCC'].replace({
    'yes': 'sim',
    'no': 'nao'
})

df_trad['MTRANS'] = df_trad['MTRANS'].replace({
    'Public_Transportation': 'Transporte_publico',
    'Walking': 'A_pe',
    'Automobile': 'Automovel',
    'Motorbike': 'Motocicleta',
    'Bike': 'Bicicleta'
})

df_trad['CALC'] = df_trad['CALC'].replace({
    'Sometimes': 'As_vezes',
    'Frequently': 'Frequentemente',
    'Always': 'Sempre',
    'no': 'nao'
})

df_trad['CAEC'] = df_trad['CAEC'].replace({
    'Sometimes': 'As_vezes',
    'Frequently': 'Frequentemente',
    'Always': 'Sempre',
    'no': 'nao'
})

df_trad['Obesity'] = df_trad['Obesity'].replace({
    'Normal_Weight': 'Peso_Normal',
    'Overweight_Level_I': 'Sobrepeso_nivel_I',
    'Overweight_Level_II': 'Sobrepeso_nivel_II',
    'Obesity_Type_I': 'Obesidade_tipo_I',
    'Insufficient_Weight': 'Abaixo_do_peso',
    'Obesity_Type_II': 'Obesidade_tipo_II',
    'Obesity_Type_III': 'Obesidade_tipo_III'
})

#------Renomeando as colunas traduzidas------
df_trad = df_trad.rename(columns={
    'Gender': 'Sexo', 'Age': 'Idade', 'Height': 'Altura(m)', 'Weight': 'Peso(kg)', 'family_history': 'Historico_familiar', 'SMOKE': 'Fumante', 'Obesity':'Obesidade'})

# -----Codificando manualmente uma parte das variáveis categoricas-----
df_trad['MTRANS'] = df_trad['MTRANS'].replace({
    'A_pe': 0,
    'Bicicleta': 1,
    'Transporte_publico': 2,
    'Motocicleta': 3,
    'Automovel': 4
})

df_trad['CALC'] = df_trad['CALC'].replace({
    'nao': 0,
    'As_vezes': 1,
    'Frequentemente': 2,
    'Sempre': 3
})

df_trad['CAEC'] = df_trad['CAEC'].replace({
    'nao': 0,
    'As_vezes': 1,
    'Frequentemente': 2,
    'Sempre': 3
})

# --- 2. Codificar variáveis categóricas ---
label_encoders = {}
for col in df_trad.select_dtypes(include='object'):
    le = LabelEncoder()
    df_trad[col] = le.fit_transform(df_trad[col])
    label_encoders[col] = le

# --- 3. Separar features e target ---
X = df_trad.drop('Obesidade', axis=1)
y = df_trad['Obesidade']

# --- 4. Normalizar ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. Dividir dados ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

# --- 6. Modelos ---
models = {
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# --- 7. Validação cruzada e avaliação ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("\n=== Resultados de Cross-Validation (5-folds) ===")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    mean_acc = np.mean(scores)
    results.append((name, mean_acc))
    print(f"{name}: {mean_acc:.4f} ± {scores.std():.4f}")

# --- 8. Comparar resultados ---
results_df = pd.DataFrame(results, columns=['Modelo', 'Acurácia Média'])
best_model_name = results_df.iloc[results_df['Acurácia Média'].idxmax()]['Modelo']
best_model = models[best_model_name]

# --- 9. Treinar modelo final ---
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)

# --- 10. Relatório de classificação ---
print("\n=== Relatório de Classificação ===")
print(classification_report(y_test, preds))

# --- 11. Matriz de confusão ---
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders['Obesidade'].classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title(f"Matriz de Confusão - {best_model_name}")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png")

# --- 12. Gráfico de acurácia dos modelos ---
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Acurácia Média', y='Modelo', palette='viridis')
plt.title("Comparação de Acurácia entre Modelos")
plt.xlabel("Acurácia Média (5-Fold CV)")
plt.ylabel("Modelo")
plt.tight_layout()
plt.savefig("graphs/model_comparison.png")

# --- 13. Correlação ---
plt.figure(figsize=(14, 8))
sns.heatmap(df_trad.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Mapa de Calor da Matriz de Correlação')
plt.savefig("graphs/matriz_correlacao.png")

# -- Visualizando Características Numéricas vs. Níveis de Obesidade
df_viz = df_trad.copy()
df_viz['Obesidade_Original'] = label_encoders['Obesidade'].inverse_transform(df_viz['Obesidade'])

numerical_features = ['Idade', 'Altura(m)', 'Peso(kg)', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

for feature in numerical_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Obesidade_Original', y=feature, data=df_viz, palette='viridis')
    plt.title(f'Distribuição de {feature} por Nível de Obesidade')
    plt.xlabel('Nível de Obesidade')
    plt.ylabel(feature)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"graphs/box_plot_{feature}.png")

# -- Visualizar Características Categóricas vs. Níveis de Obesidade
df_trad['MTRANS'] = df_trad['MTRANS'].replace({
    'A_pe': 0,
    'Bicicleta': 1,
    'Transporte_publico': 2,
    'Motocicleta': 3,
    'Automovel': 4
})

df_trad['CALC'] = df_trad['CALC'].replace({
    'nao': 0,
    'As_vezes': 1,
    'Frequentemente': 2,
    'Sempre': 3
})

df_trad['CAEC'] = df_trad['CAEC'].replace({
    'nao': 0,
    'As_vezes': 1,
    'Frequentemente': 2,
    'Sempre': 3
})

df_viz = df_trad.copy()
df_viz['Obesidade_Original'] = label_encoders['Obesidade'].inverse_transform(df_viz['Obesidade'])

# Define os mapeamentos originais para colunas codificadas manualmente.
# MTRANS: 0->A_pe, 1->Bicicleta, 2->Transporte_publico, 3->Motocicleta, 4->Automovel
mtrans_mapping = {0: 'A_pe', 1: 'Bicicleta', 2: 'Transporte_publico', 3: 'Motocicleta', 4: 'Automovel'}
# CALC: 0->nao, 1->As_vezes, 2->Frequentemente, 3->Sempre
calc_mapping = {0: 'nao', 1: 'As_vezes', 2: 'Frequentemente', 3: 'Sempre'}
# CAEC: 0->nao, 1->As_vezes, 2->Frequentemente, 3->Sempre
caec_mapping = {0: 'nao', 1: 'As_vezes', 2: 'Frequentemente', 3: 'Sempre'}

# Cria novas colunas com rótulos de texto originais para as variáveis ​​categóricas para fins de plotagem.
for col_name, encoder in label_encoders.items():
    if col_name != 'Obesidade': # Obesidade_Original já está resolvido
        # Cria uma nova coluna com rótulos de string originais usando o LabelEncoder
        df_viz[f'{col_name}_Original'] = encoder.inverse_transform(df_viz[col_name])

# Mapear de volta as colunas codificadas manualmente para seus rótulos de string originais.
df_viz['MTRANS_Original'] = df_viz['MTRANS'].map(mtrans_mapping)
df_viz['CALC_Original'] = df_viz['CALC'].map(calc_mapping)
df_viz['CAEC_Original'] = df_viz['CAEC'].map(caec_mapping)

categorical_features_original_labels = [
    'Sexo_Original', 'Historico_familiar_Original', 'FAVC_Original',
    'CAEC_Original', 'Fumante_Original', 'SCC_Original',
    'CALC_Original', 'MTRANS_Original'
]

for feature in categorical_features_original_labels:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=feature, hue='Obesidade_Original', data=df_viz, palette='viridis', legend=True)
    plt.title(f'Distribuição de {feature.replace("_Original", "")} por Nível de Obesidade')
    plt.xlabel(feature.replace("_Original", ""))
    plt.ylabel('Contagem')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Nível de Obesidade')
    plt.tight_layout()
    plt.savefig(f"graphs/dist_plot_{feature}.png")

# --- 14. Salvar modelo ---
with open('model/obesity_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'columns': X.columns
    }, f)

print(f"\nMelhor modelo: {best_model_name}")
print("Arquivos salvos em /data e /model com gráficos e modelo treinado.")
