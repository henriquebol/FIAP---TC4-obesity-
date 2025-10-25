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

# --- 2. Codificar variáveis categóricas ---
label_encoders = {}
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- 3. Separar features e target ---
X = df.drop('Obesity', axis=1)
y = df['Obesity']

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
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders['Obesity'].classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title(f"Matriz de Confusão - {best_model_name}")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png")

# --- 12. Gráfico de acurácia dos modelos ---
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Acurácia Média', y='Modelo', palette='viridis')
plt.title("Comparação de Acurácia entre Modelos")
plt.xlabel("Acurácia Média (5-Fold CV)")
plt.ylabel("Modelo")
plt.tight_layout()
plt.savefig("data/model_comparison.png")

# --- 13. Correlação ---
corr = pd.DataFrame(X, columns=df.columns[:-1]).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlação entre variáveis")
plt.tight_layout()
plt.savefig("data/correlation_heatmap.png")

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
