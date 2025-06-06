import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.ticker as ticker

# ====== CONFIGURA TUS DATOS ======
# Cargar tu dataset
df = pd.read_csv("C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\dataset_final.csv")  # Cambia al nombre de tu archivo
target_col = 'escenario'  # Cambia esto al nombre real de tu columna objetivo si es diferente

# Separar características y etiquetas
X = df.drop(columns=[target_col])
y = df[target_col]

# ============================
# 1. Pearson Correlation Coefficient (PCC)
# ============================
pcc_scores = []
for col in X.columns:
    score = np.corrcoef(X[col], y)[0, 1]
    pcc_scores.append(score)

pcc_df = pd.DataFrame({
    'Feature': X.columns,
    'PCC Importance': np.abs(pcc_scores)
}).sort_values(by='PCC Importance', ascending=False)

# ============================
# 2. ExtraTreesClassifier
# ============================
etc = ExtraTreesClassifier(n_estimators=150, random_state=42)
etc.fit(X, y)
etc_df = pd.DataFrame({
    'Feature': X.columns,
    'ETC Importance': etc.feature_importances_
}).sort_values(by='ETC Importance', ascending=False)

# ============================
# 3. RandomForestClassifier
# ============================
rfc = RandomForestClassifier(n_estimators=150, random_state=42)
rfc.fit(X, y)
rfc_df = pd.DataFrame({
    'Feature': X.columns,
    'RFC Importance': rfc.feature_importances_
}).sort_values(by='RFC Importance', ascending=False)

# ============================
# Visualización: Gráficos de barras
# ============================

def plot_importance(df, score_col, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=score_col, y='Feature', palette='viridis')
    plt.title(title)
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.show()

# Ticks personalizados para ETC y RFC
custom_ticks = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
# Añadir una etiqueta ">0.06" en lugar de 0.07 si lo deseas

def plot_importance_with_label(df, score_col, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=score_col, y='Feature', palette='viridis')

    plt.title(title)
    plt.xlabel('Importancia')
    plt.ylabel('Característica')

    # Define límites y ticks
    plt.xlim(0, 0.07)
    plt.xticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Agrega un texto " >0.06" si hay algún valor superior
    max_val = df[score_col].max()
    if max_val > 0.06:
        plt.xticks(list(plt.xticks()[0]) + [0.07])
        ax.set_xticklabels([f'{x:.2f}' if x < 0.07 else '>0.06' for x in plt.xticks()[0]])

    plt.tight_layout()
    plt.show()

# Mostrar resultados
print("=== PCC Importance ===")
print(pcc_df)

print("\n=== ExtraTreesClassifier Importance ===")
print(etc_df)

print("\n=== RandomForestClassifier Importance ===")
print(rfc_df)

# Dibujar gráficos
plot_importance(pcc_df, 'PCC Importance', 'Importancia de características - PCC')
plot_importance_with_label(etc_df, 'ETC Importance', 'Importancia de características - ExtraTreesClassifier')
plot_importance_with_label(rfc_df, 'RFC Importance', 'Importancia de características - RandomForestClassifier')
