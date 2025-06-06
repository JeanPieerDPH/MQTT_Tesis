import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Cargar tu dataset (ajusta el nombre del archivo)
df = pd.read_csv("C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\dataset_final_recortado.csv")

# Definir la columna de la etiqueta
target_col = 'escenario'  # Cambia esto al nombre real de tu columna objetivo

# Separar características y etiquetas
X_original = df.drop(columns=[target_col])
y_original = df[target_col]

# Dividir el dataset en entrenamiento y prueba (80% - 20%) con estratificación
X, X_test, y, y_test = train_test_split(
    X_original, y_original,
    test_size=0.2,
    random_state=42,
    stratify=y_original  # <-- mantiene proporción de clases
)

# Escalar características
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Configurar validación cruzada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ----------------------------
# GLM - Logistic Regression
# ----------------------------
param_grid_glm = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'multi_class': ['multinomial'],
    'max_iter': [500]
}
glm = LogisticRegression()
grid_glm = GridSearchCV(glm, param_grid_glm, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
grid_glm.fit(X_scaled, y)

print("🔹 Mejores parámetros GLM:")
print(grid_glm.best_params_)
print("🔹 Mejor precisión promedio (GLM):", grid_glm.best_score_)
print("🔹 Reporte de clasificación GLM:")
y_pred_glm = grid_glm.predict(X_scaled)
print(classification_report(y, y_pred_glm))

# ----------------------------
# Random Forest
# ----------------------------
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
grid_rf.fit(X_scaled, y)

print("\n🔹 Mejores parámetros Random Forest:")
print(grid_rf.best_params_)
print("🔹 Mejor precisión promedio (RF):", grid_rf.best_score_)
print("🔹 Reporte de clasificación RF:")
y_pred_rf = grid_rf.predict(X_scaled)
print(classification_report(y, y_pred_rf))
