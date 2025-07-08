import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import mode


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================
# 1. Cargar dataset y dividir 80-20
# =============================
df = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\4_dataset_final_recortado.csv')
target_col = 'escenario'
X_original = df.drop(columns=[target_col])
y_original = df[target_col]

X, X_test, y, y_test = train_test_split(
    X_original, y_original,
    test_size=0.2,
    random_state=42,
    stratify=y_original
)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# =============================
# 2. Clase TransformerNet (TNN)
# =============================
class TransformerNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerNet, self).__init__()
        self.embedding = nn.Linear(input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128,
                                                   dropout=0.1, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# =============================
# 3. Wrapper estilo sklearn para TNN
# =============================
class TNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, num_classes, epochs=10, lr=0.005, batch_size=64):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerNet(input_dim, num_classes).to(self.device)
        self.classes_ = np.arange(num_classes)

    def fit(self, X, y):
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                                 torch.tensor(y, dtype=torch.long))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

# =============================
# 4. Inicializar modelos con hiperparámetros óptimos
# =============================
glm = LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial',
                         max_iter=500, penalty='l2', random_state=SEED)
rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                            random_state=SEED)
tnn = TNNClassifier(input_dim=X.shape[1], num_classes=len(np.unique(y)), 
                    epochs=10, lr=0.005, batch_size=64)

modelos = {'GLM': glm, 'Random Forest': rf, 'TNN': tnn}
predicciones = {}

# =============================
# 5. Entrenar y evaluar modelos individuales
# =============================
for nombre, modelo in modelos.items():
    print(f"\n📌 Entrenando modelo: {nombre}")
    modelo.fit(X, y)

    # ===== Evaluar en entrenamiento (80%) =====
    y_pred_train = modelo.predict(X)
    print(f"\n🧪 Métricas en conjunto de ENTRENAMIENTO (80%) para {nombre}:\n")
    print(classification_report(y, y_pred_train, digits=4))
    acc = accuracy_score(y, y_pred_train)
    prec = precision_score(y, y_pred_train, average='weighted')
    rec = recall_score(y, y_pred_train, average='weighted')
    f1 = f1_score(y, y_pred_train, average='weighted')
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-score: {f1:.4f}")

    # ===== Evaluar en prueba (20%) =====
    y_pred_test = modelo.predict(X_test)
    predicciones[nombre] = y_pred_test
    print(f"\n🔍 Métricas en conjunto de PRUEBA (20%) para {nombre}:\n")
    print(classification_report(y_test, y_pred_test, digits=4))
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, average='weighted')
    rec = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-score: {f1:.4f}")


# =============================
# 6. Votación en conjunto (votación dura)
# =============================
print("\n🗳️ Aplicando votación por mayoría (ensemble)...")
pred_ensemble = np.vstack([predicciones['GLM'], predicciones['Random Forest'], predicciones['TNN']])
y_ensemble, _ = mode(pred_ensemble, axis=0, keepdims=False)

# =============================
# 7. Evaluar ensemble por clase (4 clases)
# =============================
print("\n📊 Reporte de clasificación para Ensemble (por clase):")
print(classification_report(y_test, y_ensemble, digits=4))

# =============================
# 8. Matriz de confusión actualizada (4 clases)
# =============================
etiquetas = ['Normal', 'DoS', 'DDoS', 'MitM']
ConfusionMatrixDisplay.from_predictions(
    y_test, y_ensemble, display_labels=etiquetas, values_format='.4g'
)
plt.title('Matriz de Confusión - Ensemble')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

