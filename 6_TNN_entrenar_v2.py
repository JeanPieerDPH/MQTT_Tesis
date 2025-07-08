"""
Script para entrenar TNN con distintas combinaciones de hiperparámetros
y seleccionar la mejor mediante validación cruzada 10-fold.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time

# ------------------------------
# 1. Configuración inicial
# ------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------
# 2. Modelo Transformer personalizado
# ------------------------------
class TransformerNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerNet, self).__init__()
        self.embedding = nn.Linear(input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
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

# ------------------------------
# 3. Funciones de entrenamiento y evaluación
# ------------------------------
def train_model(model, train_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, val_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(targets.numpy())
            y_pred.extend(preds.cpu().numpy())

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

# ------------------------------
# 4. Cargar y preparar los datos
# ------------------------------
df = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\4_dataset_final_recortado.csv')
target_col = 'escenario'
X = df.drop(columns=[target_col])
y = df[target_col]

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

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# ------------------------------
# 5. Búsqueda de hiperparámetros
# ------------------------------
batch_sizes = [32, 64]
learning_rates = [0.001, 0.0005]
epoch_options = [6, 10]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k_folds = 10
results = []

# Probar cada combinación
for bs in batch_sizes:
    for lr in learning_rates:
        for ep in epoch_options:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
            fold_metrics = []

            print(f"\n📦 Probar: batch_size={bs}, lr={lr}, epochs={ep}")
            start_time = time.time()

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                model = TransformerNet(X.shape[1], len(np.unique(y))).to(device)

                train_ds = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
                val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

                train_model(model, train_loader, ep, lr, device)
                metrics = evaluate_model(model, val_loader, device)
                fold_metrics.append(metrics)

                print(f"  Fold {fold+1}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

            # Promedio de los 10 folds
            avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
            avg_metrics.update({'batch_size': bs, 'learning_rate': lr, 'epochs': ep})
            results.append(avg_metrics)

            elapsed = time.time() - start_time
            print(f"🕒 Tiempo total: {elapsed:.1f} segundos")

# ------------------------------
# 6. Mostrar mejores combinaciones
# ------------------------------
df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values(by='f1', ascending=False)

print("\n✅ Mejores combinaciones ordenadas por F1-score:")
print(df_sorted[['batch_size', 'learning_rate', 'epochs', 'accuracy', 'precision', 'recall', 'f1']].head(5))

# Guardar resultados si deseas
# df_sorted.to_csv('resultados_busqueda_tnn.csv', index=False)
