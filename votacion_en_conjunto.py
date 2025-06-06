import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
# Scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------
# Modelo Transformer personalizado
# -----------------------------
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
        x = x.unsqueeze(1)  # dimensión secuencial
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
# -----------------------------
# Wrapper sklearn para TNN
# -----------------------------
class TNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, num_classes, epochs=30, lr=0.0003572271156665703, batch_size=32):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerNet(input_dim, num_classes).to(self.device)
        self.classes_ = np.arange(num_classes)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'lr': self.lr,
            'batch_size': self.batch_size
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = TransformerNet(self.input_dim, self.num_classes).to(self.device)
        return self
# -----------------------------
# Dataset 
# -----------------------------
df = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\dataset_final_recortado.csv')  # Reemplaza con tu archivo
target_col = 'escenario'  # Cambia esto al nombre real de tu columna objetivo

X = df.drop(columns=[target_col])
y = df[target_col]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separar
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)

# -----------------------------
# Modelos individuales con hiperparámetros ajustados
# -----------------------------
glm = LogisticRegression(
    C=10,
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=500,
    penalty='l2'
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

tnn = TNNClassifier(
    input_dim=X.shape[1],
    num_classes=len(np.unique(y)),
    epochs=30,
    lr=0.0003572271156665703
)

# -----------------------------
# Voting Classifier (votación dura)
# -----------------------------
# Entrenamiento de modelos individuales
glm.fit(X_train, y_train)
rf.fit(X_train, y_train)
tnn.fit(X_train, y_train)

# Predicciones individuales
pred_glm = glm.predict(X_test)
pred_rf = rf.predict(X_test)
pred_tnn = tnn.predict(X_test)

# Votación por mayoría (modo)
from scipy.stats import mode
predictions = np.vstack([pred_glm, pred_rf, pred_tnn])
y_pred, _ = mode(predictions, axis=0, keepdims=False)

# -----------------------------
# Evaluación
# -----------------------------
print("\nClasification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# -----------------------------
# Gráfico de barras
# -----------------------------
import matplotlib.pyplot as plt

# Convertimos las métricas a porcentaje
valores_pct = [v * 100 for v in [accuracy, precision, recall, f1]]
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-score']

plt.figure(figsize=(8, 5))
bars = plt.bar(metricas, valores_pct, color=['royalblue', 'seagreen', 'darkorange', 'crimson'])
plt.ylim(0, 110)  # Eje Y de 0 a 110 para dejar espacio para las etiquetas
plt.title('Rendimiento del Modelo Ensemble')
plt.ylabel('Porcentaje (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el valor encima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


from sklearn.metrics import ConfusionMatrixDisplay
etiquetas = ['Normal', 'DDoS', 'DoS']
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,display_labels=etiquetas,values_format='d')
plt.title('Matriz de Confusión del Ensemble')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
