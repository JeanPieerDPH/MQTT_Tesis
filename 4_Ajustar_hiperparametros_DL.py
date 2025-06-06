import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from keras_tuner.tuners import RandomSearch
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ========== Cargar tu dataset ==========
df = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\dataset_final_recortado.csv')  # Reemplaza con tu archivo
target_col = 'escenario'  # Reemplaza con el nombre de tu columna etiqueta

X = df.drop(columns=[target_col]).values
y = df[target_col].values

# ========== One-hot encoding si es multiclase ==========
num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes=num_classes)

# ========== Función para crear el modelo ==========
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))
    
    # Añadir capas densas
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh'])
        ))
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))

    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ========== Tuner ==========
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='tnn_tuning'
)

# ========== Validación cruzada ==========
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 1
for train_idx, val_idx in kf.split(X, y):
    print(f"\n📁 Fold {fold}")
    tuner.search(
        X[train_idx], to_categorical(y[train_idx], num_classes=num_classes),
        epochs=30,
        validation_data=(X[val_idx], to_categorical(y[val_idx], num_classes=num_classes)),
        verbose=1
    )
    fold += 1

# ========== Mostrar el mejor modelo ==========
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n✅ Mejores hiperparámetros encontrados:")
for param in best_hps.values:
    print(f"{param}: {best_hps.get(param)}")

# Entrenar mejor modelo
model = tuner.hypermodel.build(best_hps)
model.fit(X, y_cat, epochs=30, batch_size=32, verbose=1)

# Evaluar
y_pred = np.argmax(model.predict(X), axis=1)
print("\n📊 Reporte de clasificación:")
print(classification_report(y, y_pred))
