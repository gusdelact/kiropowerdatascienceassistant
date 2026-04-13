# Entrenamiento y Validación de Modelos

## Etapa 4: Script 04_train.py — Entrenamiento

Lee datos procesados de `data/processed/`, entrena el modelo según `config.yaml`, y guarda el modelo serializado en `models/`. El formato de salida es pickle/joblib para sklearn/XGBoost, .keras para TensorFlow, .pth para PyTorch.

### Estructura del Script

```python
#!/usr/bin/env python3
"""04_train.py — Entrenamiento del modelo.

Lee datos procesados, entrena modelo, guarda en formato serializado.
Artefactos: models/model.joblib (o .pkl, .keras, .pth)

Uso: uv run python scripts/04_train.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from datetime import datetime
from lib.model_utils import build_model, get_model_params

# Cargar configuración
with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("models").mkdir(parents=True, exist_ok=True)
Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

# Cargar datos procesados
target = config["features"]["target"]
train_df = pd.read_csv("data/processed/train.csv")
X_train = train_df.drop(columns=[target])
y_train = train_df[target]

# Construir y entrenar modelo
model_type = config["model"]["type"]
model_params = config["model"]["params"]
output_format = config["model"].get("output_format", "joblib")

print(f"Entrenando modelo: {model_type}")
print(f"Parámetros: {model_params}")
print(f"Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")

model = build_model(model_type, model_params)
model.fit(X_train, y_train)

# Guardar modelo
if output_format in ("joblib", "pickle", "pkl"):
    model_path = f"models/model.{output_format}"
    joblib.dump(model, model_path)
elif output_format == "keras":
    model_path = "models/model.keras"
    model.save(model_path)
elif output_format == "pth":
    import torch
    model_path = "models/model.pth"
    torch.save(model.state_dict(), model_path)

# Guardar metadata de entrenamiento
train_meta = {
    "timestamp": datetime.now().isoformat(),
    "model_type": model_type,
    "params": model_params,
    "output_format": output_format,
    "model_path": model_path,
    "n_features": X_train.shape[1],
    "n_samples": X_train.shape[0],
    "feature_names": list(X_train.columns),
}

with open("outputs/metrics/train_metadata.json", "w") as f:
    json.dump(train_meta, f, indent=2, default=str)

print(f"\n✅ Entrenamiento completado.")
print(f"   Modelo guardado en: {model_path}")
print(f"   Metadata en: outputs/metrics/train_metadata.json")
print(f"\n👤 SIGUIENTE: Revisa la metadata del entrenamiento.")
print(f"   Cuando estés listo: uv run python scripts/05_validate.py")
```

### lib/model_utils.py

```python
"""Funciones de construcción y utilidad de modelos."""

def build_model(model_type, params):
    """Construye un modelo según el tipo y parámetros."""
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    elif model_type == "random_forest_regressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**params)
    elif model_type == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**params)
    elif model_type == "xgboost_regressor":
        import xgboost as xgb
        return xgb.XGBRegressor(**params)
    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    elif model_type == "linear_regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

def get_model_params(model):
    """Extrae parámetros del modelo para logging."""
    if hasattr(model, "get_params"):
        return model.get_params()
    return {}
```

---

## Etapa 5: Script 05_validate.py — Validación con Métricas

Lee el modelo entrenado y los datos de test, calcula métricas y genera visualizaciones de validación (confusion matrix, ROC curve, residual plot, etc.).

### Estructura del Script

```python
#!/usr/bin/env python3
"""05_validate.py — Validación del modelo con métricas.

Carga modelo entrenado, evalúa en test set, genera métricas y figuras.
Artefactos: outputs/metrics/validation.json, outputs/figures/confusion_matrix.png

Uso: uv run python scripts/05_validate.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import yaml
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from lib.plot_utils import save_figure

# Cargar configuración
with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

# Cargar modelo y datos
target = config["features"]["target"]
scoring = config["training"]["scoring"]

model = joblib.load("models/model.joblib")
test_df = pd.read_csv("data/processed/test.csv")
X_test = test_df.drop(columns=[target])
y_test = test_df[target]

y_pred = model.predict(X_test)

# Determinar si es clasificación o regresión
is_classification = scoring in ("accuracy", "f1", "roc_auc", "precision", "recall")

if is_classification:
    # --- Métricas de Clasificación ---
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    save_figure(fig, "outputs/figures/confusion_matrix.png")

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["classification_report"] = report

    # ROC Curve (binario)
    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        metrics["auc_roc"] = float(auc)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        save_figure(fig, "outputs/figures/roc_curve.png")

    print("\n📊 Métricas de Clasificación:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   F1 Score:  {metrics['f1_score']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    if "auc_roc" in metrics:
        print(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")

else:
    # --- Métricas de Regresión ---
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    # Residual Plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    save_figure(fig, "outputs/figures/residual_plot.png")

    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    save_figure(fig, "outputs/figures/predicted_vs_actual.png")

    print("\n📊 Métricas de Regresión:")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE:  {metrics['mae']:.4f}")
    print(f"   R²:   {metrics['r2']:.4f}")

# Feature Importance (si disponible)
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    importances = importances.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Feature Importance")
    save_figure(fig, "outputs/figures/feature_importance.png")

# Guardar métricas
with open("outputs/metrics/validation.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print(f"\n✅ Validación completada.")
print(f"   Métricas en: outputs/metrics/validation.json")
print(f"   Figuras en: outputs/figures/")
print(f"\n👤 SIGUIENTE: Revisa las métricas y figuras de validación.")
print(f"   Si las métricas son aceptables, puedes publicar:")
print(f"   - Dataset curado: uv run python scripts/06_publish_dataset.py")
print(f"   - Modelo:         uv run python scripts/07_publish_model.py")
```

---

## Modelos Avanzados: XGBoost con Early Stopping

Para usar XGBoost con early stopping, modifica `04_train.py`:

```python
# En 04_train.py, después de cargar datos
if model_type == "xgboost":
    from sklearn.model_selection import train_test_split as split_val
    X_tr, X_val, y_tr, y_val = split_val(X_train, y_train, test_size=0.15, random_state=42)

    model = build_model(model_type, model_params)
    model.set_params(early_stopping_rounds=20)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=10)
```

**IMPORTANTE — XGBoost + cross_val_score:**

NUNCA pases un modelo con `early_stopping_rounds` a `cross_val_score()`. Falla con:
```
ValueError: Must have at least 1 validation dataset for early stopping.
```

Solución: usa dos modelos separados:

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. Modelo para cross-validation (SIN early stopping)
xgb_cv = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    random_state=42, eval_metric='mlogloss'
    # NO incluir early_stopping_rounds aquí
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_cv, X, y, cv=cv, scoring='accuracy')

# 2. Modelo final (CON early stopping)
xgb_final = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    random_state=42, eval_metric='mlogloss',
    early_stopping_rounds=20  # Solo aquí
)
xgb_final.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)
```

**Prerequisito macOS**: XGBoost requiere OpenMP. Instalar con `brew install libomp`.

## Modelos Avanzados: TensorFlow

Para TensorFlow, el `04_train.py` necesita una variante:

```python
# Variante TensorFlow en 04_train.py
if model_type == "tensorflow":
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks

    model = tf.keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # ajustar según problema
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100, batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=1
    )
    model.save("models/model.keras")
```

## Modelos Avanzados: PyTorch

```python
# Variante PyTorch en 04_train.py
if model_type == "pytorch":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_tensor = torch.FloatTensor(X_train.values)
    y_tensor = torch.LongTensor(y_train.values)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    class Net(nn.Module):
        def __init__(self, input_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_classes)
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(X_train.shape[1], y_train.nunique()).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        net.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(net(bx), by)
            loss.backward()
            optimizer.step()

    torch.save(net.state_dict(), "models/model.pth")
```

## Consultar Documentación

```
tavily_skill(query="XGBoost early_stopping_rounds eval_set", library="xgboost", language="python")
tavily_skill(query="sklearn cross_val_score StratifiedKFold", library="scikit-learn", language="python")
tavily_skill(query="Keras EarlyStopping ReduceLROnPlateau callbacks", library="tensorflow", language="python")
```
