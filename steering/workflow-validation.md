# Workflow: Validación de Modelos

> Cubre métricas y visualizaciones para clasificación (con y sin desbalance) y regresión.
> Incluye calibración del umbral con curva precision-recall, benchmarking de múltiples modelos, y feature importance.
>
> **Nota sobre paralelización:** Los ejemplos usan `n_jobs=-1` por simplicidad, pero en producción usar máximo 70% de los cores: `n_jobs=max(int(os.cpu_count() * 0.7), 1)`. Ver `workflow-model-training.md` §Heurísticas de Escalabilidad.

> **Pre-requisitos bloqueantes**: (1) `notes/00_business_context.md` (ver `business-context.md`) — la Pregunta 4 (métrica primaria + umbral mínimo) define qué métrica se reporta como decisiva y cuál es el criterio de aceptación del modelo, y la Pregunta 3 vs 4 (asimetría de costos) define dónde se calibra el umbral en la curva precision-recall; (2) `notes/04_design_validation.md` (ver `theory-driven-design.md`). El criterio "pasa / no pasa" no es un default: sale del umbral mínimo declarado en el contexto.

Este steering cubre cómo evaluar modelos **independiente de la familia**:
1. Métricas para clasificación
2. Métricas para regresión
3. Visualizaciones (confusion matrix, ROC, precision-recall, residuales)
4. Calibración del umbral de clasificación
5. Comparación entre modelos
6. Feature importance

---

## 1. Métricas para Clasificación

### Reporte estándar

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Cuándo usar cada métrica

| Métrica | Pregunta que responde | Cuándo usar |
|---|---|---|
| Accuracy | % correctos sobre el total | Solo con clases balanceadas |
| Precision | De los predichos positivos, ¿cuántos lo son? | Cuando el costo de falsos positivos es alto (spam) |
| Recall | De los positivos reales, ¿cuántos detectamos? | Cuando el costo de falsos negativos es alto (cáncer, fraude) |
| F1-score | Media armónica de precision y recall | Métrica principal con desbalance |
| ROC-AUC | Capacidad de discriminación (independiente del umbral) | Comparar modelos sin fijar umbral |
| PR-AUC | Como ROC pero enfocado en clase positiva | Mejor que ROC con desbalance fuerte |

### Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Clase 0', 'Clase 1'],
            yticklabels=['Clase 0', 'Clase 1'])
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.title('Confusion Matrix')
plt.savefig('outputs/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
```

**Interpretación:**
- **TN (top-left):** verdaderos negativos
- **FP (top-right):** falsos positivos (error tipo I)
- **FN (bottom-left):** falsos negativos (error tipo II)
- **TP (bottom-right):** verdaderos positivos

### ROC Curve (clasificación binaria)

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('outputs/figures/roc_curve.png', dpi=150, bbox_inches='tight')
```

**Heurística:**
- AUC > 0.9 → excelente
- 0.8 < AUC ≤ 0.9 → bueno
- 0.7 < AUC ≤ 0.8 → aceptable
- AUC ≈ 0.5 → modelo no aprende

---

## 2. Calibración del Umbral (Crítico con Desbalance)

**El umbral 0.5 NO es óptimo cuando hay desbalance.** Calibrar usando la curva precision-recall:

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

y_probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
precisions = precisions[:-1]
recalls = recalls[:-1]

# F1 para cada umbral
f1_scores = np.where(
    (precisions + recalls) != 0,
    2 * (precisions * recalls) / (precisions + recalls),
    0
)

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Umbral óptimo: {optimal_threshold:.4f}")
print(f"F1 con umbral 0.5: {f1_score(y_test, (y_probs >= 0.5).astype(int)):.4f}")
print(f"F1 con umbral óptimo: {f1_scores.max():.4f}")

# Predicciones con umbral calibrado
y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal))
```

**Resultado típico:** El umbral óptimo suele caer entre 0.25 y 0.40 cuando hay desbalance, y el recall mejora dramáticamente (22% → 56%) sin sacrificar mucha precision.

### Visualización de la curva PR

```python
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, f1_scores, label='F1')
plt.axvline(optimal_threshold, color='red', linestyle='--',
            label=f'Umbral óptimo = {optimal_threshold:.3f}')
plt.xlabel('Umbral')
plt.ylabel('Score')
plt.title('Calibración del Umbral')
plt.legend()
plt.savefig('outputs/figures/threshold_calibration.png', dpi=150, bbox_inches='tight')
```

---

## 3. Métricas para Regresión

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
```

| Métrica | Interpretación |
|---|---|
| MSE | Penaliza errores grandes (cuadrático). Útil para optimización. |
| RMSE | Mismas unidades que el target. Más interpretable. |
| MAE | Robusto a outliers. Misma unidad que target. |
| R² | Fracción de varianza explicada. 0 = no explica nada, 1 = perfecto. |

### Residual Plot

```python
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicho')
plt.ylabel('Residual (real - predicho)')
plt.title('Residual Plot')
plt.savefig('outputs/figures/residual_plot.png', dpi=150, bbox_inches='tight')
```

**Buscar:**
- Residuales dispersos alrededor de 0 sin patrón → modelo bien especificado
- Forma de embudo → heterocedasticidad → considerar transformación log
- Patrón curvo → falta capturar no linealidad → considerar árboles o features polinomiales

---

## 4. Comparación entre Modelos (Benchmarking)

**Patrón recomendado:** comparar varios modelos con la misma partición train/test.

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

modelos = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=0.1),
    'Lasso': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
    'Árbol de regresión': DecisionTreeRegressor(max_depth=9, min_samples_split=20),
    'Random Forest': RandomForestRegressor(max_depth=9, n_estimators=150)
}

def comparar_modelos(X_train, X_test, y_train, y_test, modelos):
    """Entrena y evalúa cada modelo, devuelve dict con MSE."""
    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)
        mse = mean_squared_error(y_test, predicciones)
        resultados[nombre] = mse
        print(f"{nombre} - MSE: {mse:.4f}")
    return resultados

resultados = comparar_modelos(X_train, X_test, y_train, y_test, modelos)
```

### Versión robusta con Cross-Validation

Una sola partición es ruidosa. Mejor:

```python
from sklearn.model_selection import cross_val_score

def comparar_modelos_cv(X, y, modelos, cv=5, scoring='neg_mean_squared_error'):
    resultados = {}
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        resultados[nombre] = {
            'mean': -scores.mean() if scoring.startswith('neg_') else scores.mean(),
            'std': scores.std()
        }
        print(f"{nombre}: {resultados[nombre]['mean']:.4f} (+/- {resultados[nombre]['std']:.4f})")
    return resultados
```

### Visualización del benchmarking

```python
import pandas as pd

df_resultados = pd.DataFrame(list(resultados.items()), columns=['Modelo', 'MSE'])
df_resultados = df_resultados.sort_values('MSE')

plt.figure(figsize=(10, 5))
sns.barplot(data=df_resultados, x='MSE', y='Modelo', palette='viridis')
plt.title('Comparación de Modelos (MSE)')
plt.savefig('outputs/figures/benchmarking.png', dpi=150, bbox_inches='tight')
```

---

## 5. Feature Importance

### Para árboles (RF, XGBoost)

```python
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    importances = importances.sort_values(ascending=True).tail(15)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='steelblue')
    plt.title('Top 15 Feature Importance')
    plt.savefig('outputs/figures/feature_importance.png', dpi=150, bbox_inches='tight')
```

### Permutation Importance (más confiable, funciona con cualquier modelo)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=-1)

importances = pd.Series(result.importances_mean, index=X_test.columns)
importances = importances.sort_values(ascending=True).tail(15)
```

### Coeficientes (modelos lineales)

```python
if hasattr(model, "coef_"):
    coefs = pd.Series(model.coef_.ravel(), index=X_test.columns)
    coefs.sort_values().plot(kind='barh', figsize=(10, 6))
    plt.title('Coeficientes del modelo')
    plt.axvline(0, color='black', linewidth=0.5)
    plt.savefig('outputs/figures/coefficients.png', dpi=150, bbox_inches='tight')
```

---

## Etapa 5 del Pipeline: Script `05_validate.py`

```python
#!/usr/bin/env python3
"""05_validate.py — Validación con métricas y figuras.

Carga modelo entrenado, evalúa en test, genera reporte JSON y figuras.
Artefactos: outputs/metrics/validation.json, outputs/figures/

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
    mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve
)
from lib.plot_utils import save_figure

with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

target = config["features"]["target"]
scoring = config["training"]["scoring"]
is_classification = scoring in ("accuracy", "f1", "roc_auc", "precision", "recall")

model = joblib.load("models/model.joblib")
test_df = pd.read_csv("data/processed/test.csv")
X_test = test_df.drop(columns=[target])
y_test = test_df[target]
y_pred = model.predict(X_test)

if is_classification:
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title("Confusion Matrix")
    save_figure(fig, "outputs/figures/confusion_matrix.png")

    # ROC + calibración de umbral (binario con probabilidades)
    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        metrics["auc_roc"] = float(auc)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
        save_figure(fig, "outputs/figures/roc_curve.png")

        # Calibración del umbral
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        precisions, recalls = precisions[:-1], recalls[:-1]
        f1s = np.where((precisions + recalls) != 0,
                       2 * (precisions * recalls) / (precisions + recalls), 0)
        optimal_threshold = float(thresholds[np.argmax(f1s)])
        metrics["optimal_threshold"] = optimal_threshold
        metrics["f1_at_optimal_threshold"] = float(f1s.max())

        y_pred_opt = (y_proba >= optimal_threshold).astype(int)
        metrics["classification_report_optimal"] = classification_report(
            y_test, y_pred_opt, output_dict=True
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, precisions, label="Precision")
        ax.plot(thresholds, recalls, label="Recall")
        ax.plot(thresholds, f1s, label="F1")
        ax.axvline(optimal_threshold, color="red", linestyle="--",
                   label=f"Óptimo = {optimal_threshold:.3f}")
        ax.set_xlabel("Umbral"); ax.legend(); ax.set_title("Calibración del Umbral")
        save_figure(fig, "outputs/figures/threshold_calibration.png")
else:
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Residual"); ax.set_title("Residual Plot")
    save_figure(fig, "outputs/figures/residual_plot.png")

# Feature importance (si aplica)
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    importances = importances.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 15 Feature Importance")
    save_figure(fig, "outputs/figures/feature_importance.png")

with open("outputs/metrics/validation.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print(f"\n✅ Validación completada.")
print(f"   Métricas en: outputs/metrics/validation.json")
print(f"   Figuras en: outputs/figures/")
```

---

## Conexión con Steering Files de Teoría

| Decisión práctica | Fundamento teórico | Cómo consultarlo |
|---|---|---|
| ¿Por qué CV con K=5? | Bias-variance del estimador de CV | `rag-books-mcp` query: `"k-fold cross validation choice of K"` (ver `theory-rag-guide.md`) |
| ¿Por qué scoring='f1'? | Métricas para clases desbalanceadas | `rag-books-mcp` query: `"F1 precision recall imbalanced classification"` |
| ¿Por qué calibrar el umbral? | Distribución de probabilidades posterior | `rag-books-mcp` query: `"probability calibration threshold precision recall curve"` |
| ¿Cómo interpretar feature importance? | Bagging vs Boosting | `rag-books-mcp` `cite_foundation(topic="permutation importance vs gini importance")` |

## Consultar Documentación

```
"sklearn precision_recall_curve average_precision_score"  → docs sklearn (vía remote_web_search)
"sklearn permutation_importance n_repeats"                → docs sklearn (vía remote_web_search)
```

## Siguiente paso

Una vez validado, proceder a publicar dataset y modelo en HF Hub. Ver `huggingface-workflows.md`.
