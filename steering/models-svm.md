# Modelos: Support Vector Machines (SVM)

> Cubre: clasificación lineal y no lineal con `LinearSVC`/`SVC`, kernels (linear, rbf, poly), preprocesamiento obligatorio, búsqueda con `RandomizedSearchCV`, y `OneClassSVM` para detección de outliers. Ejemplo: dataset `make_moons` (no lineal) y otros de sklearn.
>
> **Nota sobre paralelización:** Los ejemplos usan `n_jobs=-1` por simplicidad, pero en producción usar máximo 70% de los cores: `n_jobs=max(int(os.cpu_count() * 0.7), 1)`. Ver `workflow-model-training.md` §Heurísticas de Escalabilidad.

Esta guía es **específica para SVM**. Para el flujo común (split, CV, métricas), ver `workflow-model-training.md` y `workflow-validation.md`.

---

## Cuándo Usar Esta Familia

| Situación | Recomendación |
|---|---|
| Datasets pequeños/medianos (< 50K obs) | SVM (sobre todo con kernel) |
| Fronteras de decisión complejas no lineales | `SVC(kernel='rbf')` |
| Alta dimensionalidad (texto, genómica) | `LinearSVC` (escalable) |
| Detección de anomalías sin etiquetas | `OneClassSVM` |
| Datasets muy grandes (> 100K obs) | Cambiar a árboles o boosting |
| Necesitas probabilidades calibradas | Preferir RF/XGBoost (SVM las da mal) |

---

## 1. SVM Lineal (LinearSVC)

### Idea Intuitiva

Encuentra el hiperplano que separa las clases con el **máximo margen** posible. Las observaciones que tocan el margen son los **vectores de soporte**.

Formulación (margen suave):

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_i \xi_i \quad \text{s.t.}\quad y_i(w^Tx_i + b) \geq 1 - \xi_i,\ \xi_i \geq 0$$

- `C` grande → poco margen, pocos errores en train (riesgo de overfit)
- `C` pequeño → mucho margen, más tolerancia a errores

### Implementación

```python
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Pipeline: SVM SIEMPRE necesita estandarización
svm_lin = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(loss='hinge', max_iter=10000, random_state=42))
])

param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(svm_lin, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Mejor C: {grid.best_params_['svm__C']}")
print(classification_report(y_test, grid.predict(X_test)))
```

### Cuándo usar `LinearSVC` vs `SVC(kernel='linear')`

| `LinearSVC` | `SVC(kernel='linear')` |
|---|---|
| Optimizado para datasets grandes | Más lento con muchos datos |
| Solo lineal | Soporta cualquier kernel |
| No `predict_proba` (sin Platt scaling) | `predict_proba` con `probability=True` |
| Penalización L1 disponible | Solo L2 |

**Regla:** Si solo necesitas lineal y dataset grande → `LinearSVC`. Si quieres flexibilidad → `SVC(kernel='linear')`.

---

## 2. SVM No Lineal (SVC con Kernels)

### Kernels Disponibles

| Kernel | Forma | Cuándo usar |
|---|---|---|
| `linear` | $K(x, x') = x^Tx'$ | Relaciones lineales o alta dimensionalidad |
| `rbf` (Gaussiano) | $K(x, x') = \exp(-\gamma\|x-x'\|^2)$ | **Default**: fronteras locales/curvas |
| `poly` | $K(x, x') = (\gamma x^Tx' + r)^d$ | Interacciones polinomiales explícitas |
| `sigmoid` | $K(x, x') = \tanh(\gamma x^Tx' + r)$ | Raro; suele ser inferior a `rbf` |

### Implementación con RBF

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import RandomizedSearchCV

# Dataset no lineal de ejemplo
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

svm_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42))
])

param_dist = {
    'svm__C': [0.1, 1, 10, 100, 1000],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
}

grid = RandomizedSearchCV(svm_rbf, param_dist, n_iter=20, cv=5,
                          scoring='f1', random_state=42, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Mejores parámetros: {grid.best_params_}")
print(classification_report(y_test, grid.predict(X_test)))
```

### Hiperparámetros Clave

| Parámetro | Efecto | Heurística |
|---|---|---|
| `C` | Trade-off margen vs errores | Probar `[0.1, 1, 10, 100, 1000]` |
| `gamma` (RBF) | Influencia de cada punto | `'scale'` por default; explorar `[0.001, 0.01, 0.1, 1]` |
| `degree` (poly) | Grado del polinomio | 2-4. Cuidado: aumenta dramáticamente el costo |
| `kernel` | Tipo de transformación | Empezar con `rbf`, comparar con `linear` |
| `class_weight` | Para desbalance | `'balanced'` o dict |

**Reglas:**
- `C` y `gamma` interactúan — buscar conjuntamente
- `gamma` muy alto → overfitting (cada punto domina su vecindario)
- `gamma` muy bajo → underfitting (se vuelve casi lineal)

### Visualizar la Frontera de Decisión (didáctico, solo 2D)

```python
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title=""):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='RdYlBu')
    plt.title(title)
    plt.savefig('outputs/figures/svm_decision_boundary.png', dpi=150, bbox_inches='tight')

plot_decision_boundary(grid.best_estimator_, X_test, y_test, "SVM RBF — frontera de decisión")
```

---

## 3. SVM para Regresión (SVR)

```python
from sklearn.svm import SVR

svr = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])

param_dist = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 0.01, 0.1, 1],
    'svr__epsilon': [0.01, 0.1, 0.5, 1.0]   # tolerancia del tubo
}

grid = RandomizedSearchCV(svr, param_dist, n_iter=20, cv=5,
                          scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
grid.fit(X_train, y_train)
```

`epsilon` define el tubo dentro del cual no se penaliza el error. Más grande → modelo más simple.

---

## 4. Detección de Outliers — OneClassSVM

**Aprendizaje no supervisado:** entrenas solo con observaciones "normales" y el modelo identifica outliers.

```python
from sklearn.svm import OneClassSVM

oc_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('oc', OneClassSVM(kernel='rbf', gamma='scale', nu=0.05))
])

oc_svm.fit(X_train_normal)   # solo datos normales

# Predict: 1 = inlier, -1 = outlier
predictions = oc_svm.predict(X_new)
n_outliers = (predictions == -1).sum()
print(f"Outliers detectados: {n_outliers} ({100*n_outliers/len(predictions):.2f}%)")
```

`nu` ∈ (0, 1] aproxima la fracción esperada de outliers en el train. Default 0.5 suele ser muy alto; probar 0.01-0.1.

---

## 5. Probabilidades con SVC

`SVC` no genera probabilidades directamente. Hay que activar Platt scaling (es lento y a veces inconsistente con `predict`):

```python
svc_proba = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svc_proba.fit(X_train, y_train)
y_proba = svc_proba.predict_proba(X_test)[:, 1]
```

> **Recomendación:** Si necesitas probabilidades bien calibradas, prefiere Random Forest o Logistic Regression. SVM no es ideal para esto.

---

## Pre-requisitos para esta Familia

1. **Estandarización OBLIGATORIA** — SVM se basa en distancias, sin escalar los resultados son malos.
2. **Imputación previa** — sklearn no acepta NaN.
3. **One-Hot encoding** para categóricas.
4. **Probar primero `kernel='rbf'`** — suele ser el mejor default.
5. **Datasets no muy grandes** — SVM con kernel es O(n²) o O(n³). Por encima de 50K-100K obs, considerar otras familias.

---

## Heurística de Tuning Eficiente

```python
# 1. Empezar con grid grueso
param_dist_grueso = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto'],
    'svm__kernel': ['linear', 'rbf']
}

# 2. Refinar con grid fino alrededor del mejor punto
param_grid_fino = {
    'svm__C': [5, 10, 20, 50],
    'svm__gamma': [0.05, 0.1, 0.2, 0.5]
}
```

---

## Gotchas

- **No estandarizar antes de SVM** → resultados desastrosos (los features con escalas grandes dominan)
- **Datasets gigantes con SVC kernel** → entrenamiento muy lento. Alternativas: `LinearSVC`, `SGDClassifier(loss='hinge')`
- **`SVC(probability=True)` lento** → activa Platt scaling con CV interno. Solo activar cuando necesario.
- **`gamma='auto'` (legacy)** → comportamiento distinto a `'scale'`. Default actual es `'scale'`.
- **Pipeline obligatorio para CV** — si haces el `StandardScaler` fuera del pipeline en CV, hay leakage entre folds.

## Conexión con la Teoría

- **Margin maximization y vectores de soporte** — fundamento del SVM
- **Kernel trick** — proyectar a espacios de alta dimensión sin computarlos explícitamente
- **Conexión con regularización** — `C` es inverso al `λ` de Ridge/Lasso (`C` chico = más regularización)

Para profundizar, consulta `rag-books-mcp` siguiendo `theory-rag-guide.md` — queries útiles: `search_theory(query="SVM regularization parameter C kernel")`, `get_section(book="esl", chapter=12)` (SVMs y discriminantes flexibles).

## Consultar Documentación

```
"sklearn SVC kernel rbf gamma scale auto"            → docs sklearn (vía remote_web_search)
"sklearn LinearSVC vs SVC kernel linear differences" → docs sklearn (vía remote_web_search)
"sklearn OneClassSVM nu novelty detection"           → docs sklearn (vía remote_web_search)
```
