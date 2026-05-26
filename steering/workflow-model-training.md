# Workflow: Entrenamiento de Modelos (General)

> Heurísticas basadas en problemas de clasificación con desbalance, regresión con datos tabulares, y problemas de fronteras no lineales.
>
> Para heurísticas específicas de cada familia, ver los archivos `models-*.md`.

> **Pre-requisitos bloqueantes**: (1) `notes/00_business_context.md` (ver `business-context.md`) — la Pregunta 4 (métrica primaria + umbral) define el `scoring` del `RandomizedSearchCV` / `GridSearchCV`, y la Pregunta 3 vs 4 (asimetría de costos) define `class_weight` o `scale_pos_weight` en clasificación; (2) `notes/03_design_modeling.md` (ver `theory-driven-design.md`). El `scoring` del search NO se elige por default: sale del documento de contexto.

Este steering cubre el flujo **común a todos los modelos**:
1. **Confirmación del modelo con el usuario (OBLIGATORIO)**
2. Carga del dataset procesado
3. Manejo de desbalance (SMOTE)
4. Búsqueda de hiperparámetros (GridSearchCV / RandomizedSearchCV)
5. Cross-Validation
6. Construcción del modelo final y serialización

Para detalles de hiperparámetros, grids y diagnósticos específicos, consulta el archivo `models-*.md` de la familia que corresponda.

---

## ⚠️ Paso 0: Confirmación Obligatoria del Modelo con el Usuario

**REGLA INQUEBRANTABLE:** Antes de entrenar cualquier modelo, se DEBE preguntar al usuario qué modelo (o modelos) desea utilizar. Nunca asumir ni elegir automáticamente.

### Qué hacer:

1. Presentar al usuario las familias de modelos disponibles según su tipo de problema:

| Tipo de problema | Modelos disponibles |
|---|---|
| Regresión | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `RandomForestRegressor`, `ExtraTreesRegressor`, `XGBRegressor`, `SVR` |
| Clasificación binaria | `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier`, `SVC`, `LinearSVC` |
| Clasificación multiclase | `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier`, `SVC` |

2. Ofrecer una recomendación basada en las características del dataset (tamaño, linealidad, desbalance).
3. **ESPERAR** la respuesta explícita del usuario.
4. Solo después de la confirmación, proceder con el entrenamiento.

### Ejemplo de interacción:

> "He completado el EDA y feature engineering. Para tu problema de clasificación binaria con desbalance moderado, las opciones de modelo son:
>
> - `LogisticRegression` — interpretable, buen baseline
> - `RandomForestClassifier` — robusto, captura no linealidades
> - `XGBClassifier` — mejor rendimiento típico, requiere más tuning
>
> Mi recomendación es empezar con `LogisticRegression` como baseline y comparar con `RandomForestClassifier`. ¿Qué modelo(s) quieres entrenar?"

### Qué NO hacer:

- ❌ Elegir el modelo sin preguntar
- ❌ Asumir que el usuario quiere el modelo "más potente"
- ❌ Saltarse este paso porque "es obvio"
- ❌ Entrenar un modelo y después preguntar si está bien

---

## Fundamento Teórico

### ¿Por qué Cross-Validation?

Una sola partición train/test puede dar un estimado ruidoso del error de generalización. K-fold CV reduce esa varianza promediando K particiones. Para fundamentos consulta `rag-books-mcp` siguiendo `theory-rag-guide.md` — query: `"k-fold cross validation choice of K bias variance"`.

**Regla práctica:**
- 5-fold CV es el default razonable
- 10-fold CV cuando los datos son pocos (< 1000 obs)
- LOOCV (Leave-One-Out) sólo para datasets muy pequeños

### Bias-Variance del estimador de CV

Más folds → menor bias, mayor varianza, más costo computacional.
Menos folds → mayor bias, menor varianza, más rápido.

`cv=5` balancea bien.

---

## Paso 1: Cargar Datos Procesados

```python
import pandas as pd
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

target = config["features"]["target"]
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_test = test_df.drop(columns=[target])
y_test = test_df[target]
```

> Nota: el split, la imputación y la estandarización ya se hicieron en `03_feature_engineering.py`. No repetir aquí.

## Paso 2: Manejo de Desbalance — SMOTE

**Aplica solo a clasificación con clases desbalanceadas. SOLO sobre train. NUNCA sobre test:**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(y_train_resampled.value_counts(normalize=True))

# Usar el dataset balanceado para entrenar
# Evaluar SIEMPRE sobre el test original (sin SMOTE)
```

**Cómo funciona SMOTE:**
1. Toma un ejemplo de la clase minoritaria
2. Encuentra k vecinos más cercanos de la misma clase
3. Genera un ejemplo sintético interpolando entre el original y un vecino
4. Agrega variedad sin duplicar (evita memorización)

**Alternativas a SMOTE:**
- `class_weight='balanced'` en sklearn
- `scale_pos_weight` en XGBoost
- Calibración de umbral (ver `workflow-validation.md`)

> Cuándo NO usar SMOTE: si el desbalance es leve (clase minoritaria > 30%) o si la clase minoritaria realmente es rara y debe permanecer así.

## Paso 3: Búsqueda de Hiperparámetros

Dos opciones según tamaño del espacio:

### GridSearchCV (espacios pequeños y discretos)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 5, 10, 12, 15]
}

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',          # o 'neg_mean_squared_error' para regresión
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor CV score: {grid.best_score_:.4f}")
```

### RandomizedSearchCV (espacios grandes o continuos)

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_dist = {
    'n_estimators': np.arange(50, 300, 25),
    'max_depth': np.arange(3, 15),
    'learning_rate': [0.005, 0.01, 0.025, 0.05, 0.1]
}

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=30,            # número de combinaciones aleatorias
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
```

**Heurística:** Empezar con RandomizedSearchCV (n_iter=30) para explorar, luego refinar con GridSearchCV alrededor del mejor punto.

### ⚠️ Heurísticas de Escalabilidad para Datasets Grandes

**REGLA CRÍTICA:** Si el dataset de entrenamiento (post-SMOTE si aplica) supera 100,000 filas, aplicar las siguientes optimizaciones ANTES de lanzar la búsqueda de hiperparámetros:

#### 1. Subsamplear para búsqueda de hiperparámetros

Buscar hiperparámetros sobre un subconjunto representativo, luego entrenar el modelo final con todos los datos:

```python
from sklearn.model_selection import train_test_split

# Si post-SMOTE > 100K filas, subsamplear para la búsqueda
if len(X_train_resampled) > 100_000:
    # Usar máximo 100K filas para la búsqueda (estratificado)
    X_search, _, y_search, _ = train_test_split(
        X_train_resampled, y_train_resampled,
        train_size=100_000,
        stratify=y_train_resampled,
        random_state=42
    )
    print(f"⚡ Subsample para búsqueda: {len(X_search):,} filas (de {len(X_train_resampled):,})")
else:
    X_search, y_search = X_train_resampled, y_train_resampled

# Búsqueda sobre el subsample
random_search.fit(X_search, y_search)

# Modelo final con TODOS los datos
best_model = random_search.best_estimator_
best_model.fit(X_train_resampled, y_train_resampled)
```

#### 2. Reducir iteraciones según tamaño

| Filas post-SMOTE | n_iter recomendado | CV folds |
|---|---|---|
| < 50,000 | 30-50 | 5 |
| 50,000 - 200,000 | 20-30 | 5 |
| 200,000 - 1,000,000 | 15-20 | 3 |
| > 1,000,000 | 10-15 (sobre subsample de 100K) | 3 |

#### 3. Evitar doble paralelización (XGBoost + sklearn)

**NUNCA** usar `n_jobs=-1` simultáneamente en XGBoost y en RandomizedSearchCV/GridSearchCV. Causa contención de threads y puede ser más lento que secuencial.

```python
import os
import math

n_cores = os.cpu_count()
# Usar máximo 70% de los cores disponibles para no saturar el sistema
n_cores_70 = max(int(math.floor(n_cores * 0.7)), 1)

# Opción A: Paralelizar sklearn, XGBoost secuencial (RECOMENDADO para búsqueda)
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(nthread=1, tree_method="hist"),  # 1 thread por modelo
    n_jobs=min(n_cores_70, 5),  # sklearn paraleliza los folds (70% cores)
    ...
)

# Opción B: XGBoost paralelo, sklearn secuencial (RECOMENDADO para modelo final)
final_model = XGBClassifier(
    **best_params,
    nthread=n_cores_70,  # XGBoost usa 70% de los cores
    tree_method="hist"
)
# Entrenar sin n_jobs en sklearn
final_model.fit(X_train_resampled, y_train_resampled)
```

#### 4. Optimización para Apple Silicon (M1/M2/M3/M4)

En Macs con Apple Silicon, los cores son heterogéneos (performance + efficiency). Heurísticas:

```python
import os
import math
import subprocess

def get_apple_silicon_config():
    """Detecta cores en Apple Silicon y configura threads óptimos (usa 70% máximo)."""
    total_cores = os.cpu_count()

    # Intentar detectar performance cores
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True, text=True
        )
        perf_cores = int(result.stdout.strip())
    except:
        perf_cores = total_cores // 2  # Fallback conservador

    # Aplicar límite del 70% sobre los performance cores
    usable_cores = max(int(math.floor(perf_cores * 0.7)), 1)

    return {
        "total_cores": total_cores,
        "perf_cores": perf_cores,
        "usable_cores_70pct": usable_cores,
        # Para sklearn n_jobs: 70% de performance cores, máximo 4
        "sklearn_jobs": min(usable_cores, 4),
        # Para XGBoost nthread: 70% de performance cores
        "xgb_threads": max(usable_cores, 2),
    }

config_hw = get_apple_silicon_config()

# Búsqueda de hiperparámetros
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(nthread=1, tree_method="hist"),
    n_jobs=config_hw["sklearn_jobs"],
    ...
)

# Modelo final
final_model = XGBClassifier(
    **best_params,
    nthread=config_hw["xgb_threads"],
    tree_method="hist"
)
```

**Reglas para Apple Silicon:**
- `tree_method="hist"` siempre (más eficiente en CPU moderno)
- NUNCA `n_jobs=-1` en ambos niveles simultáneamente
- Usar máximo 70% de los cores disponibles para no saturar el sistema
- Para búsqueda: sklearn paraleliza (n_jobs=70% perf_cores), XGBoost secuencial (nthread=1)
- Para modelo final: XGBoost paraleliza (nthread=70% perf_cores), sin sklearn paralelo
- Si el dataset post-SMOTE > 500K filas: subsamplear a 100K para búsqueda

#### 5. Alternativa a SMOTE para datasets grandes

Si la clase mayoritaria ya tiene > 500K filas, SMOTE generará un dataset enorme. Alternativas más eficientes:

```python
# Opción 1: scale_pos_weight en XGBoost (NO genera filas extra)
ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
model = XGBClassifier(scale_pos_weight=ratio, tree_method="hist")

# Opción 2: class_weight='balanced' en sklearn
model = RandomForestClassifier(class_weight='balanced')

# Opción 3: SMOTE con sampling_strategy limitado (no igualar 100%)
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Minoritaria = 30% de mayoritaria
```

**Heurística de decisión:**
- Dataset < 50K filas → SMOTE completo (sampling_strategy='auto')
- Dataset 50K-500K filas → SMOTE con `sampling_strategy=0.3` o `class_weight='balanced'`
- Dataset > 500K filas → `scale_pos_weight` o `class_weight='balanced'` (NO usar SMOTE)

### Métricas de Scoring Comunes

| Problema | Scoring recomendado |
|---|---|
| Regresión | `neg_mean_squared_error`, `r2`, `neg_mean_absolute_error` |
| Clasificación binaria balanceada | `accuracy`, `roc_auc` |
| Clasificación binaria desbalanceada | `f1`, `average_precision` |
| Clasificación multiclase | `f1_weighted`, `f1_macro` |

## Paso 4: Cross-Validation Manual (sin búsqueda)

Útil para verificar la robustez de un modelo con hiperparámetros fijos:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

print(f"F1 por fold: {scores}")
print(f"F1 promedio: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**Para regresión:** usar `KFold` en vez de `StratifiedKFold`.

## Paso 5: Construcción del Modelo Final

Después de la búsqueda, reentrenar con los mejores hiperparámetros sobre TODO el train set:

```python
best_model = grid.best_estimator_
# o instanciar de nuevo:
# best_model = ModelClass(**grid.best_params_)
# best_model.fit(X_train, y_train)
```

`GridSearchCV` y `RandomizedSearchCV` con `refit=True` (default) ya devuelven el modelo reentrenado en `.best_estimator_`.

## Paso 6: Serialización

```python
import joblib
import json
from datetime import datetime

joblib.dump(best_model, "models/model.joblib")

# Guardar metadata
train_meta = {
    "timestamp": datetime.now().isoformat(),
    "model_type": type(best_model).__name__,
    "best_params": grid.best_params_ if hasattr(grid, "best_params_") else {},
    "cv_score": float(grid.best_score_) if hasattr(grid, "best_score_") else None,
    "n_features": X_train.shape[1],
    "n_samples": X_train.shape[0],
    "feature_names": list(X_train.columns),
}

with open("outputs/metrics/train_metadata.json", "w") as f:
    json.dump(train_meta, f, indent=2, default=str)
```

---

## Etapa 4 del Pipeline: Script `04_train.py`

```python
#!/usr/bin/env python3
"""04_train.py — Entrenamiento del modelo.

Lee datos procesados, entrena el modelo configurado, guarda artefactos.
Artefactos: models/model.joblib, outputs/metrics/train_metadata.json

Uso: uv run python scripts/04_train.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import yaml
import joblib
import json
from pathlib import Path
from datetime import datetime
from lib.model_utils import build_model

with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("models").mkdir(parents=True, exist_ok=True)
Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

target = config["features"]["target"]
train_df = pd.read_csv("data/processed/train.csv")
X_train = train_df.drop(columns=[target])
y_train = train_df[target]

# Manejo opcional de desbalance
if config["training"].get("smote", False):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=config["data"]["random_state"])
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"SMOTE aplicado. Distribución: {y_train.value_counts(normalize=True).to_dict()}")

model_type = config["model"]["type"]
model_params = config["model"]["params"]

print(f"Entrenando modelo: {model_type}")
print(f"Parámetros: {model_params}")

model = build_model(model_type, model_params)
model.fit(X_train, y_train)

joblib.dump(model, "models/model.joblib")

train_meta = {
    "timestamp": datetime.now().isoformat(),
    "model_type": model_type,
    "params": model_params,
    "n_features": X_train.shape[1],
    "n_samples": X_train.shape[0],
    "feature_names": list(X_train.columns),
}
with open("outputs/metrics/train_metadata.json", "w") as f:
    json.dump(train_meta, f, indent=2, default=str)

print(f"\n✅ Entrenamiento completado.")
print(f"   Modelo en: models/model.joblib")
print(f"   Metadata en: outputs/metrics/train_metadata.json")
print(f"\n👤 SIGUIENTE: uv run python scripts/05_validate.py")
```

### `lib/model_utils.py`

```python
"""Construcción de modelos según config."""

def build_model(model_type, params):
    if model_type == "linear_regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)
    elif model_type == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    elif model_type == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(**params)
    elif model_type == "elastic_net":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**params)
    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    elif model_type == "decision_tree_regressor":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(**params)
    elif model_type == "random_forest_regressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    elif model_type == "random_forest_classifier":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    elif model_type == "extra_trees_regressor":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**params)
    elif model_type == "xgboost_classifier":
        import xgboost as xgb
        return xgb.XGBClassifier(**params)
    elif model_type == "xgboost_regressor":
        import xgboost as xgb
        return xgb.XGBRegressor(**params)
    elif model_type == "svc":
        from sklearn.svm import SVC
        return SVC(**params)
    elif model_type == "linear_svc":
        from sklearn.svm import LinearSVC
        return LinearSVC(**params)
    elif model_type == "svr":
        from sklearn.svm import SVR
        return SVR(**params)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")
```

---

## Reglas Universales

1. **Preguntar obligatoriamente al usuario qué modelo utilizar** — NUNCA entrenar sin confirmación explícita del usuario sobre el modelo elegido
2. **Nunca aplicar SMOTE antes del split** — causa data leakage
3. **Validación cruzada siempre** — mínimo 5 folds
4. **El accuracy no es suficiente con clases desbalanceadas** — usar F1, precision, recall
5. **Estandarizar después del split** — fit en train, transform en test (esto se hizo en Feature Engineering)
6. **Reproducibilidad** — fijar `random_state` en train_test_split, modelo y CV

## Decidiendo qué Familia Usar

| Pregunta | Familia recomendada | Steering |
|---|---|---|
| Relaciones aproximadamente lineales, alta interpretabilidad | Regresión Lineal + Ridge/Lasso | `models-linear-regression.md` |
| Relaciones no lineales, robustez sin tuning intensivo | Random Forest | `models-trees-rf.md` |
| Necesitas el mejor rendimiento posible | XGBoost / Boosting | `models-ensemble.md` |
| Pocos datos, alta dimensionalidad, fronteras complejas | SVM | `models-svm.md` |
| Quieres comparar todos | Benchmarking (ver `models-trees-rf.md`) | `models-trees-rf.md` |

## Consultar Documentación

Para verificar firmas de sklearn o imbalanced-learn, usa la búsqueda web del agente (`remote_web_search` / `web_fetch`) apuntando a la doc oficial. Queries típicos:

- `"sklearn GridSearchCV scoring multiple metrics"`
- `"imblearn SMOTE k_neighbors sampling_strategy"`
- `"sklearn StratifiedKFold cross_val_score"`

## Siguiente paso

Una vez entrenado el modelo, evaluar su desempeño con `workflow-validation.md`.
