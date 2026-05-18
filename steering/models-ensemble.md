# Modelos: Ensembles — Bagging y Boosting

> Cubre Bagging (Random Forest Classifier) y Boosting (XGBoost) para clasificación binaria con desbalance.
> Ejemplo: dataset de crédito (30,000 obs, 25 variables, desbalance ~78/22).
>
> **Nota sobre paralelización:** Los ejemplos usan `n_jobs=-1` por simplicidad, pero en producción usar máximo 70% de los cores: `n_jobs=max(int(os.cpu_count() * 0.7), 1)`. Ver `workflow-model-training.md` §Heurísticas de Escalabilidad para la configuración completa.

Esta guía es **específica para clasificación con métodos ensemble** y enfatiza el manejo del desbalance. Para la base teórica de bagging/boosting, consulta `rag-books-mcp` (ver `theory-rag-guide.md`) con queries como `"bagging variance bias decomposition"` o `"gradient boosting friedman"`. Para Random Forest aplicado a regresión sin desbalance, ver `models-trees-rf.md`.

---

## Cuándo Usar Esta Familia

| Situación | Recomendación |
|---|---|
| Clasificación con desbalance moderado/fuerte | Random Forest + SMOTE + calibración de umbral |
| Necesitas el mejor rendimiento posible y puedes tunear | XGBoost |
| Dataset grande y limpio | XGBoost con early stopping |
| Quieres robustez sin tuning intensivo | Random Forest |
| Detectas que un solo modelo no basta | Stacking (ver final del archivo) |

### Bagging vs Boosting (Resumen)

| Aspecto | Bagging (RF) | Boosting (XGBoost) |
|---|---|---|
| Reduce | Varianza | Bias |
| Construcción | Paralela (independiente) | Secuencial (correctiva) |
| Sobreajuste con más árboles | No | Sí (necesita early stopping) |
| Sensibilidad a hiperparámetros | Baja | Alta |
| Robustez al ruido | Alta (promedia ruido) | Media (puede memorizarlo) |

---

## 1. Random Forest Classifier (Bagging)

### Implementación con Búsqueda de Hiperparámetros

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': [5, 6, 7, 8, 9, 10, 11, 12],
    'max_depth': np.arange(3, 12),
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_cv = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='f1',         # IMPORTANTE: f1 con desbalance, no accuracy
    random_state=42,
    n_jobs=-1
)
rf_cv.fit(X_train, y_train)

print(f"Mejores hiperparámetros: {rf_cv.best_params_}")
print(f"Mejor F1 (CV): {rf_cv.best_score_:.4f}")

y_pred = rf_cv.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Heurísticas Específicas para RF Classifier

| Parámetro | Recomendación | Razón |
|---|---|---|
| `scoring='f1'` | Siempre con desbalance | El accuracy ignora la clase minoritaria |
| `n_estimators` | 100-200 | Más no daña, pero rendimiento marginal |
| `max_features` | `sqrt(n_features)` o probar 5-12 | Reduce correlación entre árboles |
| `max_depth` | 5-9 | Evita overfitting |
| `class_weight` | `'balanced'` o dict | Alternativa a SMOTE |

---

## 2. XGBoost (Boosting)

### Pre-requisito en macOS

XGBoost requiere OpenMP:

```bash
brew install libomp
```

### Implementación con Búsqueda de Hiperparámetros

```python
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

param_dist = {
    'max_depth': np.arange(3, 8),
    'n_estimators': [50, 100, 150, 200, 250],
    'learning_rate': [0.01, 0.025, 0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.01, 0.1, 1]
}

xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_cv = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)
xgb_cv.fit(X_train_s, y_train)

print(f"Mejores hiperparámetros: {xgb_cv.best_params_}")
y_pred = xgb_cv.predict(X_test_s)
print(classification_report(y_test, y_pred))
```

### Hiperparámetros Clave de XGBoost

| Parámetro | Rol | Heurística |
|---|---|---|
| `learning_rate` (η) | Shrinkage / regularización | 0.01-0.1. Bajo + más estimadores = mejor generalización |
| `n_estimators` | Pasos de boosting | 100-300. Con learning_rate bajo subir más |
| `max_depth` | Complejidad por árbol | 3-6. Boosting prefiere árboles poco profundos |
| `subsample` | Fracción de filas por árbol | 0.7-0.9 |
| `colsample_bytree` | Fracción de features por árbol | 0.6-0.9 |
| `reg_alpha` (L1) | Lasso en pesos de hojas | 0-1 |
| `reg_lambda` (L2) | Ridge en pesos de hojas | 0-1 |
| `scale_pos_weight` | Peso de clase positiva | `n_negativos / n_positivos` |

### Early Stopping (Recomendado)

```python
# Separar un validation set del train
from sklearn.model_selection import train_test_split as split_val
X_tr, X_val, y_tr, y_val = split_val(X_train_s, y_train, test_size=0.15, random_state=42)

xgb_final = XGBClassifier(
    **xgb_cv.best_params_,
    early_stopping_rounds=20,
    eval_metric='logloss',
    random_state=42
)
xgb_final.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=10)

print(f"Mejor iteración: {xgb_final.best_iteration}")
```

### ⚠️ Trampa Crítica: XGBoost + early_stopping + cross_val_score

**Nunca pases un modelo con `early_stopping_rounds` a `cross_val_score()`. Falla con:**

```
ValueError: Must have at least 1 validation dataset for early stopping.
```

**Solución:** dos modelos separados.

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. Modelo SIN early stopping para CV
xgb_cv_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    random_state=42, eval_metric='logloss'
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_cv_model, X_train_s, y_train, cv=cv, scoring='f1')

# 2. Modelo CON early stopping para entrenar el final
xgb_final = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    random_state=42, eval_metric='logloss',
    early_stopping_rounds=20
)
xgb_final.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=10)
```

---

## 3. Manejo de Desbalance en Ensemble (Crítico)

### Opción A — SMOTE post-split

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_s, y_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=9, random_state=42, n_jobs=-1)
rf.fit(X_train_resampled, y_train_resampled)

# Evaluar SIEMPRE en el test original (sin SMOTE)
y_pred = rf.predict(X_test_s)
print(classification_report(y_test, y_pred))
```

### Opción B — `class_weight` o `scale_pos_weight`

```python
# Random Forest
rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
rf_balanced.fit(X_train_s, y_train)

# XGBoost
ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_balanced = XGBClassifier(scale_pos_weight=ratio, random_state=42, n_jobs=-1)
xgb_balanced.fit(X_train_s, y_train)
```

### Opción C — Calibración del umbral (post-entrenamiento)

```python
from sklearn.metrics import precision_recall_curve

y_probs = rf.predict_proba(X_test_s)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
precisions, recalls = precisions[:-1], recalls[:-1]

f1s = np.where(
    (precisions + recalls) != 0,
    2 * (precisions * recalls) / (precisions + recalls),
    0
)
optimal_threshold = thresholds[np.argmax(f1s)]
print(f"Umbral óptimo: {optimal_threshold:.4f}")

y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal))
```

> Ver `workflow-validation.md` para la explicación completa de calibración con curva precision-recall.

### Comparación de Resultados

| Métrica | RF (umbral 0.5) | RF (umbral calibrado) | XGBoost |
|---|---|---|---|
| Accuracy | ~79% | ~81% | ~82% |
| F1 clase minoritaria | ~0.33 | ~0.56 | ~0.49 |
| Recall clase minoritaria | ~22% | ~56% | ~38% |

**Conclusiones:**
- Calibrar el umbral mejora dramáticamente el recall (22% → 56%)
- XGBoost gana en accuracy pero RF calibrado domina F1 de la clase minoritaria
- La elección final depende del costo de negocio: ¿falsos negativos o falsos positivos pesan más?

---

## 4. Comparación Completa Bagging vs Boosting

### Cuándo elegir cada uno

**Random Forest cuando:**
- Quieres robustez con poco tuning
- El dataset tiene mucho ruido
- Necesitas OOB error como diagnóstico rápido
- Tiempo de entrenamiento importa menos que simplicidad

**XGBoost cuando:**
- Necesitas exprimir el máximo rendimiento
- Puedes invertir tiempo en tuning
- El dataset es grande y limpio
- Hay desbalance fuerte (`scale_pos_weight`)
- Necesitas manejo nativo de valores faltantes

### Recordatorios

- **RF nunca sobreajusta con más árboles** → puedes subir `n_estimators` sin miedo
- **XGBoost SÍ sobreajusta** → usar early stopping siempre
- **El learning rate de XGBoost es el hiperparámetro más importante** → priorizarlo en el tuning

---

## 5. Stacking (Avanzado)

Combinar las predicciones de múltiples modelos base usando un meta-modelo:

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train_s, y_train)
print(classification_report(y_test, stacking.predict(X_test_s)))
```

**Regla de ESL:** El meta-modelo debe ser simple (regresión logística) para evitar overfitting.

---

## Pre-requisitos

1. **Estandarización** — opcional para árboles, pero práctica común para reusar el mismo preprocessor
2. **One-Hot encoding** para categóricas (sklearn) o **DMatrix** con tipos categóricos en XGBoost
3. **Variable objetivo como `int`** para XGBoost
4. **Manejo de desbalance** — al menos una de las tres opciones (SMOTE, class_weight, calibración)

## Gotchas

- **Mezclar `early_stopping_rounds` con `cross_val_score`** → `ValueError`. Usar dos modelos.
- **Olvidar reproducibilidad** → fijar `random_state` en split, modelo, CV y SMOTE.
- **`scoring='accuracy'` con desbalance** → métricas engañosas. Usar `f1` o `roc_auc`.
- **No instalar `libomp` en macOS** → XGBoost falla al importar.
- **Aplicar SMOTE al test** → leakage que infla artificialmente las métricas.

## Conexión con la Teoría

Para fundamentos (varianza de promedios correlacionados, gradient boosting, AdaBoost), consulta `rag-books-mcp` siguiendo `theory-rag-guide.md`. Queries útiles: `cite_foundation(topic="bagging variance reduction")`, `get_section(book="esl", chapter=10, section="10.10")` para gradient boosting, `get_section(book="esl", chapter=10, section="10.1")` para AdaBoost.

## Consultar Documentación

```
"xgboost early_stopping_rounds eval_set verbose"            → docs xgboost (vía remote_web_search)
"imblearn SMOTE k_neighbors sampling_strategy"              → docs imbalanced-learn (vía remote_web_search)
"sklearn StackingClassifier final_estimator passthrough"    → docs sklearn (vía remote_web_search)
```
