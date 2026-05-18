# Workflow: Feature Engineering

> Cubre tratamiento de outliers con IQR (factor 1.6), eliminación de categorías raras, conversión de tipos, eliminación de features correlacionadas, estandarización con `StandardScaler`, y construcción de pipelines con `ColumnTransformer`.

## ⚠️ ANTES DE CODIFICAR — Theory-Driven Design (OBLIGATORIO)

**No escribir ni una línea de FE sin haber producido `notes/01_design_fe.md`.** Esta es la primera fase del pipeline donde el protocolo `theory-driven-design.md` (Reglas 1, 3 y 4) es vinculante.

Antes de cualquiera de estas líneas:

```python
# Disparadores que requieren notes/01_design_fe.md ya escrito:
train_test_split(...)
StandardScaler()
OneHotEncoder(...)
SMOTE(...)
np.log1p(y)
pd.qcut(y, ...)  # bins para stratify en regresión
```

el agente DEBE haber:

1. **Consultado `rag-books-mcp`** con al menos las 6 queries que `theory-driven-design.md` §Regla 3 exige para esta fase:
   - Distribución del target y transformaciones (log1p, Box-Cox, Yeo-Johnson).
   - Estrategia de split (estratificado por clase / por bins del target).
   - Manejo de outliers (IQR clipping, winsorización, dejarlos).
   - Escalado: cuándo es obligatorio según familia de modelo candidata.
   - Encoding de categóricas según cardinalidad y familia de modelo candidata.
   - Manejo de desbalance (solo si es clasificación): SMOTE / `class_weight`, siempre post-split.

2. **Producido `notes/01_design_fe.md`** con las 5 secciones obligatorias de `theory-driven-design.md` §Regla 2 (Contexto, Decisiones, Consultas al RAG con citas, Implicaciones para el código, Riesgos). El RAG informa el diseño; las citas en `notes/` son la evidencia.

3. **Verificado** que el docstring de `03_feature_engineering.py` referencia el documento (`Diseño: notes/01_design_fe.md`), como exige la Regla 5.

**Excepciones**: ver `theory-driven-design.md` §Regla 6. Las más relevantes en FE son: fixes puntuales sobre un pipeline ya entrenado, refactor operativo, baseline rápido explícitamente solicitado por el usuario (en cuyo caso el `notes/` se difiere al modelo final).

**Modo degradado** (RAG no responde): producir `notes/01_design_fe.md` igual con cabecera "⚠️ Sin verificación bibliográfica" — ver `theory-driven-design.md` §Regla 7. NO inventar citas a ESL/ISLP.

> Nota sobre EDA: la fase 2 (EDA) está exenta de este protocolo por la excepción explícita en `theory-driven-design.md` §Regla 4 — EDA *informa* el diseño de FE, no al revés. Los hallazgos del EDA (skew del target, multicolinealidad, desbalance, categorías raras) son los **insumos** que disparan las queries de esta fase.

---

## Fundamento Teórico

### Feature Engineering y Bias-Variance

Las decisiones de feature engineering afectan directamente el tradeoff bias-varianza:

| Decisión | Efecto en Bias | Efecto en Varianza |
|---|---|---|
| Agregar más features | ↓ Reduce | ↑ Aumenta |
| Eliminar features irrelevantes | ↑ Puede aumentar ligeramente | ↓ Reduce significativamente |
| Crear interacciones (X1*X2) | ↓ Reduce | ↑ Aumenta |
| Estandarizar | Sin efecto | Sin efecto (pero mejora convergencia) |
| PCA (reducir dimensiones) | ↑ Puede aumentar | ↓ Reduce |
| Tratar outliers (clipping) | ↑ Ligero aumento | ↓ Reduce |

### Selección por Information Gain

La ganancia de información mide cuánto reduce la entropía una feature al particionar el dataset:

$$IG(S, F_i) = H(S) - \sum_{v \in values(F_i)} \frac{|S_v|}{|S|} H(S_v)$$

**Conexión con árboles:** Es exactamente el criterio que usan los árboles de decisión para elegir splits. Por eso:
- Features con alta IG → aparecen en los primeros splits del árbol
- Features con baja IG → aparecen en splits profundos o no aparecen
- Esto se refleja en `feature_importances_` de RF y XGBoost

---

## Flujo de Feature Engineering

```
1. Eliminar registros con categorías raras o inválidas (decididas en EDA)
2. Tratar outliers (clipping IQR sobre variables numéricas)
3. Eliminar features altamente correlacionadas (> 0.85)
4. Conversión de tipos (categóricas → str, target → int si XGBoost)
5. Split train/test ESTRATIFICADO (antes de cualquier transformación dependiente de los datos)
6. Imputación de nulos (fit en train, transform en test)
7. Estandarización numérica (fit en train, transform en test)
8. Encoding de categóricas (One-Hot, Ordinal o Target)
9. (Opcional) Crear features derivadas (interacciones, temporales)
```

> **Regla de oro:** El split se hace ANTES de imputar, escalar o codificar para evitar data leakage.

---

## Paso 1: Eliminar Categorías Raras

Decidido en EDA, aplicado aquí:

```python
# Ejemplo: EDUCATION tiene categorías 0, 5, 6 con < 1%, MARRIAGE tiene 0 con < 0.2%
df = df[~df['EDUCATION'].isin([0, 5, 6])]
df = df[df['MARRIAGE'] != 0]

# O genérico
for col in variables_categoricas:
    proporciones = df[col].value_counts(normalize=True)
    raras = proporciones[proporciones < 0.01].index
    df = df[~df[col].isin(raras)]
```

## Paso 2: Tratamiento de Outliers (IQR Clipping)

**Usar clipping (truncar a las cotas), NO eliminar registros (preserva el tamaño del dataset):**

```python
def ajusta_outliers(df, variables_numericas, factor=1.6):
    """
    Trunca outliers usando rango intercuartílico.
    Factor 1.6 es más conservador que el clásico 1.5.
    """
    for col in variables_numericas:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        rango_iq = q3 - q1
        cota_inf = q1 - factor * rango_iq
        cota_sup = q3 + factor * rango_iq
        df[col] = df[col].clip(cota_inf, cota_sup)
    return df

dataf = ajusta_outliers(df, variables_numericas, factor=1.6)
```

**¿Por qué factor 1.6 y no 1.5?** Es ligeramente más conservador — preserva más datos legítimos. El valor exacto depende del dominio. Probar 1.5, 1.6, 2.0 y elegir el que mejor preserve la señal sin contaminar con extremos.

## Paso 3: Eliminar Features Correlacionadas

```python
# Si BILL_AMT1-6 están correlacionadas (> 0.9 entre sí), quedarse con BILL_AMT1 y BILL_AMT6
columnas_a_eliminar = ['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5']
X = dataf.drop(columns=['ID', 'default_pymt_nm'] + columnas_a_eliminar)
y = dataf['default_pymt_nm']
```

**Detección automática de pares con alta correlación:**

```python
def features_correlacionadas(df, umbral=0.85):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [col for col in upper.columns if any(upper[col] > umbral)]

a_eliminar = features_correlacionadas(dataf[variables_numericas], umbral=0.85)
```

## Paso 4: Conversión de Tipos

```python
# Variables categóricas a str (para que sklearn las trate correctamente)
dataf[variables_categoricas] = dataf[variables_categoricas].astype(str)

# Variable objetivo a int (XGBoost lo requiere)
dataf['default_pymt_nm'] = dataf['default_pymt_nm'].astype(int)
```

## Paso 5: Split Train/Test (CRÍTICO — ANTES de transformar)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,           # Mantener proporción de clases
    random_state=42
)
```

**Reglas críticas:**
- `stratify=y` para clasificación (mantener proporción de clases)
- `test_size=0.25` es un buen default
- Para regresión, no usar `stratify` salvo que el target tenga distribución muy sesgada

## Paso 6: Imputación de Nulos

**Fit SOLO en train, transform en ambos:**

```python
from sklearn.impute import SimpleImputer

# Numéricas: mediana
imputer_num = SimpleImputer(strategy='median')
X_train[variables_numericas] = imputer_num.fit_transform(X_train[variables_numericas])
X_test[variables_numericas] = imputer_num.transform(X_test[variables_numericas])

# Categóricas: moda
imputer_cat = SimpleImputer(strategy='most_frequent')
X_train[variables_categoricas] = imputer_cat.fit_transform(X_train[variables_categoricas])
X_test[variables_categoricas] = imputer_cat.transform(X_test[variables_categoricas])
```

## Paso 7: Estandarización (Variables Numéricas)

**Fit SOLO en train, transform en ambos. NUNCA `fit_transform` en test:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[variables_numericas] = scaler.fit_transform(X_train[variables_numericas])
X_test[variables_numericas] = scaler.transform(X_test[variables_numericas])
```

**¿Cuándo es obligatorio estandarizar?**

| Modelo | Estandarización necesaria |
|---|---|
| Regresión Lineal con Ridge/Lasso/ElasticNet | **Sí** (la penalización es sensible a escala) |
| SVM (todos los kernels) | **Sí** (la distancia depende de la escala) |
| KNN | **Sí** |
| Redes Neuronales | **Sí** (gradiente más estable) |
| Árboles de Decisión / Random Forest | No (los splits son invariantes a escala) |
| XGBoost / Gradient Boosting | No (igual razón) |

**Práctica común:** estandarizar siempre. No daña a los árboles y es necesaria para los demás.

## Paso 8: Encoding de Categóricas

### One-Hot Encoding (default para nominales con pocas categorías)

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[variables_categoricas])
X_test_cat = encoder.transform(X_test[variables_categoricas])

cat_names = encoder.get_feature_names_out(variables_categoricas)
X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_names, index=X_train.index)
X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_names, index=X_test.index)
```

### Pipeline Completo con `ColumnTransformer`

**Recomendado en producción** — encadena imputación + escalado + encoding en un solo objeto:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, variables_numericas),
    ('cat', categorical_transformer, variables_categoricas)
], remainder='drop')

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Guardar el preprocessor para reusarlo en inferencia
import joblib
joblib.dump(preprocessor, 'models/preprocessor.joblib')
```

---

## Etapa 3 del Pipeline: Script `03_feature_engineering.py`

Lee datos crudos, aplica transformaciones, guarda el dataset procesado y el preprocessor serializado.

```python
#!/usr/bin/env python3
"""03_feature_engineering.py — Feature Engineering.

Lee datos crudos, aplica transformaciones, guarda dataset procesado
y artefactos de preprocesamiento.
Artefactos: data/processed/, models/preprocessor.joblib

Uso: uv run python scripts/03_feature_engineering.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from lib.feature_utils import ajusta_outliers, features_correlacionadas

with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

raw_path = config["data"]["raw_path"]
df = pd.read_csv(f"{raw_path}/train.csv")

# 1. Eliminar columnas innecesarias
drop_cols = config["features"].get("drop_columns", [])
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 2. Eliminar registros con categorías raras (configurable)
for col, raras in config["features"].get("categorias_raras", {}).items():
    if col in df.columns:
        df = df[~df[col].isin(raras)]

# 3. Clipping de outliers
numeric_features = config["features"]["numeric"]
numeric_features = [c for c in numeric_features if c in df.columns]
df = ajusta_outliers(df, numeric_features, factor=config["features"].get("iqr_factor", 1.6))

# 4. Separar features y target
target = config["features"]["target"]
X = df.drop(columns=[target])
y = df[target]

# 5. Split ANTES de transformar
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"],
    stratify=y if y.dtype == "object" or y.nunique() < 20 else None
)

# 6. Pipeline de preprocesamiento
categorical_features = config["features"]["categorical"]
categorical_features = [c for c in categorical_features if c in X.columns]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 7. Reconstruir nombres de features
feature_names = numeric_features.copy()
if categorical_features:
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names.extend(cat_names)

# 8. Guardar
train_df = pd.DataFrame(X_train_processed, columns=feature_names)
train_df[target] = y_train.values
train_df.to_csv("data/processed/train.csv", index=False)

test_df = pd.DataFrame(X_test_processed, columns=feature_names)
test_df[target] = y_test.values
test_df.to_csv("data/processed/test.csv", index=False)

joblib.dump(preprocessor, "models/preprocessor.joblib")

print(f"\n✅ Feature Engineering completado.")
print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
print(f"   Features: {len(feature_names)}")
print(f"   Preprocessor en: models/preprocessor.joblib")
print(f"\n👤 SIGUIENTE: Revisa data/processed/.")
print(f"   Cuando estés listo: uv run python scripts/04_train.py")
```

### `lib/feature_utils.py`

```python
"""Funciones reutilizables de feature engineering."""
import pandas as pd
import numpy as np

def ajusta_outliers(df, variables_numericas, factor=1.6):
    """Clipping de outliers con IQR."""
    df = df.copy()
    for col in variables_numericas:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - factor * iqr, q3 + factor * iqr)
    return df

def features_correlacionadas(df, umbral=0.85):
    """Devuelve features con correlación absoluta > umbral con alguna otra."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [col for col in upper.columns if any(upper[col] > umbral)]

def create_temporal_features(df, date_column):
    """Crea features temporales a partir de una columna de fecha."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[f"{date_column}_year"] = df[date_column].dt.year
    df[f"{date_column}_month"] = df[date_column].dt.month
    df[f"{date_column}_day_of_week"] = df[date_column].dt.dayofweek
    df[f"{date_column}_is_weekend"] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    return df

def create_interaction_features(df, col_a, col_b):
    """Crea features de interacción entre dos columnas numéricas."""
    df = df.copy()
    df[f"{col_a}_{col_b}_ratio"] = df[col_a] / (df[col_b] + 1e-8)
    df[f"{col_a}_{col_b}_product"] = df[col_a] * df[col_b]
    return df
```

---

## Gotchas Comunes

- **Data leakage por escalar antes del split** — fit del scaler sólo con train, jamás con todo el dataset
- **Data leakage por SMOTE antes del split** — SMOTE va DESPUÉS del split, sólo sobre train (ver `workflow-model-training.md`)
- **One-hot con alta cardinalidad** — si una columna tiene > 20 categorías, considerar Target Encoding o agrupación
- **Olvidar guardar el preprocessor** — sin él la app de inferencia no podrá transformar datos nuevos
- **Categóricas como int** — sklearn las trata como numéricas; convertirlas a `str` antes de OneHot

## Conexión con la Teoría

- **Multicolinealidad detectada en EDA** → en este paso eliminas features o aplicas Ridge/Elastic Net (consulta `rag-books-mcp` con query `"multicollinearity ridge regression VIF"`, ver `theory-rag-guide.md`)
- **Distribución sesgada del target** → considerar `np.log1p(y)` antes de modelar regresión
- **Selección por Information Gain** → coincide con `feature_importances_` de árboles (consulta `rag-books-mcp` con `cite_foundation(topic="feature importance impurity decrease")`, ver `theory-rag-guide.md`)

## Consultar Documentación

Para verificar firmas de sklearn o pandas, usa la búsqueda web del agente (`remote_web_search` / `web_fetch`) apuntando a la doc oficial. Queries típicos:

- `"sklearn ColumnTransformer remainder passthrough"`
- `"sklearn OneHotEncoder handle_unknown sparse_output"`
- `"pandas qcut bins labels"`

## Siguiente paso

Con datos procesados y preprocessor guardado, pasar a entrenamiento. Ver:
- `workflow-model-training.md` — flujo general común a todos los modelos
- `models-linear-regression.md`, `models-trees-rf.md`, `models-ensemble.md`, `models-svm.md` — guías por familia de modelo
