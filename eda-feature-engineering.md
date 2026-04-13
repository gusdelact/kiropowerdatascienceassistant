# EDA y Feature Engineering

## Etapa 2: Script 02_eda.py — Análisis Exploratorio

Este script lee los datos crudos de `data/raw/`, genera visualizaciones en `outputs/figures/` y un reporte en `outputs/reports/`. El científico de datos revisa los resultados antes de pasar a Feature Engineering.

### Estructura del Script

```python
#!/usr/bin/env python3
"""02_eda.py — Análisis Exploratorio de Datos.

Lee datos crudos, genera visualizaciones y reporte estadístico.
Artefactos: outputs/figures/, outputs/reports/eda_report.json

Uso: uv run python scripts/02_eda.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from pathlib import Path
from lib.plot_utils import save_figure

# Cargar configuración
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Crear directorios de salida
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/reports").mkdir(parents=True, exist_ok=True)

# Cargar datos crudos
raw_path = config["data"]["raw_path"]
df = pd.read_csv(f"{raw_path}/train.csv")  # ajustar nombre de archivo

print(f"Dataset shape: {df.shape}")
print(f"Columnas: {list(df.columns)}")
print(f"\n{df.info()}")
print(f"\n{df.describe()}")

# --- Análisis de Valores Faltantes ---
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis", ax=ax)
ax.set_title("Mapa de Valores Faltantes")
save_figure(fig, "outputs/figures/missing_values_heatmap.png")

# --- Distribuciones de Variables Numéricas ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
n_cols = min(len(numeric_cols), 3)
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
for i, col in enumerate(numeric_cols):
    ax = axes.flatten()[i] if len(numeric_cols) > 1 else axes
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribución: {col}")
plt.tight_layout()
save_figure(fig, "outputs/figures/distributions.png")

# --- Variables Categóricas ---
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
cat_summary = {}
for col in cat_cols:
    cat_summary[col] = {
        "unique": int(df[col].nunique()),
        "top_values": df[col].value_counts().head(5).to_dict()
    }

# --- Matriz de Correlación ---
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de Correlación")
save_figure(fig, "outputs/figures/correlation_matrix.png")

# --- Outliers ---
fig, axes = plt.subplots(1, min(len(numeric_cols), 6), figsize=(4 * min(len(numeric_cols), 6), 5))
for i, col in enumerate(numeric_cols[:6]):
    ax = axes[i] if len(numeric_cols) > 1 else axes
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
plt.tight_layout()
save_figure(fig, "outputs/figures/outliers_boxplots.png")

# --- Generar Reporte ---
report = {
    "shape": {"rows": df.shape[0], "columns": df.shape[1]},
    "numeric_columns": numeric_cols,
    "categorical_columns": cat_cols,
    "missing_values": missing_pct.to_dict(),
    "categorical_summary": cat_summary,
    "basic_stats": df.describe().to_dict(),
}

with open("outputs/reports/eda_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\n✅ EDA completado.")
print(f"   Figuras en: outputs/figures/")
print(f"   Reporte en: outputs/reports/eda_report.json")
print("\n👤 SIGUIENTE: Revisa las figuras y el reporte.")
print("   Ajusta config.yaml si necesitas cambiar columnas o target.")
print("   Cuando estés listo: uv run python scripts/03_feature_engineering.py")
```

### lib/plot_utils.py (función compartida)

```python
import matplotlib.pyplot as plt
from pathlib import Path

def save_figure(fig, path, dpi=150):
    """Guarda figura y cierra para liberar memoria."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Guardada: {path}")
```

---

## Etapa 3: Script 03_feature_engineering.py

Lee datos crudos, aplica transformaciones y guarda el dataset procesado en `data/processed/`. También guarda los artefactos de preprocesamiento (scaler, encoder) para reutilizarlos en la app de UI.

### Estructura del Script

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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Cargar configuración
with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

# Cargar datos crudos
raw_path = config["data"]["raw_path"]
df = pd.read_csv(f"{raw_path}/train.csv")

# Eliminar columnas innecesarias
drop_cols = config["features"].get("drop_columns", [])
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Separar features y target
target = config["features"]["target"]
X = df.drop(columns=[target])
y = df[target]

# Separar train/test ANTES de cualquier transformación (evitar data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"],
    stratify=y if y.dtype == "object" or y.nunique() < 20 else None
)

# Definir transformadores
numeric_features = config["features"]["numeric"]
categorical_features = config["features"]["categorical"]

# Filtrar solo columnas que existen
numeric_features = [c for c in numeric_features if c in X.columns]
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

# Fit SOLO en train, transform en ambos
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Obtener nombres de features después de transformación
feature_names = numeric_features.copy()
if categorical_features:
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names.extend(cat_names)

# Guardar datasets procesados
train_df = pd.DataFrame(X_train_processed, columns=feature_names)
train_df[target] = y_train.values
train_df.to_csv("data/processed/train.csv", index=False)

test_df = pd.DataFrame(X_test_processed, columns=feature_names)
test_df[target] = y_test.values
test_df.to_csv("data/processed/test.csv", index=False)

# Guardar preprocessor para reutilizar en la app
joblib.dump(preprocessor, "models/preprocessor.joblib")

print(f"\n✅ Feature Engineering completado.")
print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
print(f"   Features: {len(feature_names)}")
print(f"   Preprocessor guardado en: models/preprocessor.joblib")
print(f"\n👤 SIGUIENTE: Revisa data/processed/ y verifica las transformaciones.")
print(f"   Cuando estés listo: uv run python scripts/04_train.py")
```

## Funciones Compartidas: lib/feature_utils.py

```python
"""Funciones reutilizables de feature engineering."""
import pandas as pd
import numpy as np

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

def detect_outliers_iqr(series, factor=1.5):
    """Detecta outliers usando el método IQR."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series[(series < lower) | (series > upper)]
```

## Gotchas Comunes

- **Data leakage**: NUNCA hagas fit del scaler/imputer con datos de test. Usa `fit_transform` solo en train y `transform` en test
- **One-hot encoding con alta cardinalidad**: Si una columna tiene >20 categorías, considera target encoding
- **Outliers**: No elimines automáticamente. Revisa en la etapa de EDA y decide en config.yaml
- **Distribuciones sesgadas**: Aplica `np.log1p()` antes de escalar

## Consultar Documentación

```
tavily_skill(query="sklearn ColumnTransformer remainder passthrough", library="scikit-learn", language="python")
tavily_skill(query="pandas cut bins labels qcut", library="pandas", language="python")
```
