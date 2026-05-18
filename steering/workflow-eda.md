# Workflow: Análisis Exploratorio de Datos (EDA)

> Este steering cubre el flujo de EDA para problemas de clasificación y regresión.
> Ejemplo principal: dataset de crédito (30,000 obs, 25 variables, clasificación binaria con desbalance ~78/22).
> También aplica a problemas de regresión (ej: predicción de precios de vivienda).

## 📌 Posición en el protocolo Theory-Driven Design

EDA está **exento** del protocolo de `theory-driven-design.md`: NO requiere un `notes/0N_design_*.md` previo y NO exige consultas obligatorias al RAG antes de codificar `02_eda.py`. La razón está en `theory-driven-design.md` §Regla 4 (excepciones): "EDA *informa* el diseño de FE, no al revés."

Lo que sí se espera del EDA respecto al RAG:

- **Salida obligatoria**: el `outputs/reports/eda_report.json` y las observaciones cualitativas (skew del target, multicolinealidad, desbalance, categorías raras, outliers) son los **insumos** que dispararán las consultas al RAG en la siguiente fase (Feature Engineering, ver `workflow-feature-engineering.md` §"ANTES DE CODIFICAR").
- **Consulta puntual permitida (no obligatoria)**: si durante el EDA detectas un patrón que claramente cambia decisiones aguas abajo (ej. multicolinealidad > 0.9 entre varias variables, skew > 2 en el target, desbalance < 5% para la minoritaria), puedes hacer una consulta acotada al RAG para anclar la observación. Esa consulta NO sustituye a las queries obligatorias de la fase de FE — solo las prepara.

Resumen rápido:

| Pregunta | Respuesta |
|---|---|
| ¿Necesito `notes/00_design_eda.md`? | No, no existe. |
| ¿Debo consultar el RAG antes de escribir `02_eda.py`? | No es obligatorio. |
| ¿Puedo consultar el RAG durante el EDA? | Sí, puntualmente, cuando un hallazgo amerite anclaje teórico. |
| ¿Qué disparará las consultas obligatorias al RAG? | Los hallazgos del EDA, al iniciar la fase de FE. |

---

## Fundamento Teórico

> Basado en: ISLP Cap 2 (Statistical Learning), ESL Cap 7 (Model Assessment)

### ¿Por Qué Hacemos EDA?

El objetivo del EDA es entender la **función f** que relaciona X con Y (ISLP §2.1):

$$Y = f(X) + \epsilon$$

Antes de modelar, necesitamos entender:
1. **La naturaleza de f**: ¿Es lineal? ¿Hay interacciones? ¿Qué variables importan?
2. **La distribución de X**: ¿Hay outliers? ¿Multicolinealidad? ¿Valores faltantes?
3. **La distribución de Y**: ¿Está balanceada? ¿Es continua o categórica?
4. **El ruido ε**: ¿Cuánta variabilidad irreducible hay?

### Error Reducible vs Irreducible

De ISLP ecuación 2.3:
$$E(Y - \hat{Y})^2 = [f(X) - \hat{f}(X)]^2 + \text{Var}(\epsilon)$$

El EDA nos ayuda a reducir el **error reducible** al:
- Identificar las variables más informativas (reducir bias)
- Detectar problemas que aumentarían la varianza (outliers, multicolinealidad)
- Elegir transformaciones apropiadas (log, polinomios)

---

## Flujo de EDA Recomendado

```
1. Inspección inicial (shape, info, head, describe)
2. Verificar duplicados y nulos
3. Analizar distribución de la variable objetivo
4. Clasificar variables (numéricas vs categóricas)
5. Distribuciones univariadas
6. Detección de outliers
7. Identificar categorías raras o inválidas
8. Análisis de correlación (multicolinealidad)
```

---

## Paso 1: Inspección Inicial

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/raw/dataset.csv')

print(f"Shape: {df.shape}")
df.head()
df.info()
df.describe()
```

## Paso 2: Verificar Duplicados y Nulos

```python
# Duplicados
num_duplicados = len(df[df.duplicated()])
print(f"Número de duplicados: {num_duplicados}")

# Nulos por columna
print(df.isnull().sum())
```

**Heurística:**
- Si hay duplicados → eliminar con `df.drop_duplicates()` SOLO si tiene sentido en el dominio
- Si una columna tiene > 50% nulos → considerar eliminarla
- Si los nulos son < 5% → imputar (mediana para numéricas, moda para categóricas)
- Si los nulos son 5-50% → analizar si son MCAR/MAR/MNAR antes de decidir

## Paso 3: Distribución de la Variable Objetivo

**Antes de modelar, entender el balance de clases (clasificación) o la distribución (regresión):**

```python
# Clasificación
df['default_pymt_nm'].value_counts(normalize=True)
# Salida típica:
# 0    0.7788
# 1    0.2212

# Regresión
df['MEDV'].describe()
df['MEDV'].hist(bins=30)
```

**Reglas clave:**
- **Clasificación:** Si la clase minoritaria < 25%, planificar SMOTE o calibración de umbral (ver `workflow-validation.md`)
- **Regresión:** Si la distribución es muy sesgada → considerar `np.log1p()` antes de modelar

## Paso 4: Clasificar Variables (Heurística del umbral 13)

**Variables con pocos valores únicos suelen ser categóricas aunque sean numéricas:**

```python
# Inspección rápida
df.nunique()

# Clasificación automática
variables_numericas = []
variables_categoricas = []

for col in df.columns:
    if df[col].nunique() > 13:
        variables_numericas.append(col)
    else:
        variables_categoricas.append(col)

print(f'Variables numéricas: {variables_numericas}')
print(f'Variables categóricas: {variables_categoricas}')
```

**Justificación:** Variables como `PAY_0` con valores -1 a 9 (11 valores únicos) se modelan mejor como categóricas que como numéricas.

**Tipos de variables:**
- **Categóricas nominales** — el orden no importa (colores, ciudades)
- **Categóricas ordinales** — el orden importa (nivel educativo)
- **Numéricas escala de razón** — cero absoluto, hace sentido la multiplicación (ingresos, edad)
- **Numéricas de intervalo** — sin cero absoluto, solo suma (temperatura en °C)

## Paso 5: Distribuciones Univariadas

### Variables numéricas

```python
fig, axes = plt.subplots(nrows=(len(variables_numericas) + 2) // 3, ncols=3,
                         figsize=(15, 4 * ((len(variables_numericas) + 2) // 3)))
for i, col in enumerate(variables_numericas):
    ax = axes.flatten()[i]
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Distribución: {col}')
plt.tight_layout()
plt.savefig('outputs/figures/distribuciones_numericas.png', dpi=150, bbox_inches='tight')
```

**Buscar:**
- Distribuciones bimodales o multimodales → posible variable categórica oculta
- Distribuciones muy sesgadas → considerar transformación log
- Picos extremos → posibles outliers

### Variables categóricas

```python
for col in variables_categoricas:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True))
```

**Buscar:**
- Categorías con < 1% de los registros → eliminar (paso 7)
- Categorías sin sentido en el dominio (ej: `EDUCATION = 0` cuando debe ser 1-4)

## Paso 6: Detección de Outliers (Boxplots)

**Visualizar antes de tratar (el tratamiento se hace en Feature Engineering):**

```python
plt.figure(figsize=(15, 12))
plt.suptitle('Visualización de Outliers', fontsize=20, y=1.02)

n_cols = 4
n_rows = (len(variables_numericas) + n_cols - 1) // n_cols
for i, col in enumerate(variables_numericas):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x=df[col])
    plt.xlabel(col)
plt.tight_layout()
plt.savefig('outputs/figures/outliers_boxplots.png', dpi=150, bbox_inches='tight')
```

**No eliminar outliers en EDA — solo identificarlos. El tratamiento (clipping IQR) se aplica en `workflow-feature-engineering.md`.**

## Paso 7: Identificar Categorías Raras o Inválidas

```python
# Ejemplo: EDUCATION tiene categorías 0, 5, 6 con < 1% cada una → eliminar registros
# MARRIAGE tiene categoría 0 con < 0.2% → eliminar registros

for col in variables_categoricas:
    proporciones = df[col].value_counts(normalize=True)
    raras = proporciones[proporciones < 0.01]
    if len(raras) > 0:
        print(f"\n{col} — categorías raras (< 1%):")
        print(raras)
```

**Decisión:** El tratamiento (eliminar registros) se hace en Feature Engineering, no en EDA.

## Paso 8: Análisis de Correlación

```python
# Solo variables numéricas
corr_matrix = df[variables_numericas].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax)
ax.set_title('Matriz de Correlación')
plt.tight_layout()
plt.savefig('outputs/figures/matriz_correlacion.png', dpi=150, bbox_inches='tight')
```

**Heurísticas:**
- Si dos variables tienen `|corr| > 0.85` → multicolinealidad → eliminar una
- Ejemplo: `BILL_AMT1`-`BILL_AMT6` están muy correlacionadas → quedarse con `BILL_AMT1` y `BILL_AMT6`

**Conexión con la teoría:**
- Multicolinealidad alta → considerar Ridge o Elastic Net (consulta `rag-books-mcp` con query `"ridge regression multicollinearity"` siguiendo `theory-rag-guide.md`)
- Si vas a usar árboles (RF, XGBoost) → la multicolinealidad afecta menos pero `feature_importances_` se diluye

---

## Etapa 2 del Pipeline: Script `02_eda.py`

Este script lee datos crudos de `data/raw/`, genera visualizaciones en `outputs/figures/` y un reporte JSON en `outputs/reports/`. El científico de datos revisa los resultados antes de pasar a Feature Engineering.

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

with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/reports").mkdir(parents=True, exist_ok=True)

raw_path = config["data"]["raw_path"]
df = pd.read_csv(f"{raw_path}/train.csv")

print(f"Dataset shape: {df.shape}")

# 1. Duplicados y nulos
n_duplicados = len(df[df.duplicated()])
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

# 2. Clasificar variables (heurística umbral 13)
variables_numericas = [c for c in df.columns if df[c].nunique() > 13]
variables_categoricas = [c for c in df.columns if df[c].nunique() <= 13]

# 3. Distribuciones numéricas
n_cols = 3
n_rows = (len(variables_numericas) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
for i, col in enumerate(variables_numericas):
    ax = axes.flatten()[i] if len(variables_numericas) > 1 else axes
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribución: {col}")
plt.tight_layout()
save_figure(fig, "outputs/figures/distributions.png")

# 4. Boxplots para detección de outliers
fig, axes = plt.subplots(1, min(len(variables_numericas), 6),
                         figsize=(4 * min(len(variables_numericas), 6), 5))
for i, col in enumerate(variables_numericas[:6]):
    ax = axes[i] if len(variables_numericas) > 1 else axes
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
plt.tight_layout()
save_figure(fig, "outputs/figures/outliers_boxplots.png")

# 5. Matriz de correlación
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df[variables_numericas].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de Correlación")
save_figure(fig, "outputs/figures/correlation_matrix.png")

# 6. Categorías raras
cat_raras = {}
for col in variables_categoricas:
    proporciones = df[col].value_counts(normalize=True)
    raras = proporciones[proporciones < 0.01]
    if len(raras) > 0:
        cat_raras[col] = raras.to_dict()

# Reporte
report = {
    "shape": {"rows": df.shape[0], "columns": df.shape[1]},
    "duplicados": n_duplicados,
    "variables_numericas": variables_numericas,
    "variables_categoricas": variables_categoricas,
    "missing_values": missing_pct.to_dict(),
    "categorias_raras": cat_raras,
    "estadisticos": df.describe().to_dict(),
}

with open("outputs/reports/eda_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\n✅ EDA completado.")
print(f"   Figuras en: outputs/figures/")
print(f"   Reporte en: outputs/reports/eda_report.json")
print("\n👤 SIGUIENTE: Revisa figuras y reporte. Ajusta config.yaml si es necesario.")
print("   Cuando estés listo: uv run python scripts/03_feature_engineering.py")
```

### `lib/plot_utils.py`

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

## Gotchas Comunes

- **No eliminar outliers en EDA** — solo identificarlos. El tratamiento (clipping) va en Feature Engineering.
- **No imputar nulos en EDA** — solo cuantificar. La imputación va en Feature Engineering DESPUÉS del split.
- **No transformar variables en EDA** — solo describir. Las transformaciones van en Feature Engineering.
- **Variables categóricas codificadas como int** — son categóricas aunque pandas las infiera como numéricas.

## Consultar Documentación

Si necesitas verificar una API de pandas, seaborn u otra librería, usa la búsqueda web del agente (`remote_web_search` / `web_fetch`) apuntando a la doc oficial. Ejemplos de queries:

- `"seaborn boxplot orient horizontal"` → docs de seaborn.
- `"pandas value_counts normalize percentage"` → docs de pandas.

Para Gradio usa el server MCP `gradio` (no la web genérica).

## Siguiente paso

Una vez completado el EDA, el siguiente paso es Feature Engineering. Ver `workflow-feature-engineering.md`.
