# Workflow: Análisis Exploratorio de Datos (EDA)

> Este steering cubre el flujo de EDA para problemas de clasificación y regresión.
> Ejemplo principal: dataset de crédito (30,000 obs, 25 variables, clasificación binaria con desbalance ~78/22).
> También aplica a problemas de regresión (ej: predicción de precios de vivienda).

## 📌 Posición en el protocolo Theory-Driven Design

EDA tiene un régimen **especial** dentro de `theory-driven-design.md`: produce su propio documento `notes/00_design_eda.md`, pero con **timing bipartito** distinto al de las demás fases.

- **Antes** de codificar `02_eda.py`: crear `notes/00_design_eda.md` en estado *pre-EDA* con las secciones 1 (Contexto) y 2 (Preguntas guía e hipótesis). No se exigen consultas al RAG todavía.
- **Durante** el EDA: ejecutar `02_eda.py`, generar figuras y `outputs/reports/eda_report.json`, registrar hallazgos.
- **Al cerrar** el EDA, antes de iniciar `notes/01_design_fe.md`: completar las secciones 3 (Consultas al RAG), 4 (Implicaciones / puente a FE) y 5 (Riesgos) de la nota 00. Esta es la transición que **bloquea** la fase de FE si no está cumplida.

La diferencia con FE/Modelado/Validación es que en aquellas fases el RAG va *antes del código* porque las decisiones son a priori. En EDA las decisiones son *a posteriori* del descubrimiento: las consultas al RAG anclan los hallazgos en teoría y los traducen en decisiones para FE.

Resumen rápido:

| Pregunta | Respuesta |
|---|---|
| ¿Necesito `notes/00_design_eda.md`? | **Sí**, con timing bipartito (pre-EDA y al cierre). |
| ¿Debo consultar el RAG antes de escribir `02_eda.py`? | No. Las consultas se hacen al cerrar el EDA, sobre los hallazgos. |
| ¿Puedo consultar el RAG durante el EDA? | Sí, puntualmente, cuando un hallazgo amerite anclaje teórico inmediato. |
| ¿Qué bloquea la transición a FE? | Que `notes/00_design_eda.md` esté en estado **cerrado** (secciones 1-5 completas). |

Plantilla y estructura mínima en `theory-driven-design.md` §"Plantilla mínima de `notes/00_design_eda.md`".

### Consultas obligatorias al cerrar el EDA

Cuando completes la sección 3 de la nota 00, debes haber consultado el RAG sobre todos los hallazgos accionables que dispararán decisiones aguas abajo. Las temáticas mínimas (cuando aplican al dataset):

- Disciplina del flujo iterativo (R4DS Cap. 10) — variación, covariación, qué tipos de pregunta hacer.
- Lectura del target (skew, multimodalidad, ceros estructurales) y consecuencias para FE/modelado.
- Outliers como descubrimiento vs. error (R4DS §10.4 + FES Cap. 6).
- Multicolinealidad y su impacto en familias candidatas (ESL §3.4).
- Patrones de missing data (MCAR / MAR / MNAR) y consecuencias (FES Cap. 8).

Si el dataset no presenta una situación, declarar "no aplica" en la nota y omitir esa consulta.

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

## Filosofía R4DS para EDA (Cap. 10)

> Esta sección integra los principios de **R for Data Science, 2nd Ed.** (Wickham, Çetinkaya-Rundel, Grolemund) en el flujo. R4DS está escrito en R/tidyverse pero las ideas son agnósticas del lenguaje y se traducen 1-a-1 a pandas/seaborn. **Nunca generes código en R para el usuario**; usa R4DS para la disciplina del proceso, no para la sintaxis.

### EDA es un estado mental, no una receta

R4DS Cap. 10 insiste en que el EDA no es una checklist mecánica sino un **ciclo iterativo**:

1. **Generar preguntas** sobre los datos.
2. **Buscar respuestas** visualizando, transformando y modelando.
3. **Refinar** las preguntas o generar nuevas a partir de lo aprendido.

> "EDA is fundamentally a creative process. And like most creative processes, the key to asking quality questions is to generate a large quantity of questions." — R4DS §10.2

Operacionalmente esto significa que cada bloque del flujo numerado de abajo (pasos 1-8) **no es un fin** sino un disparador de preguntas para el siguiente. Si el paso 3 muestra desbalance del target, el paso 4 deja de ser "clasificar todas las variables por igual" y se convierte en "qué variables segmentan mejor a la clase minoritaria".

### Las dos preguntas centrales

R4DS propone que toda pregunta de EDA cae en uno de dos tipos (§10.3):

| Pregunta | En qué buscar |
|---|---|
| **Variación** dentro de una variable | Cómo varía el valor entre observaciones de la misma variable. |
| **Covariación** entre variables | Cómo varían los valores de dos o más variables conjuntamente. |

Esto es el norte que guía los pasos 5-8 del flujo numerado: paso 5 (distribuciones univariadas) responde a *variación*, pasos 6-8 (outliers, categorías raras, correlación) son distintas formas de *covariación*.

### Heurística R4DS sobre valores inusuales

R4DS §10.4 distingue entre:
- **Outliers genuinos** (errores de medición, casos raros legítimos): investigar antes de actuar.
- **Valores faltantes disfrazados** (`-1`, `9999`, `0` como marcador): renombrar a `NaN` antes de modelar.

Antes de aplicar el clipping IQR del paso 6, pregúntate: *¿este outlier representa un error o un caso real importante?* Si es lo segundo, el clipping puede destruir señal.

### Tabla de equivalencias tidyverse → pandas/seaborn

R4DS escribe su flujo en R; tu código debe estar en Python. Esta tabla cubre los patrones más usados en EDA:

| R / tidyverse | Python / pandas + seaborn |
|---|---|
| `df %>% filter(x > 0)` | `df[df["x"] > 0]` |
| `df %>% mutate(z = x + y)` | `df.assign(z=df["x"] + df["y"])` |
| `df %>% group_by(g) %>% summarise(m = mean(x))` | `df.groupby("g")["x"].mean()` |
| `df %>% count(var)` | `df["var"].value_counts()` |
| `ggplot(df, aes(x)) + geom_histogram()` | `sns.histplot(data=df, x="x")` |
| `ggplot(df, aes(x, y)) + geom_point()` | `sns.scatterplot(data=df, x="x", y="y")` |
| `ggplot(df, aes(x, fill=cat)) + geom_bar()` | `sns.countplot(data=df, x="x", hue="cat")` |
| `ggplot(df, aes(cat, y)) + geom_boxplot()` | `sns.boxplot(data=df, x="cat", y="y")` |
| `geom_freqpoly(aes(color=cat))` | `sns.histplot(..., hue="cat", element="step", stat="density")` |
| `cor(df %>% select_if(is.numeric))` | `df.select_dtypes("number").corr()` |

### Cuándo consultar R4DS via el RAG durante EDA

Las consultas obligatorias se hacen al **cerrar** el EDA (sección anterior). Durante el EDA en sí, estas son consultas naturales si quieres anclar una decisión sobre la marcha:

| Situación | Query sugerida | Tool |
|---|---|---|
| Iniciar EDA y dudar qué mirar primero | `"exploratory data analysis variation covariation"` con `book="r4ds"` | `search_theory` |
| Decidir si un outlier es genuino o ruido | `"unusual values outliers investigate"` con `book="r4ds"` | `search_theory` |
| Identificar nulos disfrazados | `"missing values implicit explicit recode"` con `book="r4ds"` | `search_theory` |
| Elegir entre histograma / freqpoly / boxplot para covariación | `"histogram boxplot covariation continuous categorical"` con `book="r4ds"` | `search_theory` |
| Reagrupar categorías raras | `"factor lump small categories"` con `book="r4ds"` | `search_theory` |

> Cuando integres una cita de R4DS en un comentario o reporte: parafrasea la idea, atribuye a Wickham et al., y muestra el equivalente en pandas. Nunca pegues el código R como solución.

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

## Stack estadístico para EDA: pandas + seaborn + scipy.stats

El stack base del EDA en este power es **pandas + numpy + seaborn/matplotlib + `scipy.stats`**. Cubre el 90% de las necesidades:

- Distribuciones (`describe`, `value_counts`, `hist`, `kde`, `boxplot`).
- Asociación entre numéricas (`pearsonr`, `spearmanr`, `kendalltau`).
- Asociación entre categóricas (`chi2_contingency`).
- Comparaciones entre grupos (`ttest_ind`, `mannwhitneyu`, `ks_2samp`).
- Tests de normalidad (`shapiro`, `normaltest`).

`scipy.stats` viene como dependencia transitiva del stack, así que **no añade peso** al entorno. Úsalo libremente durante el EDA cuando necesites una prueba puntual; cita el resultado en `notes/00_design_eda.md` solo si dispara una decisión aguas abajo.

### Cuándo subir de `scipy.stats` a `statsmodels` (opt-in)

`statsmodels` es una dependencia adicional **opcional**, justificable solo cuando aporta algo que el stack base no cubre. Tres triggers concretos:

1. **Series de tiempo**: ADF / KPSS para estacionariedad, descomposición STL, ACF/PACF, ARIMA/SARIMAX. `scipy` no cubre esto y es el caso más claro de inclusión.
2. **VIF (multicolinealidad)** cuando el EDA detecta correlaciones > 0.85 entre varias features y la nota 00 quiere fundamentar la decisión "eliminar feature vs. usar Ridge/ElasticNet" antes de pasar a FE. La función relevante es `statsmodels.stats.outliers_influence.variance_inflation_factor`.
3. **Inferencia formal** cuando el entregable lo exige (informes regulados, académicos, salud): `smf.ols(...).fit().summary()` para tabla con coeficientes, errores estándar, p-values, IC, AIC/BIC, condition number. sklearn no entrega eso porque su foco es predicción, no inferencia.

Reglas operativas si entra `statsmodels`:

- Convive con `scikit-learn`, **no lo reemplaza**: sklearn lleva la predicción y el deploy a HF Spaces; `statsmodels` emite la inferencia como artefacto auxiliar dentro del EDA o de la nota 00.
- No uses `statsmodels` para hacer hypothesis testing sobre el dataset completo y luego seleccionar features con esos resultados: eso es leakage sutil. Las decisiones de FE/modelado se justifican vía RAG (ESL/ISLP), no vía p-values del train.
- Si la única razón es "quiero ver `summary()`", no añadas la dependencia: usa `scipy.stats.linregress` o `seaborn.regplot` con `ci=95`, son suficientes para EDA.
- Documenta la inclusión en `pyproject.toml` con justificación corta en el PR/commit (qué trigger de los tres dispara la dependencia).

### Tabla rápida: qué usar para qué

| Necesidad en EDA | Herramienta |
|---|---|
| Resumen estadístico básico | `df.describe()` |
| Conteos y frecuencias | `df["x"].value_counts(normalize=True)` |
| Correlación numérica | `df.corr()` o `scipy.stats.pearsonr` / `spearmanr` |
| Asociación categórica | `scipy.stats.chi2_contingency` |
| Comparar dos grupos numéricos | `scipy.stats.ttest_ind` / `mannwhitneyu` |
| Normalidad | `scipy.stats.shapiro` / `normaltest` |
| Regresión visual rápida | `sns.regplot(...)` |
| Multicolinealidad (VIF) | `statsmodels.stats.outliers_influence.variance_inflation_factor` (opt-in) |
| Estacionariedad / series de tiempo | `statsmodels.tsa.stattools.adfuller` / `kpss` (opt-in) |
| Tabla inferencial completa de OLS | `statsmodels.formula.api.ols(...).fit().summary()` (opt-in) |

---

## Gotchas Comunes

- **No eliminar outliers en EDA** — solo identificarlos. El tratamiento (clipping) va en Feature Engineering.
- **No imputar nulos en EDA** — solo cuantificar. La imputación va en Feature Engineering DESPUÉS del split.
- **No transformar variables en EDA** — solo describir. Las transformaciones van en Feature Engineering.
- **Variables categóricas codificadas como int** — son categóricas aunque pandas las infiera como numéricas.
- **No hacer hypothesis testing sobre todo el dataset para seleccionar features** — los p-values calculados sobre el conjunto completo y luego usados para filtrar features son una forma sutil de leakage. La selección de features se decide en FE con justificación teórica (RAG), no con tests del EDA.

## Consultar Documentación

Si necesitas verificar una API de pandas, seaborn u otra librería, usa la búsqueda web del agente (`remote_web_search` / `web_fetch`) apuntando a la doc oficial. Ejemplos de queries:

- `"seaborn boxplot orient horizontal"` → docs de seaborn.
- `"pandas value_counts normalize percentage"` → docs de pandas.

Para Gradio usa el server MCP `gradio` (no la web genérica).

## Siguiente paso

Una vez completado el EDA:

1. Cierra `notes/00_design_eda.md` (secciones 3-5: consultas al RAG sobre los hallazgos, tabla de implicaciones para FE, riesgos). La plantilla está en `theory-driven-design.md`.
2. Solo entonces inicia `notes/01_design_fe.md` y la fase de Feature Engineering (`workflow-feature-engineering.md`).

La nota 00 es el **puente formal** entre exploración y diseño: cada hallazgo accionable del EDA debe convertirse en una fila de la tabla "Implicaciones para el código" que disparará una decisión concreta en la nota 01.
