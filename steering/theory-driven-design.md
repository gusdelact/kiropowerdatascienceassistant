---
inclusion: always
---

# Theory-Driven Design en proyectos de Data Science

Esta convención aplica a **todo proyecto de DS / ML que entrene un modelo usando este power**. Define el protocolo obligatorio para usar el servidor MCP `rag-books-mcp` (ESL + ISLP) **antes** de escribir el código de feature engineering, modelado o validación.

> **Relación con `theory-rag-guide.md`**: aquel steering describe *cómo* invocar las tools del RAG (qué tool usar, formato de citas, modo degradado). Este steering describe *cuándo* y *con qué entregable* (los `notes/0N_design_*.md` como artefactos obligatorios antes de codificar). Los dos son complementarios: `theory-rag-guide.md` es el manual de la herramienta; este es el proceso del proyecto.

El objetivo es invertir el patrón anti-deseado:

> ❌ "Escribo el pipeline, luego consulto el RAG y decoro la Model Card con citas."
>
> ✅ "Consulto el RAG para diseñar el pipeline, dejo evidencia del diseño en `notes/`, luego escribo el código que materializa esas decisiones."

El RAG no es un decorador de justificaciones a posteriori. Es la **fuente de conocimiento que informa qué se va a codificar**.

---

## Regla 0: precedencia y composición

Esta regla **no compite** con `theory-rag-guide.md`. Lo extiende:

- `theory-rag-guide.md` define qué tools existen, cómo se llaman y cómo se citan los fragmentos. Sigue siendo la referencia operativa de la herramienta.
- Este steering define que el resultado de esas consultas debe **materializarse en un documento de diseño en `notes/`** antes de escribir el código de la fase correspondiente.

Donde haya conflicto aparente entre ambos (por ejemplo, la tabla de queries por línea de código en `theory-rag-guide.md` invita a consultar el RAG después de elegir la línea), gana este steering: la consulta va **antes** de la decisión, y queda registrada en el `notes/*.md`.

---

## Regla 1: el pipeline tiene cuatro fases de diseño obligatorias

El proyecto debe producir cuatro documentos en `notes/`, **en este orden**:

0. `notes/00_design_eda.md` — diseño y hallazgos del análisis exploratorio.
1. `notes/01_design_fe.md` — diseño del feature engineering.
2. `notes/02_design_modeling.md` — diseño del modelado.
3. `notes/03_design_validation.md` — diseño de la validación.

Cada uno se redacta consultando el `rag-books-mcp` y citando las secciones específicas. **No se permite escribir código de la fase N si el `notes/0N_*.md` correspondiente no existe**, salvo en los casos de la Regla 6 (excepciones).

**Timing distinto para `notes/00_design_eda.md`**: a diferencia de las notas 01-03 que se producen *antes* del código de su fase, la nota 00 se produce **como parte de la fase de EDA**. Tiene una estructura mínima pre-EDA (preguntas guía e hipótesis) y se **completa al finalizar el EDA** con los hallazgos y el puente hacia FE. Bloquea la transición a `notes/01_design_fe.md`, no la ejecución de `02_eda.py`.

El orden de las fases coincide con el orden del flujo del power. La diferencia es que cada transición fase → código (o EDA → diseño de FE) pasa por el documento de diseño.

---

## Regla 2: estructura mínima de cada `notes/0N_design_*.md`

Cada documento de diseño tiene cinco secciones obligatorias:

```markdown
# Diseño de <fase>: <proyecto>

## 1. Contexto del problema
- Tipo de problema (regresión / clasificación binaria / multiclase / clustering / etc.).
- Tamaño del dataset (n filas, p features), tipos de variables.
- Métrica de éxito esperada por el usuario.
- Restricciones (cómputo, interpretabilidad, latencia).

## 2. Decisiones a tomar en esta fase
Lista numerada de las decisiones concretas que afectan el código.
Ej. (FE): "¿Aplicar log1p al target?", "¿Split estratificado por bins?", "¿Qué encoding para categóricas?".
Ej. (Modelado): "¿Familia de modelo?", "¿Regularización?", "¿Hiperparámetros sensibles a teoría?".
Ej. (Validación): "¿Métrica primaria?", "¿K en CV?", "¿Benchmarking contra qué baselines?".

## 3. Consultas al RAG
Para cada decisión de la sección 2, una subsección con:

### Decisión <i>: <título corto>
- **Query**: literal del query enviado a `search_theory` / `cite_foundation` / `get_section`.
- **Fragmentos relevantes**: 1–3 paráfrasis breves (≤30 palabras cada una) con la sección citada `[ESL §X.Y]` o `[ISLP §X.Y]`.
- **Decisión tomada**: una oración.
- **Alternativa descartada y por qué**: una oración.

## 4. Implicaciones para el código
Tabla mínima:

| Decisión | Línea / función que la materializa | Archivo |
|---|---|---|
| `log1p` al target porque skew=2.3 | `np.log1p(y_train)` y back-transform en predict | `03_feature_engineering.py`, `predict.py` |
| ... | ... | ... |

## 5. Riesgos identificados
Lista breve de riesgos que la teoría anticipa (extrapolación, leakage, sensibilidad a outliers, etc.) y cómo los mitigamos en el código.
```

Sin las cinco secciones, el documento no cumple. Si una sección es N/A, justificar en una línea por qué.

---

## Regla 3: queries mínimas por fase

Cada `notes/0N_design_*.md` debe contener **al menos** las siguientes consultas al RAG (más las que el problema requiera):

### Fase 0: EDA (`notes/00_design_eda.md`)

EDA tiene un régimen distinto a las demás fases: las consultas al RAG **no son obligatorias antes** de codificar `02_eda.py` (porque EDA es exploratorio por naturaleza), pero **sí lo son al cerrarlo**, para anclar los hallazgos que dispararán las decisiones de FE. Las consultas mínimas al cerrar el EDA son:

- Disciplina del flujo iterativo (preferir R4DS Cap. 10): qué tipos de pregunta hacer (variación / covariación), cuándo iterar.
- Lectura de la distribución del target (skew, multimodalidad, ceros estructurales) y qué significa para FE/modelado.
- Tratamiento de outliers desde la perspectiva de descubrimiento vs. error (R4DS §10.4 + FES Cap. 6).
- Diagnóstico de multicolinealidad y su impacto en familias de modelo candidatas (ESL §3.4).
- Patrones de missing data observados (MCAR / MAR / MNAR) y consecuencias (FES Cap. 8).

Si el dataset no presenta una de estas situaciones (ej. sin desbalance, sin missings, sin outliers), declarar "no aplica" en el documento y no consultar por compromiso.

### Fase 1: feature engineering (`notes/01_design_fe.md`)

- Distribución del target y transformaciones (`log1p`, Box-Cox, Yeo-Johnson).
- Estrategia de split: estratificado por target en regresión (bins) o por clase en clasificación.
- Manejo de outliers (IQR clipping, winsorización, dejarlos).
- Escalado: cuándo es obligatorio vs opcional según familia de modelo *candidata*.
- Encoding de categóricas según cardinalidad y familia de modelo candidata.
- Manejo de desbalance (solo si es clasificación): SMOTE / `class_weight` / `scale_pos_weight`, siempre post-split.

### Fase 2: modelado (`notes/02_design_modeling.md`)

- Elección de familia de modelo (lineal / árboles / SVM / ensemble / DL) según n, p, linealidad esperada, interpretabilidad.
- Regularización (Ridge / Lasso / ElasticNet, `alpha`, `C`, `reg_alpha`, `reg_lambda`).
- Hiperparámetros sensibles a teoría (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `max_features`).
- Justificación del baseline (qué se compara contra qué y por qué).

### Fase 3: validación (`notes/03_design_validation.md`)

- Métrica primaria y métricas secundarias (por qué F1 / PR-AUC / RMSE / MAPE / R²).
- K en cross-validation y estrategia (KFold, StratifiedKFold, GroupKFold).
- Calibración del umbral (solo si es clasificación con desbalance).
- Benchmarking: contra qué modelos y bajo qué condiciones se considera "empate".

Si una decisión de la lista no aplica al proyecto (ej. desbalance en regresión), declararlo explícitamente en el documento y *no* consultar el RAG por compromiso. La regla es "consultar lo que aplica", no "consultar todo".

### Cómo combinar libros en cada query

El RAG indexa cinco libros. Cuando consultes para una decisión de FE/modelado/validación, **diversifica las fuentes** según el tipo de decisión:

| Decisión | Libro principal | Libro de apoyo |
|---|---|---|
| Por qué algo funciona (teoría) | ESL, ISLP | — |
| Cómo implementarlo en Python | PDSH | ISLP (labs) |
| Heurísticas de feature engineering | FES | ESL §3 |
| Filosofía / disciplina del flujo iterativo (sobre todo en EDA) | **R4DS** | FES |
| Re-agrupación de categorías raras | R4DS §15 (factors) + FES §5 | — |

**Regla específica para R4DS** (CC BY-NC-ND, ejemplos en R/tidyverse):

1. Cita R4DS para la **idea o disciplina del proceso**, nunca para el código.
2. Atribuye explícitamente: `[R4DS Cap. X — Wickham, Çetinkaya-Rundel, Grolemund]`.
3. Si parafraseas un patrón tidyverse, muestra el equivalente en pandas/seaborn en la misma celda. Tabla de equivalencias en `theory-rag-guide.md` §"Importante sobre R4DS" y `workflow-eda.md` §"Filosofía R4DS para EDA".
4. **NUNCA dejes código en R en `scripts/`, `notes/`, ni en la respuesta al usuario.** Si un fragmento del libro es útil pero está en R, parafrasea la idea y traduce.

---

## Regla 4: prohibición explícita de código sin diseño

Antes de escribir el primer `import sklearn` (o equivalente) de cada fase, el agente debe:

1. Verificar que el `notes/0N_design_*.md` correspondiente existe y cumple la Regla 2.
2. Si no existe, **detener la generación de código** y producir el documento de diseño primero.
3. Si existe pero está incompleto, completarlo antes de codificar.

Líneas de código que disparan esta verificación:

| Fase | Disparadores |
|---|---|
| EDA | Iniciar `02_eda.py` con preguntas guía → crear `notes/00_design_eda.md` (estructura pre-EDA). Cerrar EDA antes de pasar a FE → completar `notes/00_design_eda.md` (hallazgos + puente). |
| FE | `train_test_split`, `StandardScaler`, `OneHotEncoder`, `SMOTE`, `np.log1p` sobre target, `pd.qcut` para bins de stratify |
| Modelado | Cualquier `from sklearn.linear_model`, `from sklearn.ensemble`, `from sklearn.svm`, `from xgboost`, `import torch`, etc. |
| Validación | `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`, `precision_recall_curve`, `mean_squared_error` con intención de reportar |

Excepción: el script de **ingesta** (`01_ingest.py`) no requiere diseño previo. El script de **EDA** (`02_eda.py`) puede ejecutarse con la nota 00 en estado pre-EDA (preguntas e hipótesis); la nota 00 debe quedar **completa** antes de iniciar `notes/01_design_fe.md`.

---

## Regla 5: integración con el código y los artefactos

Cada documento de diseño debe **dejar rastro** en el código que produce:

- Los scripts de pipeline (`03_feature_engineering.py`, `04_train.py`, `05_validate.py`) deben empezar con un docstring que referencie el documento de diseño que les corresponde:

  ```python
  """Feature engineering pipeline.

  Diseño: notes/01_design_fe.md
  """
  ```

- La Model Card publicada en HF Hub debe incluir una sección "Fundamento Teórico" que **resuma** los cuatro documentos de diseño (00 EDA, 01 FE, 02 modelado, 03 validación) con sus citas. Las citas en la Model Card son consecuencia del diseño, no su origen.

- Los `notes/*.md` se versionan junto con el código (no van a `.gitignore`). Son parte de la entrega del proyecto.

---

## Regla 6: excepciones (cuándo NO se exige diseño previo)

No es necesario producir un `notes/*.md` cuando:

- El proyecto es un **fix puntual** sobre un pipeline ya entrenado (corregir un bug en `predict.py`, ajustar la UI de Gradio, regenerar una figura).
- El cambio es **operativo** (rename, refactor, mover un archivo, actualizar dependencias).
- El usuario pide explícitamente un **baseline rápido con defaults razonables** ("dame un script de 30 líneas para empezar"). En ese caso, el documento de diseño se difiere y se produce *antes* del modelo final.
- El `rag-books-mcp` no responde (modo degradado de la Regla 7).

En las dos primeras excepciones, ningún diseño se requiere. En la tercera, el diseño se produce antes de la siguiente iteración seria.

---

## Regla 7: modo degradado (RAG no disponible)

Si las tools `search_theory` / `cite_foundation` / `get_section` no responden:

1. **Declarar al usuario** explícitamente que `rag-books-mcp` no está activo.
2. Producir el `notes/0N_design_*.md` igualmente, con una sección "⚠️ Sin verificación bibliográfica" en la cabecera.
3. Las decisiones se justifican con conocimiento general y se etiquetan como tales.
4. **No inventar números de sección de ESL/ISLP** bajo ninguna circunstancia.
5. Sugerir al usuario activar el servidor (ver `mcp-servers/rag-books-mcp/README.md`) y revisitar el diseño cuando esté disponible.

---

## Regla 8: tamaño y forma de las consultas

Para evitar tanto la pobreza (consultar 1 vez para todo) como el exceso (consultar 30 veces decisiones triviales):

- **Mínimo 3 consultas** por documento de diseño.
- **Máximo razonable**: 1 consulta por decisión real de la sección 2. Si hay 8 decisiones, son 8 consultas.
- Las consultas deben preferir `cite_foundation` cuando se quiere fundamentación lista para integrar; `search_theory` para exploración; `get_section` cuando ya se tiene la referencia exacta.
- Nunca pegar bloques largos del libro. Las paráfrasis son ≤30 palabras por fragmento (compliance con la licencia y mejor para el lector).

---

## Plantilla mínima de `notes/00_design_eda.md`

EDA usa una plantilla **bipartita**: una parte se llena *al inicio* (preguntas e hipótesis), la otra *al cierre* (hallazgos y puente a FE). Las cinco secciones de la Regla 2 siguen vigentes pero adaptadas al carácter exploratorio.

```markdown
# Diseño y hallazgos del EDA: <proyecto>

> **Estado**: [ ] pre-EDA · [ ] EDA en curso · [ ] cerrado (puente a FE listo)

## 1. Contexto del problema
- Problema: <regresión|clasificación binaria|...>.
- Dataset: <n> filas × <p> features. Fuente y fecha de captura.
- Target: <nombre>. ¿Qué representa? ¿Cómo se midió?
- Métrica de éxito esperada por el usuario.
- Restricciones (cómputo, interpretabilidad, latencia).

## 2. Preguntas guía e hipótesis (PRE-EDA)
Llenar antes de codificar `02_eda.py`. Una pregunta por punto, formato R4DS:
- **Variación**: ¿Cómo se distribuye el target? ¿Hay sesgo, multimodalidad, ceros estructurales?
- **Variación**: ¿Qué predictores parecen tener varianza casi nula?
- **Covariación**: ¿Qué predictores correlacionan fuerte con el target? ¿Entre sí?
- **Covariación**: ¿Hay interacciones esperadas por dominio?
- **Calidad**: ¿Hay nulos disfrazados (-1, 9999, 0)? ¿Patrón de missingness?
- **Outliers**: ¿Qué valores extremos podrían ser errores vs. casos genuinos?
- Hipótesis del dominio (1-3): "espero que X tenga relación monotónica con Y porque ...".

## 3. Consultas al RAG (al cerrar el EDA)
Anclar los hallazgos relevantes para FE. Una subsección por hallazgo accionable:

### Hallazgo 1: <título corto>
- **Observación**: <qué se vio en el EDA, con número/figura>.
- **Query**: literal del query enviado a `search_theory` / `cite_foundation` / `get_section`.
- **Fragmentos**: 1-2 paráfrasis breves (≤30 palabras) con cita `[ESL §X.Y]` / `[ISLP §X.Y]` / `[FES Cap. X]` / `[R4DS §X.Y]`.
- **Implicación para FE/modelado**: una oración accionable.

### Hallazgo 2: ...

## 4. Implicaciones para el código (puente a FE)
Tabla de hallazgos → decisiones que entrarán en `notes/01_design_fe.md`:

| Hallazgo | Decisión que dispara en FE | Sección de `01_design_fe.md` |
|---|---|---|
| Skew del target = 2.3 | Evaluar `log1p(y)` | Decisión 1 |
| Multicolinealidad `X1`-`X2` > 0.9 | Eliminar `X2` o usar Ridge | Decisión 5 + Decisión modelado |
| 8% de missings en `Z` no aleatorios | Imputar con flag `Z_was_missing` | Decisión 7 |
| Categorías con < 1% en `cat_var` | Lump a "Otro" | Decisión 4 |

## 5. Riesgos identificados
- Sesgo de selección sospechado (¿cómo se muestreó el dataset?).
- Leakage potencial (variables que solo existen post-evento).
- Drift temporal si hay variable de fecha.
- Outliers cuya eliminación podría destruir señal.

## Artefactos producidos
- `outputs/figures/distributions.png`
- `outputs/figures/outliers_boxplots.png`
- `outputs/figures/correlation_matrix.png`
- `outputs/reports/eda_report.json`
```

---

## Plantilla mínima de `notes/01_design_fe.md`

```markdown
# Diseño de Feature Engineering: <proyecto>

## 1. Contexto del problema
- Problema: <regresión|clasificación binaria|...>
- Dataset: <n> filas × <p> features. Numéricas: <k>, categóricas: <m>.
- Target: <nombre>, distribución observada en EDA: skew=<x>, rango=[<min>, <max>].
- Métrica esperada: <RMSE|F1|...>.

## 2. Decisiones a tomar
1. ¿Transformar el target?
2. ¿Estrategia de split?
3. ¿Manejo de outliers?
4. ¿Escalado?
5. ¿Encoding de categóricas?
6. ¿Manejo de desbalance? (solo clasificación)

## 3. Consultas al RAG
### Decisión 1: transformación del target
- **Query**: `cite_foundation(topic="log transformation skewed regression target")`
- **Fragmentos**:
  - <paráfrasis ≤30 palabras> [ISLP §<X.Y>]
- **Decisión**: <una oración>.
- **Alternativa descartada**: <una oración>.

### Decisión 2: ...
...

## 4. Implicaciones para el código
| Decisión | Línea / función | Archivo |
|---|---|---|
| ... | ... | ... |

## 5. Riesgos identificados
- ...
```

---

## Qué NO hacer

- Generar el `notes/*.md` *después* de escribir el código y rellenarlo con citas para que parezca diseñado.
- Citar ESL/ISLP sin haber consultado el RAG en esa sesión.
- Tratar el RAG como un decorador de la Model Card. Las citas en Model Card son consecuencia del diseño, no su origen.
- Saltarse el `notes/*.md` "porque ya lo hice en otro proyecto similar". Cada proyecto tiene su propio contexto y su propio documento.
- Consultar el RAG una sola vez al inicio y reutilizar las citas para todas las fases.

## Qué sí hacer

- Antes de cada `from sklearn... import` significativo, comprobar mentalmente: "¿en qué `notes/*.md` está documentada esta decisión?".
- Si el usuario te corrige sobre el uso del RAG, regenerar el `notes/*.md` correspondiente con consultas reales y dejar el documento como evidencia del rediseño.
- Tratar los `notes/*.md` como entregables de la misma calidad que el código: revisables, versionados, citables.
