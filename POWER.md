---
name: "data-science-assistant"
displayName: "Data Science Assistant"
description: "Guía completa de heurísticas, teoría y mejores prácticas para análisis de datos, modelado predictivo y validación de modelos de Machine Learning. Cubre el flujo completo desde EDA hasta evaluación de modelos ensemble, con fundamentos teóricos de ESL, ISLP, FES, PDSH y R4DS."
keywords: ["data science", "machine learning", "EDA", "feature engineering", "regresion lineal", "ridge", "lasso", "elastic net", "arboles", "random forest", "extra trees", "ensemble", "bagging", "boosting", "xgboost", "svm", "support vector machine", "clasificación", "regresion", "modelado", "validación", "preprocesamiento", "SMOTE", "hiperparámetros", "bias-variance", "regularization", "cross-validation", "PCA", "clustering", "deep learning", "neural networks"]
author: "Data Science Assistant Contributors"
---

# Data Science Assistant

## 🚦 Comportamiento Inicial (OBLIGATORIO)

> **Cuando el usuario activa este Power sin dar instrucciones específicas, NO ejecutar ningún ejemplo ni pipeline automáticamente.**
>
> En su lugar, el agente DEBE:
> 1. Saludar brevemente y explicar qué puede hacer este Power (1-2 oraciones).
> 2. Preguntar al usuario: **¿Qué proyecto o análisis de datos quieres trabajar?**
> 3. Una vez que el usuario indique el dataset (`usuario/nombre-dataset` en HF Hub), el agente DEBE: (a) identificar y confirmar el **régimen del problema** (regresión / clasificación binaria / clasificación multiclase) en una sola frase, y (b) aplicar el **cuestionario corto de contexto de negocio** definido en `business-context.md` adaptado al régimen. Las 5 preguntas son obligatorias: 1) problema, 2) decisión que dispara la predicción; en clasificación 3) costo FP, 4) costo FN; en regresión 3) costo de sobre-estimar, 4) costo de sub-estimar; 5) métrica primaria + umbral mínimo de utilidad. El resultado se materializa en `notes/00_business_context.md`. Este paso es **bloqueante**: sin él no se ejecuta ingesta, EDA, FE, modelado ni validación.
> 4. Esperar la respuesta del usuario antes de tomar cualquier acción.
>
> **Nunca asumir un dataset, un problema ni un modelo por defecto.** El usuario siempre decide qué datos usar y qué tipo de análisis realizar. Tampoco se asume el contexto de negocio: las cinco preguntas de `business-context.md` son obligatorias salvo en las excepciones de la Regla 5 de ese steering (datasets pedagógicos canónicos o smoke tests internos, declarados explícitamente).
>
> Ejemplo de primera respuesta correcta:
> > "Soy tu asistente de Data Science. Puedo guiarte desde la exploración de datos hasta el despliegue de modelos ML. ¿Qué proyecto quieres trabajar? Si tienes un dataset en Hugging Face Hub, compárteme la referencia (formato `usuario/nombre-dataset`). Cuando me lo des, te haré cinco preguntas cortas de contexto de negocio antes de tocar los datos, para elegir bien la métrica y el costo de los errores."

---

## ⚠️ Limitaciones de esta Versión

> **Esta versión del Power SOLO acepta Hugging Face como fuente de datasets.**
>
> - La única fuente válida para ingesta de datos es **Hugging Face Hub** (formato `usuario/nombre-dataset`).
> - NO se soportan otras fuentes: Kaggle, URLs directas, archivos locales arbitrarios, APIs externas, bases de datos, etc.
> - Si el usuario solicita un dataset de otra fuente, se debe indicar que no está soportado y pedir que busque un equivalente en Hugging Face Hub.
> - Para buscar datasets disponibles: `hf datasets ls --search "tema"`

## Overview

Este power proporciona una guía estructurada y heurísticas probadas para ejecutar proyectos de ciencia de datos y machine learning de principio a fin. Cada steering file es autocontenido con ejemplos de código, heurísticas y conexiones con la teoría.

El flujo recomendado sigue cinco fases:
0. **Contexto de Negocio** — cuestionario corto + `notes/00_business_context.md` (BLOQUEANTE)
1. **EDA** — Análisis exploratorio
2. **Feature Engineering** — Transformaciones, split, scaling, encoding
3. **Modelado** — Entrenamiento por familia de modelo
4. **Validación** — Métricas, calibración, comparación

## Available Steering Files

### Workflow (cómo ejecutar cada fase)

- **business-context.md** — Paso 0 obligatorio. Cuestionario corto de 5 preguntas (problema, decisión que dispara la predicción, costo de sobre-predecir, costo de sub-predecir, métrica + umbral mínimo) que se materializa en `notes/00_business_context.md`. Es bloqueante: sin él no se hace EDA, FE, modelado ni validación. Define la métrica primaria y el costo asimétrico de los errores que el resto del pipeline debe respetar.
- **workflow-eda.md** — Análisis exploratorio: duplicados, nulos, distribuciones, outliers, correlación. Heurística del umbral 13 para clasificar variables.
- **workflow-feature-engineering.md** — IQR clipping, eliminación de categóricas raras, split estratificado, imputación, estandarización, encoding, `ColumnTransformer`.
- **workflow-model-training.md** — Flujo común a todos los modelos: SMOTE, GridSearch/RandomizedSearch, cross-validation, serialización.
- **workflow-validation.md** — Métricas (clasificación y regresión), confusion matrix, ROC, calibración del umbral, residuales, benchmarking, feature importance.

### Modelos por Familia (heurísticas específicas)

- **models-linear-regression.md** — OLS, Ridge, Lasso, ElasticNet. Diagnóstico de colinealidad.
- **models-trees-rf.md** — Decision Trees, Random Forest (regresión), ExtraTrees, OOB error, feature importance.
- **models-ensemble.md** — Bagging y Boosting para clasificación con desbalance. RF Classifier + XGBoost + SMOTE + calibración de umbral + Stacking.
- **models-svm.md** — `LinearSVC`, `SVC` con kernels (linear/rbf/poly), `SVR`, `OneClassSVM`.

### Teoría (por qué funciona)

- **theory-rag-guide.md** — Manual operativo del servidor MCP `rag-books-mcp` (ESL + ISLP + FES + PDSH + R4DS). Define las 4 tools (`search_theory`, `cite_foundation`, `get_section`, `list_available_topics`), el formato de citas, los triggers por línea de código y el modo degradado. Aclara que R4DS está en R y solo se usa por sus principios. Es la referencia de **cómo** consultar la teoría.
- **theory-driven-design.md** — Protocolo obligatorio de uso del RAG **antes** de codificar. Exige producir cuatro documentos de diseño (`notes/01_design_eda.md`, `notes/02_design_fe.md`, `notes/03_design_modeling.md`, `notes/04_design_validation.md`) con consultas reales al RAG, decisiones tomadas y alternativas descartadas. La nota 01 (EDA) tiene timing bipartito (pre-EDA y al cierre); las notas 02-04 se producen antes del código de su fase. Es la referencia de **cuándo** y **con qué entregable**.

### Infraestructura

- **gradio-interfaces.md** — Dashboard de resultados (EDA + validación) y app de inferencia para HF Spaces.
- **huggingface-workflows.md** — Ingesta y publicación de datasets/modelos en HF Hub.
- **mlops-deployment.md** — Pipelines MLOps con `uv` y despliegue a HF Spaces.
- **notebooks-ds.md** — Convención de notebooks autocontenidos para Google Colab: badge "Open in Colab", instalación con `uv` sin tocar paquetes preinstalados, separación entrenamiento / inferencia, `model_info.json` y publicación en HF Hub.

---

## ¿Qué Tipo de Problema Tengo?

Tabla rápida para elegir la familia de modelo según el problema:

| Tipo de problema | Familia | Steering |
|---|---|---|
| Regresión lineal, multicolinealidad, regularización | Lineal + L1/L2 | `models-linear-regression.md` |
| Predicción con relaciones no lineales, benchmarking | Random Forest / ExtraTrees | `models-trees-rf.md` |
| Clasificación binaria con desbalance | Ensemble (Bagging + Boosting) | `models-ensemble.md` |
| Fronteras complejas no lineales, datasets pequeños/medianos | SVM | `models-svm.md` |

---

## Flujo de Trabajo Recomendado

```
Datos crudos (SOLO desde Hugging Face Hub)
    │
    ▼
0. Contexto de Negocio (business-context.md) ⚠️ BLOQUEANTE
   ⚠️ Antes de ingesta, EDA o cualquier código: producir notes/00_business_context.md
      con las 5 preguntas obligatorias (adaptadas al régimen del problema).
      Sin este documento NO se ejecuta ningún script.
   • Paso A: identificar y confirmar régimen (regresión / clasificación binaria / multiclase)
   • Pregunta 1: ¿Qué problema de negocio resolvemos?
   • Pregunta 2: ¿Qué decisión o acción dispara la predicción?
   • Pregunta 3:
       - Clasificación: costo de un FP (falso positivo)
       - Regresión:     costo de sobre-estimar el target (por unidad)
   • Pregunta 4:
       - Clasificación: costo de un FN (falso negativo) → ratio costo_FN/costo_FP
       - Regresión:     costo de sub-estimar el target → ratio costo_sub/costo_sobre
                        (puede ser "simétrico")
   • Pregunta 5: Métrica primaria + umbral mínimo de utilidad
       - Clasificación: recall / precision / F1 / PR-AUC / balanced_accuracy + valor mínimo
       - Regresión:     MAE / RMSE / MAPE / RMSLE / pinball_loss + valor máximo
   Excepción declarada: datasets pedagógicos canónicos (Iris, Titanic, Penguins, etc.)
   pueden usar versión corta (preguntas 1 + 5), declarándolo en el documento.
    │
    ▼
1. EDA (workflow-eda.md)
   ℹ️ EDA produce notes/01_design_eda.md (timing bipartito: pre-EDA + al cierre).
      No bloquea la ejecución de 02_eda.py, pero sí la transición a FE.
      Sus hallazgos (skew, multicolinealidad, desbalance, categorías raras) son los
      INSUMOS que dispararán las queries obligatorias al RAG en la fase 2.
      SÍ requiere `notes/00_business_context.md` ya producido (paso 0): el contexto
      define qué distribuciones y segmentos priorizar.
   • Duplicados, nulos, distribuciones
   • Clasificar variables (numéricas vs categóricas, umbral 13)
   • Identificar outliers y categorías raras
    │
    ▼
2. Feature Engineering (workflow-feature-engineering.md)
   ⚠️ Antes de codificar: producir notes/02_design_fe.md (theory-driven-design.md)
      con al menos las 6 queries al rag-books-mcp que exige la Regla 3
      (target transform, split, outliers, escalado, encoding, desbalance).
   • Eliminar registros con categorías raras
   • Clipping IQR de outliers
   • Eliminar features correlacionadas
   • Split train/test ESTRATIFICADO
   • Imputación + Estandarización + Encoding (fit en train)
    │
    ▼
3. Decisión de modelo (⚠️ OBLIGATORIO: preguntar al usuario)
   ⚠️ Antes de codificar: producir notes/03_design_modeling.md (theory-driven-design.md)
   • Presentar al usuario la tabla de familias disponibles
   • Ofrecer recomendación basada en el tipo de problema
   • ESPERAR confirmación explícita del usuario antes de continuar
   • ¿Lineal? → models-linear-regression.md
   • ¿No lineal? → models-trees-rf.md
   • ¿Mejor rendimiento? → models-ensemble.md
   • ¿Datos pequeños/medianos? → models-svm.md
    │
    ▼
4. Entrenamiento (workflow-model-training.md)
   • SMOTE post-split (si desbalance)
   • RandomizedSearchCV o GridSearchCV con CV=5
   • Reentrenar con mejores hiperparámetros
    │
    ▼
5. Validación (workflow-validation.md)
   ⚠️ Antes de codificar: producir notes/04_design_validation.md (theory-driven-design.md)
   • Métricas apropiadas al problema
   • Confusion matrix, ROC, residuales
   • Calibración del umbral (clasificación con desbalance)
   • Feature importance
   • Comparación entre modelos
    │
    ▼
6. Publicar Dataset Curado (huggingface-workflows.md §Etapa 6) ⚠️ NO OMITIR
   • Generar DATA_CARD.md con metadatos del dataset procesado
   • Subir data/processed/ a HF Hub como dataset independiente
   • El slug DEBE ser descriptivo (ver Principio #4)
   • Script: 06_publish_dataset.py
    │
    ▼
7. Publicar Modelo (huggingface-workflows.md §Etapa 7)
   • Generar MODEL_CARD.md con métricas y metadata
   • Subir model.joblib + preprocessor.joblib a HF Hub
   • Script: 07_publish_model.py
    │
    ▼
8. Desplegar App de Inferencia (mlops-deployment.md)
   ⚠️ Antes de generar CUALQUIER app Gradio: leer `gradio-interfaces.md` y `mlops-deployment.md`
   ⚠️ Antes de invocar `08_deploy_hf.py`: ejecutar el "Checklist antes de 08_deploy_hf.py"
      de `gradio-interfaces.md` (puerto 7860 condicional, huggingface-hub>=0.28.1,
      pip dry-run sin error). Estos dos errores se han repetido en producción y
      consumen tiempo de cola del Space.
   • Crear HF Space con app Gradio de inferencia
   • La app descarga el modelo desde HF Hub al iniciar
   • Puerto OBLIGATORIO en HF Spaces: 7860 con server_name="0.0.0.0"
   • Script: 08_deploy_hf.py
    │
    ▼
9. Generar Notebooks para Google Colab (notebooks-ds.md)
   ⚠️ Antes de generar los notebooks: leer `notebooks-ds.md` (Reglas 6, 8, 9 y 10).
   • DOS notebooks separados: `notebooks/01_entrenamiento.ipynb` y `notebooks/02_inferencia.ipynb`
   • AUTOCONTENIDOS: NO importar de `lib/` ni leer `config.yaml`, `data/` o `models/` locales
   • Destino primario: Google Colab (CPU, sin secretos). Detectar `IN_COLAB` para PROJECT_ROOT
   • Setup de deps con `uv pip install --system` SOLO para lo que falte (huggingface-hub).
     NUNCA `--upgrade` sobre numpy/pandas/sklearn/matplotlib (rompe el kernel de Colab)
   • Primera celda markdown con badge "Open In Colab" apuntando al `.ipynb` en HF Hub
   • Sanity check local antes de subir; subir SOLO los `.ipynb` (no recursivo)
   • Script: scripts/build_notebooks.py
```

> ⚠️ **IMPORTANTE**: Las etapas 6, 7, 8 y 9 son OBLIGATORIAS para completar el proyecto. El pipeline NO está completo si solo se llega a validación. Después de validar, SIEMPRE continuar con la publicación del dataset curado (etapa 6), el modelo (etapa 7), el despliegue (etapa 8) y los notebooks de Colab (etapa 9).

### Dashboard de Resultados (SE GENERA SIEMPRE)

La app `app_results/app_results.py` es un dashboard local de Gradio que visualiza los resultados de EDA y validación. **Se genera siempre como parte del proyecto.**

⚠️ **Antes de generar esta app: leer `gradio-interfaces.md`** — contiene reglas obligatorias de visualización (NO usar `gr.Gallery`, usar `gr.Image` individual por figura con altura fija).

- Lee figuras de `outputs/figures/` (prefijo `eda_*` y `val_*`)
- Lee métricas de `outputs/metrics/` y reportes de `outputs/reports/`
- No ejecuta nada, solo visualiza

**Al finalizar la etapa 5 (validación), el agente DEBE indicar al usuario:**
> "✅ Validación completada. Para visualizar interactivamente los resultados de EDA y validación, ejecuta: `uv run python app_results/app_results.py`"

## Principios Universales

0. **Business-Context First (BLOQUEANTE)** — Antes de cualquier código (incluida la ingesta), identificar el **régimen del problema** (regresión / clasificación) y producir `notes/00_business_context.md` siguiendo `business-context.md` con las cinco preguntas adaptadas al régimen. En clasificación: problema, decisión, costo FP, costo FN, métrica + umbral. En regresión: problema, decisión, costo de sobre-estimar, costo de sub-estimar, métrica (MAE/RMSE/MAPE/RMSLE/pinball) + umbral máximo aceptable. El resto del pipeline hereda de aquí la métrica primaria, el `scoring` del search de hiperparámetros, y el costo asimétrico de errores. Sin este documento, las decisiones del agente quedan a la imaginación y producen métricas equivocadas. Excepción explícita: datasets pedagógicos canónicos (Iris, Titanic, Penguins, etc.) pueden usar la versión corta (preguntas 1 + 5), declarándolo en el documento.
1. **No actuar sin instrucciones del usuario** — NUNCA ejecutar un pipeline, generar código ni elegir un dataset por cuenta propia. Siempre esperar a que el usuario indique explícitamente qué quiere hacer. Si el usuario solo dice "hola" o activa el Power sin contexto, preguntar qué proyecto quiere trabajar.
2. **Hugging Face es la ÚNICA fuente de datasets aceptada** — Esta versión del Power solo soporta ingesta desde Hugging Face Hub. Si el usuario menciona Kaggle, URLs, archivos locales u otra fuente, informar que no está soportado y ayudar a buscar un dataset equivalente en HF Hub. El formato requerido es `usuario/nombre-dataset`.
3. **Preguntar obligatoriamente al usuario qué modelo de ML utilizar** — Antes de entrenar, SIEMPRE preguntar al usuario qué modelo (o modelos) desea utilizar. Nunca asumir ni elegir el modelo automáticamente. Presentar la tabla de familias disponibles y esperar confirmación explícita del usuario. Si el usuario no sabe, ofrecer recomendaciones basadas en el tipo de problema pero dejar la decisión final al usuario.
4. **Nombres descriptivos, NUNCA genéricos** — Los slugs de datasets, modelos y Spaces DEBEN derivarse del nombre real del proyecto/dataset del usuario. NUNCA usar "dummy", "test", "example", "mi-proyecto" ni placeholders genéricos. Ejemplo correcto: `iris-curated`, `salary-predictor-lr`. Ver template en `huggingface-workflows.md`.
5. **Nunca aplicar SMOTE antes del split** — causa data leakage. SMOTE solo en train.
6. **El accuracy no es suficiente con clases desbalanceadas** — usar F1, precision, recall, PR-AUC.
7. **Validación cruzada siempre** — mínimo 5 folds.
8. **Estandarizar después del split** — fit en train, transform en test. Nunca fit en todo el dataset.
9. **Umbral de clasificación ≠ 0.5** — calibrar con la curva precision-recall cuando hay desbalance.
10. **Reproducibilidad** — fijar `random_state` en split, modelo, CV y SMOTE.
11. **Estandarización obligatoria para SVM, Lineal con regularización, KNN, Redes Neuronales** — opcional para árboles.
12. **Theory-Driven Design** — Antes de escribir código de FE, modelado o validación, producir el `notes/0N_design_*.md` correspondiente con consultas reales al `rag-books-mcp`. El RAG informa el diseño, no decora la Model Card a posteriori. Ver `theory-driven-design.md` para el protocolo completo y excepciones.

## Troubleshooting

### Problema: Build/Runtime error al desplegar app de inferencia a HF Spaces (puerto)
**Causa:** `app_inference/app_inference.py` hardcodea `server_port=7861` (u otro puerto distinto de 7860). HF Spaces solo expone el puerto 7860.
**Síntoma en logs:** `OSError: Cannot find empty port in range: 7861-7861`.
**Solución:**
1. Reemplazar el `launch()` por el patrón condicional:
   ```python
   port = 7860 if os.environ.get("SPACE_ID") else 7861
   demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
   ```
2. Re-subir con `08_deploy_hf.py` (o `hf upload`).
**Steering:** `gradio-interfaces.md` §"Errores Documentados en HF Spaces" → Error 1, y `mlops-deployment.md` §Gotchas.

### Problema: Build error en HF Spaces — `ResolutionImpossible` huggingface-hub vs gradio
**Causa:** `app_inference/requirements.txt` pinea `huggingface-hub==0.25.0` pero `gradio>=5.31` exige `huggingface-hub>=0.28.1`. pip no puede resolver y aborta.
**Síntoma en logs:** `ERROR: Cannot install gradio and huggingface-hub==0.25.0 because these package versions have conflicting dependencies`.
**Solución:**
1. Cambiar la línea en `app_inference/requirements.txt` a `huggingface-hub>=0.28.1` (rango, NO pin estricto).
2. Verificar localmente con `pip install --dry-run -r app_inference/requirements.txt`.
3. Re-subir con `08_deploy_hf.py`.
**Steering:** `gradio-interfaces.md` §"Errores Documentados en HF Spaces" → Error 2, y `huggingface-workflows.md` §Gotchas.

### Problema: Notebook en Colab falla con `ImportError: cannot import name '_center' from 'numpy._core.umath'`
**Causa:** El notebook hizo `%pip install --upgrade pandas numpy ...` o `!uv pip install --system -U numpy ...`. Esto reinstala numpy en disco mientras el kernel ya la tenía cargada en memoria; los binarios C de pandas/sklearn (compilados contra la versión vieja) chocan con la nueva. Cualquier `import pandas` o `import sklearn` posterior explota.
**Síntoma en logs:** trace que termina en `from numpy._core.umath import (...)` con `ImportError: cannot import name '_center'`.
**Solución:**
1. NO actualizar paquetes que Colab ya trae preinstalados (numpy, pandas, sklearn, matplotlib, seaborn, joblib, scipy). Solo instalar lo que falte (típicamente `huggingface-hub`).
2. Usar el patrón `_ensure(pkg, import_name)` con `importlib.import_module` antes de instalar (idempotente).
3. Si el kernel ya quedó contaminado: en Colab `Runtime → Disconnect and delete runtime` y reabrir el notebook.
4. Regenerar el `.ipynb` siguiendo `notebooks-ds.md` Reglas 6, 8 y 9.
**Steering:** `notebooks-ds.md` §Regla 6 (sin `--upgrade` en Colab), §Regla 8 (`uv pip install` con `_ensure`), §Troubleshooting.

### Problema: Recall muy bajo para la clase minoritaria
**Causa:** Desbalance de clases + umbral de 0.5
**Solución:**
1. Calibrar el punto de corte con la curva precision-recall (`workflow-validation.md`)
2. Aplicar SMOTE al conjunto de entrenamiento (`workflow-model-training.md`)
3. Usar `scoring='f1'` en la búsqueda de hiperparámetros
4. Probar `class_weight='balanced'` o `scale_pos_weight`
**Teoría:** consulta `rag-books-mcp` siguiendo `theory-rag-guide.md` — query: `"class imbalance accuracy paradox cost sensitive learning"`.

### Problema: Overfitting (train accuracy >> test accuracy)
**Causa:** Modelo demasiado complejo (alta varianza)
**Solución:**
1. Reducir `max_depth` en árboles
2. Aumentar regularización (`subsample`, `colsample_bytree` en XGBoost; aumentar `alpha` en Ridge/Lasso)
3. Reducir `n_estimators` en Boosting (en RF subir no daña)
4. Usar early stopping en XGBoost (`models-ensemble.md`)
**Teoría:** consulta `rag-books-mcp` siguiendo `theory-rag-guide.md` — queries sugeridos: `"bias variance tradeoff overfitting"` y `"early stopping boosting regularization"`.

### Problema: Variables categóricas causan errores
**Causa:** Tipo de dato incorrecto
**Solución:**
1. Convertir a `str` antes de modelar: `df[cat_cols] = df[cat_cols].astype(str)`
2. XGBoost requiere que la variable objetivo sea `int`
3. Usar `OneHotEncoder` con `handle_unknown='ignore'`

### Problema: Data leakage
**Causa:** Preprocesamiento aplicado antes del split
**Solución:**
1. SIEMPRE hacer split ANTES de SMOTE
2. Fit del scaler SOLO en train, transform en test
3. No usar información del test para seleccionar features
**Teoría:** consulta `rag-books-mcp` siguiendo `theory-rag-guide.md` — query: `"data leakage preprocessing fit transform train test"`.

### Problema: No sabes si usar Ridge, Lasso o árboles
**Causa:** Incertidumbre sobre la estructura de la señal
**Solución:**
- Si la señal es **sparse** (pocas variables importan) → Lasso o árboles
- Si la señal es **distribuida** (todas contribuyen) → Ridge
- Si hay **multicolinealidad** → Ridge o Elastic Net
- Si hay **interacciones no lineales** → Árboles (RF/XGBoost)
**Heurística práctica:** Hacer un benchmarking (ver `models-trees-rf.md` §4). Si los lineales empatan con árboles → señal lineal. Si los árboles dominan → no lineal.
**Teoría:** consulta `rag-books-mcp` siguiendo `theory-rag-guide.md` — query: `"ridge lasso elastic net when to use sparse signal"`.

### Problema: SVM tarda demasiado
**Causa:** Dataset grande con kernel no lineal (O(n²)/O(n³))
**Solución:**
1. Usar `LinearSVC` si la frontera es lineal
2. Submuestrear el dataset
3. Cambiar a familia más escalable (RF, XGBoost)
**Steering:** `models-svm.md`.

### Problema: XGBoost falla con `early_stopping_rounds` en `cross_val_score`
**Causa:** Conflicto conocido entre la API de XGBoost y `cross_val_score`
**Solución:** Usar dos modelos separados — uno sin early stopping para CV, otro con early stopping para entrenar el final.
**Steering:** `models-ensemble.md` §2.

### Problema: Entrenamiento con timeout / demasiado lento post-SMOTE
**Causa:** SMOTE duplicó el dataset a millones de filas y RandomizedSearchCV intenta 150 fits sobre ese volumen. Además, doble paralelización (XGBoost `n_jobs=-1` + sklearn `n_jobs=-1`) causa contención de threads.
**Solución:**
1. Si dataset > 500K filas: NO usar SMOTE. Usar `scale_pos_weight` en XGBoost o `class_weight='balanced'`
2. Si dataset 50K-500K: SMOTE con `sampling_strategy=0.3` (no igualar al 100%)
3. Subsamplear a 100K filas para la búsqueda de hiperparámetros, entrenar modelo final con todos los datos
4. NUNCA `n_jobs=-1` en ambos niveles. Para búsqueda: sklearn paralelo + XGBoost secuencial. Para modelo final: XGBoost paralelo + sklearn secuencial. Usar máximo 70% de los cores disponibles.
5. En Apple Silicon: usar `tree_method="hist"` y limitar al 70% de los performance cores.
**Steering:** `workflow-model-training.md` §Heurísticas de Escalabilidad.

## Fuentes Teóricas

Los steering files de teoría (`theory-*.md`) están basados en cinco libros de referencia. Las notaciones §7.2, §15.3, etc. refieren a secciones de estos libros para quien quiera profundizar:

1. **ESL** — *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman, 2nd Ed.)
   - Disponible gratis: https://hastie.su.domains/ElemStatLearn/
   - Tratamiento matemático riguroso
   - Capítulos clave: 3 (Linear Methods), 7 (Model Assessment), 10 (Boosting), 15 (Random Forests), 16 (Ensemble)

2. **ISLP** — *An Introduction to Statistical Learning with Python* (James, Witten, Hastie, Tibshirani, 2nd Ed.)
   - Disponible gratis: https://www.statlearning.com/
   - Explicaciones accesibles con ejemplos en Python
   - Capítulos clave: 2 (Statistical Learning), 5 (Resampling), 6 (Regularization), 8 (Trees), 9 (SVM), 10 (Deep Learning)

3. **FES** — *Feature Engineering and Selection* (Kuhn, Johnson)
   - Disponible gratis: http://www.feat.engineering/
   - Heurísticas de feature engineering, encoding, selección
   - Capítulos clave: 4 (Exploratory Visualizations), 5 (Categorical), 6 (Numeric), 8 (Missing), 10–12 (Selection)

4. **PDSH** — *Python Data Science Handbook* (VanderPlas)
   - Disponible gratis: https://jakevdp.github.io/PythonDataScienceHandbook/
   - Implementación operativa: NumPy, pandas, matplotlib, scikit-learn
   - Capítulos clave: 3 (pandas), 4 (matplotlib), 5 (Machine Learning con sklearn)

5. **R4DS** — *R for Data Science, 2nd Edition* (Wickham, Çetinkaya-Rundel, Grolemund)
   - Disponible gratis: https://r4ds.hadley.nz/
   - Licencia: **CC BY-NC-ND 3.0 US** (NoDerivatives)
   - Filosofía iterativa de EDA y data wrangling. Capítulo clave: **10 EDA**.
   - ⚠️ **Los ejemplos están en R con tidyverse.** El agente NO debe generar código en R para el usuario. Lo que se reusa son los principios (ciclo iterativo, qué mirar primero, cómo formular preguntas) y se traducen a pandas/seaborn. Detalles y tabla de equivalencias en `theory-rag-guide.md` §"Libros indexados".
   - ✅ **Disponible en todos los modos desde v2.2.0** (Space v1, Space v2 + dataset HF, stdio local). Atribución explícita y mecanismo de takedown documentados en el [DATA_CARD del dataset](https://huggingface.co/datasets/gusdelact/rag-esl-islp-chromadb).

### 🔍 RAG sobre los Libros (MCP Server independiente)

Los cinco libros (con la limitación de R4DS notada arriba) están vectorizados en ChromaDB y disponibles como un **servidor MCP independiente** llamado `rag-books-mcp`. **No forma parte de este Power** — se instala y configura por separado.

#### Por qué es independiente

El RAG sobre ESL/ISLP/FES/PDSH/R4DS es útil más allá de este Power (cualquier proyecto académico, otra área de ML, redacción de papers). Mantenerlo separado evita acoplar el Power a un binario pesado y permite que otros proyectos lo reutilicen.

#### Dos variantes (ambas exponen las mismas 4 tools)

| Variante | Transporte | Usar cuando… |
|---|---|---|
| **A · HF Space (default recomendado)** | streamable HTTP en `/gradio_api/mcp/` | Quieres una URL pública sin instalar nada local. Comparte fácilmente entre máquinas y usuarios. |
| **B · Git Repo local (stdio)** | `uv run python -m rag_books_mcp.server` | Necesitas correrlo offline, en EC2 sin internet, o auditar el código. |

> **Comportamiento del agente:** la primera vez que detecte que necesita usar el RAG y `rag-books-mcp` no esté configurado, **debe preguntar al usuario qué variante prefiere y proponer A (HF Space) como default**. No debe asumir B salvo que el usuario lo pida explícitamente o el contexto lo requiera (sin red, etc.).

#### Variante A · Configurar contra HF Space (default)

1. Asegúrate de que el Space ya existe (lo despliegas una vez con `deploy_to_hf_space.py` desde el repo del server) o usa una URL pública compartida.
2. Registra el servidor en tu `~/.kiro/settings/mcp.json`:

```jsonc
{
  "mcpServers": {
    "rag-books-mcp": {
      "url": "https://<HF_USER>-<SPACE_NAME>.hf.space/gradio_api/mcp/"
    }
  }
}
```

Si tu cliente MCP no soporta streamable HTTP nativo, usa `mcp-remote` como puente:

```jsonc
{
  "mcpServers": {
    "rag-books-mcp": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://<HF_USER>-<SPACE_NAME>.hf.space/gradio_api/mcp/"
      ]
    }
  }
}
```

Para Spaces privados, agrega `"headers": { "Authorization": "Bearer <HF_TOKEN>" }`.

#### Variante B · Configurar contra repo local (stdio)

1. Clona o copia el repositorio `rag-books-mcp` a una ubicación estable (recomendado: `mcp-servers/rag-books-mcp/` dentro de tu workspace, o `~/mcp-servers/`).
2. Regenera el venv: `cd ruta/a/rag-books-mcp && uv sync`.
3. Registra el servidor en tu `~/.kiro/settings/mcp.json` con la ruta absoluta:

```jsonc
{
  "mcpServers": {
    "rag-books-mcp": {
      "command": "uv",
      "args": [
        "run", "--directory",
        "/ruta/absoluta/a/rag-books-mcp",
        "python", "-m", "rag_books_mcp.server"
      ]
    }
  }
}
```

4. Verifica con: `uv run python -c "from rag_books_mcp.server import mcp; print('ok')"` desde la carpeta del servidor.

#### Tools disponibles (cuando el server está activo)

| Tool | Uso |
|------|-----|
| `search_theory(query, book, top_k)` | Búsqueda semántica en ESL/ISLP/FES/PDSH/R4DS. `book ∈ {esl, islp, fes, pdsh, r4ds, both, all}`. |
| `get_section(book, chapter, section)` | Recuperar una sección específica. |
| `cite_foundation(topic, detail_level)` | Fundamentación teórica citando los libros disponibles. |
| `list_available_topics()` | Ver capítulos y temas indexados. |

Los steering files `theory-*.md` de este Power asumen que `rag-books-mcp` está disponible y delegan toda la teoría a estas herramientas en tiempo de uso.

#### Si el servidor no está disponible

Los `theory-*.md` lo detectarán y caerán a un modo degradado: respondes con conocimiento general del modelo y declaras explícitamente que no se pudo consultar la fuente teórica. **No inventes citas a ESL/ISLP/FES/PDSH/R4DS si el RAG no respondió.**
