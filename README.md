# Data Science Assistant — Kiro Power

Asistente integral para desarrollo de aplicaciones de ciencia de datos y machine learning. Cubre desde ingesta de datos hasta despliegue de modelos.

## Qué hace

- Guía el desarrollo de proyectos ML con una **arquitectura modular por etapas** (8 scripts independientes + generador de notebooks)
- Usa **Hugging Face Hub como única fuente de datos** y como repositorio central de datasets curados, modelos y Spaces (Kaggle no es soportado en esta versión)
- Obliga al agente a **diseñar antes de codificar**: produce `notes/0N_design_*.md` con consultas reales al RAG antes de escribir código de FE, modelado o validación
- Incluye un **MCP server independiente (`rag-books-mcp`)** que expone ESL e ISLP con búsqueda semántica, en lugar de búsqueda web genérica
- Promueve **intervención humana** entre cada etapa del pipeline
- Genera **dos aplicaciones Gradio**: un dashboard de resultados (local) y una app de inferencia que se despliega a HF Spaces
- Genera **dos notebooks autocontenidos para Google Colab** (entrenamiento + inferencia) publicados en HF Hub con badge "Open in Colab"
- Codifica gotchas de producción descubiertos en validaciones reales: puerto 7860 en HF Spaces, `huggingface-hub>=0.28.1`, no SMOTE en datasets > 500K filas, calibración de umbral con curva PR cuando hay desbalance, etc.

## Proceso de Diseño

El Power no se diseñó en abstracto. Se construyó en tres oleadas sobre material real, y cada decisión tiene una huella concreta en el repositorio. Para la cronología completa con archivos, fechas y conversaciones que detonaron cada cambio, ver [`MEMORIA_CONSTRUCCION_POWER.md`](../../MEMORIA_CONSTRUCCION_POWER.md) en la raíz del workspace.

### Fuentes de Referencia

**Libros base (canónicos, fundamentación teórica):**

1. **An Introduction to Statistical Learning with Applications in Python (ISLP)** — James, Witten, Hastie, Tibshirani & Taylor. Springer, 2nd Edition (2023). Disponible gratuitamente en https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html. Cubre los fundamentos de aprendizaje estadístico, validación cruzada, regularización, árboles de decisión, SVM, clustering y más. Es la referencia principal para la estructura del pipeline de validación y selección de modelos.

2. **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani & Friedman. Springer, 2nd Edition (2009). Disponible gratuitamente en https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf. Tratamiento más profundo y matemático de los mismos temas. Se usó como referencia para las decisiones de Feature Engineering, boosting (base teórica de XGBoost), y métricas de evaluación.

Ambos libros están vectorizados en ChromaDB (1977 chunks, embeddings con `all-MiniLM-L6-v2`) y se sirven como MCP server independiente (`rag-books-mcp`). El agente los consulta en tiempo de uso, no los memoriza.

**Notebooks de clase (estructura pedagógica de los `models-*.md`):**

Los cuatro archivos `models-*.md` no se inventaron desde cero. Heredaron taxonomía, intuición e hiperparámetros de cinco notebooks de la maestría que viven en `notebooks_ejemplo/`:

| Notebook | Tema | Steering generado |
|---|---|---|
| `ANH_IA_L01_L02_*` | Regresión lineal, dataset Boston, mínimos cuadrados | `models-linear-regression.md` |
| `ANH_IA_L03L04_RTreeRF_*` + `ANH_L05_L06_RFRLCT_*` | Regression Trees + intuición de Random Forest (bootstrap + agregación) | `models-trees-rf.md` |
| `ANH_L07_L08_L09_Ensemble_*` | Bagging, RF Classifier, Boosting | `models-ensemble.md` |
| `ANH_IA_L10_SVM_*` | SVM con kernels, make_moons, frontera no lineal | `models-svm.md` |

Los notebooks aportaron el "qué" del modelo (intuición → fórmula → código → ejemplos). Las validaciones reales aportaron el "cómo no romperlo en producción" (heurísticas operacionales que se sumaron encima).

### Metodología de Construcción

El power evolucionó en tres oleadas claramente trazables en el repositorio:

1. **Esqueleto inicial (abril 2026)** — `POWER.md`, `mlops-deployment.md`, `gradio-interfaces.md`, `README.md` mínimos. Cuatro MCP servers contemplados (Tavily, Kaggle, HF, Gradio Docs). Pipeline conceptual sin steering files de modelos ni de teoría.

2. **Materialización del 16 de mayo (madrugada, 00:45 – 02:25)** — En menos de dos horas se crearon `huggingface-workflows.md`, `ARQUITECTURA.md`, los cuatro `workflow-*.md` y los cuatro `models-*.md` directamente sobre los notebooks de clase. La taxonomía del power calca la taxonomía del curso: L01-02 → lineales, L03-06 → árboles, L07-09 → ensembles, L10 → SVM.

3. **Refinamiento del 16-18 de mayo (cinco pipelines reales)** — Cada validación dejó hallazgos que se promovieron a steering files o a §Troubleshooting de `POWER.md`. Esta oleada cambió el carácter del power: dejó de ser un manual y se volvió una destilación de fricciones reales.

### Tres decisiones de diseño que reformaron el power

A lo largo del refinamiento, tres decisiones cambiaron la forma del power:

**1. Hugging Face Hub como única fuente de datos.** La versión inicial soportaba Kaggle (con doble credencial: token `KGAT_` para el MCP + `kaggle.json` para el CLI) y triple sintaxis de licencia (`"Apache 2.0"` con espacio en JSON de Kaggle, `apache-2.0` en YAML de HF, `Apache-2.0` en `config.yaml`). Eliminar Kaggle bajó la fricción de setup, eliminó la triple sintaxis y concentró el ciclo dataset-curado → modelo → app en un único namespace. Los proyectos `proyectos/salary-predictor/kaggle-model/` y `proyectos/iris-classifier/kaggle-model/` son las huellas del power viejo.

**2. RAG sobre libros canónicos en lugar de búsqueda web genérica.** Se eliminó Tavily como MCP server (resultados no auditables, costo, redundancia con las herramientas web del agente) y se sustituyó por `rag-books-mcp`, un servidor MCP independiente que expone semánticamente ESL e ISLP. El cambio elevó la calidad de las citas: de "según un blog" pasó a `[ESL §10.10]`, `[ISLP §8.1.2]`. El `mcp.json` actual del power está literalmente vacío; el RAG vive aparte para que sea reusable por otros proyectos.

**3. "Diseñar antes de ejecutar" como protocolo bloqueante.** Una vez que el agente tiene un RAG de calidad, hay que decidir si lo consulta antes (para fundamentar) o después (para decorar). La interrupción del usuario en una de las validaciones (*"te interrumpí porque hiciste el código sin diseñar y tomar en cuenta los libros"*) detonó `theory-driven-design.md`: el agente está obligado a producir `notes/02_design_fe.md`, `notes/03_design_modeling.md` y `notes/04_design_validation.md` con consultas reales al RAG **antes** de escribir código de la fase correspondiente (la nota 01 de EDA tiene timing bipartito y se cierra al terminar el EDA). El power tiende cada vez más a ese patrón: la fase de diseño dejó de ser un comentario flotante para convertirse en un artefacto físico del proyecto.

El resultado es un Power que no solo genera código, sino que encapsula **método de trabajo** basado en experiencia práctica real.

## MCP Servers

| Server | Función | Requiere Token |
|--------|---------|----------------|
| **gradio-docs** | Documentación oficial de Gradio con schemas exactos de componentes | No |
| **rag-books-mcp** | RAG sobre ESL e ISLP. Búsqueda semántica en los libros de referencia. Soporta dos transportes: **HF Space (default)** o **Git repo local (stdio)**. | No (embeddings locales) |

> Para búsquedas web generales (docs de pandas, sklearn, statsmodels, etc.) el agente usa las herramientas de búsqueda incorporadas (`remote_web_search` / `web_fetch`). El power no incluye un MCP server adicional para esa capa.

## Steering Files

Guías detalladas que se cargan bajo demanda según la etapa del proyecto:

**Workflow (cómo ejecutar cada fase):**

| Archivo | Etapas | Contenido |
|---------|--------|-----------|
| `workflow-eda.md` | 02 | Análisis exploratorio: duplicados, nulos, distribuciones, outliers, correlación |
| `workflow-feature-engineering.md` | 03 | IQR clipping, eliminación de categóricas raras, split, imputación, scaling, encoding |
| `workflow-model-training.md` | 04 | Flujo común: SMOTE, GridSearch, CV, serialización |
| `workflow-validation.md` | 05 | Métricas, calibración del umbral, ROC, residuales, benchmarking |

**Modelos por familia (heurísticas específicas):**

| Archivo | Familia |
|---------|---------|
| `models-linear-regression.md` | OLS, Ridge, Lasso, ElasticNet |
| `models-trees-rf.md` | Decision Trees, Random Forest, ExtraTrees |
| `models-ensemble.md` | Bagging + Boosting (RF Classifier, XGBoost, Stacking) |
| `models-svm.md` | LinearSVC, SVC kernels, SVR, OneClassSVM |

**Teoría (por qué funciona):**

| Archivo | Tema |
|---------|------|
| `theory-rag-guide.md` | Manual operativo del servidor MCP `rag-books-mcp` (ESL + ISLP). Define las 4 tools, el formato de citas y el modo degradado. Es el **cómo** consultar la teoría. |
| `theory-driven-design.md` | Protocolo obligatorio de uso del RAG **antes** de codificar. Exige producir `notes/01_design_eda.md`, `notes/02_design_fe.md`, `notes/03_design_modeling.md` y `notes/04_design_validation.md` con consultas reales y decisiones documentadas. Es el **cuándo** y el entregable. |

**Infraestructura:**

| Archivo | Etapas | Contenido |
|---------|--------|-----------|
| `huggingface-workflows.md` | 01, 06, 07 | Ingesta, publicación de datasets con Data Card, modelos con Model Card |
| `gradio-interfaces.md` | Apps | Dashboard de resultados (EDA + validación) y app de inferencia (HF Spaces) |
| `mlops-deployment.md` | 08 | Inicialización con uv, despliegue a HF Spaces, checklist de producción |
| `matplotlib-headless.md` | 02, 05 | Backend `Agg`, persistencia de figuras a `outputs/figures/`, convenciones de nomenclatura |
| `notebooks-ds.md` | 09 | Notebooks autocontenidos para Google Colab: badge "Open in Colab", `uv pip install` solo para lo que falte, separación entrenamiento / inferencia, `model_info.json`, publicación en HF Hub |

## Configuración

### 1. Obtener tokens

| Token | Dónde obtenerlo |
|-------|-----------------|
| Hugging Face Token | https://huggingface.co/settings/tokens → New token (con permisos de escritura) |

### 2. Configurar el RAG MCP Server (rag-books-mcp)

El server de RAG soporta **dos variantes** que exponen las mismas 4 tools. Por defecto recomendamos la opción A (HF Space) porque no requiere instalación local en cada máquina que consuma el RAG.

> ⚠️ **Si Kiro te muestra "Connection closed" o "os error 2" al conectar**, NO toques el código del server. Lee primero la sección **Troubleshooting** del README del server (`mcp-servers/rag-books-mcp/README.md`). Documenta los tres tropiezos más comunes: precedencia user vs workspace de `mcp.json`, transporte correcto del HF Space, y `PATH` heredado por Kiro.

#### Opción A · HF Space (default recomendado)

Despliega el server una vez como HF Space y conéctalo por URL. Desde el repo del server (no desde el Power):

```bash
cd mcp-servers/rag-books-mcp
uv sync

export HF_TOKEN=hf_xxx       # https://huggingface.co/settings/tokens (write)
export HF_USER=tu-usuario
uv run python deploy_to_hf_space.py
```

Esto crea `https://huggingface.co/spaces/<HF_USER>/rag-books-mcp`. El endpoint MCP queda en:

```
https://<HF_USER>-rag-books-mcp.hf.space/gradio_api/mcp/
```

Configúralo en `~/.kiro/settings/mcp.json`:

```jsonc
{
  "mcpServers": {
    "rag-books-mcp": {
      "url": "https://<HF_USER>-rag-books-mcp.hf.space/gradio_api/mcp/"
    }
  }
}
```

#### Opción B · Git Repo local (stdio)

Útil para uso offline, en EC2 sin internet, o cuando quieres auditar el código.

```bash
cd mcp-servers/rag-books-mcp
uv sync
```

**Verificar que funciona:**
```bash
uv run python -c "from rag_books_mcp.server import mcp; print('✅ RAG Server listo')"
```

Configúralo en `~/.kiro/settings/mcp.json` con la ruta absoluta:

```jsonc
{
  "mcpServers": {
    "rag-books-mcp": {
      "command": "uv",
      "args": [
        "run", "--directory",
        "/ruta/absoluta/a/mcp-servers/rag-books-mcp",
        "python", "-m", "rag_books_mcp.server"
      ]
    }
  }
}
```

**Re-ingesta (solo si modificas los capítulos):**
```bash
uv run python -m rag_books_mcp.ingest --books-dir /ruta/a/carpeta/ebook
```

La base vectorial incluye:
- **ESL:** 1093 chunks de 22 capítulos
- **ISLP:** 884 chunks de 16 capítulos
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (se descarga automáticamente la primera vez)

### 3. Instalar y configurar HF CLI

```bash
# Instalar
brew install hf
# O: curl -LsSf https://hf.co/cli/install.sh | bash

# Login
hf auth login
# Pega tu token de https://huggingface.co/settings/tokens (con permisos write)
```

### 4. Instalar uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 5. Verificar

En Kiro, activa el power y verifica que los MCP servers conectan correctamente.

## Estructura del Power

```
data-science-assistant/
├── POWER.md              # Documento rector: principios, arquitectura, referencia
├── ARQUITECTURA.md       # Diagramas y documentación de la arquitectura
├── mcp.json              # Configuración de MCP servers
├── README.md             # Este archivo
├── rag-books-mcp/        # MCP Server de RAG (autocontenido)
│   ├── pyproject.toml    # Dependencias del server
│   ├── README.md         # Documentación del RAG server
│   ├── rag_books_mcp/    # Código fuente
│   │   ├── __init__.py
│   │   ├── server.py     # MCP Server (FastMCP + ChromaDB)
│   │   └── ingest.py     # Script de vectorización (opcional)
│   └── chroma_db/        # Base vectorial pre-construida (41 MB)
│       ├── chroma.sqlite3
│       └── [colecciones HNSW]
└── steering/
    ├── workflow-eda.md
    ├── workflow-feature-engineering.md
    ├── workflow-model-training.md
    ├── workflow-validation.md
    ├── models-linear-regression.md
    ├── models-trees-rf.md
    ├── models-ensemble.md
    ├── models-svm.md
    ├── theory-rag-guide.md
    ├── theory-driven-design.md
    ├── huggingface-workflows.md
    ├── gradio-interfaces.md
    ├── mlops-deployment.md
    ├── matplotlib-headless.md
    └── notebooks-ds.md
```

## Pipeline de un Proyecto Generado

```
01_ingest.py        → Descarga datos de Hugging Face Hub
02_eda.py           → Análisis exploratorio (figuras, reporte)
03_feature_eng.py   → Feature Engineering (preprocessor.joblib)
04_train.py         → Entrenamiento del modelo (model.joblib)
05_validate.py      → Validación con métricas (JSON + figuras)
06_publish_data     → Publica dataset curado a HF Hub + Data Card
07_publish_model    → Publica modelo a HF Hub + Model Card
08_deploy_hf.py     → Despliega app de inferencia a HF Spaces
build_notebooks.py  → Genera notebooks/01_entrenamiento.ipynb y
                      notebooks/02_inferencia.ipynb (autocontenidos
                      para Colab) y los sube al repo del modelo en HF Hub
```

## Estructura de un Proyecto Generado

```
mi-proyecto-ml/
├── config.yaml                 # Configuración centralizada
├── pyproject.toml              # Dependencias (uv)
├── scripts/                    # 8 etapas independientes + build_notebooks.py
├── notebooks/                  # 01_entrenamiento.ipynb y 02_inferencia.ipynb (Colab)
├── app_results/                # Dashboard local de resultados (EDA + validación)
├── app_inference/              # UI pública (usuario final → HF Spaces)
├── models/                     # Artefactos serializados
├── data/raw/ + data/processed/
├── outputs/figures/ + outputs/metrics/
├── cards/DATA_CARD.md + cards/MODEL_CARD.md
└── lib/                        # Funciones compartidas
```

## Principios Clave

- **uv obligatorio** — Nunca `pip install` directo. Siempre `uv add`, `uv run`, `uvx`. En Colab, `!uv pip install --system` solo para lo que falte (no `--upgrade`).
- **Fuente de datos explícita** — El usuario debe proporcionar la referencia del dataset antes de generar código.
- **Licencia Apache 2.0** — Todo artefacto publicado usa esta licencia.
- **Dashboard de resultados** — Se genera siempre. Visualiza figuras de EDA y validación con Gradio local.
- **App de Inferencia** — Se despliega a HF Spaces. Descarga el modelo desde HF Hub al iniciar.
- **HF Hub como fuente de verdad** — La app de inferencia descarga el modelo desde Hugging Face al iniciar.
- **Notebooks autocontenidos para Colab** — Cada proyecto entrega `01_entrenamiento.ipynb` y `02_inferencia.ipynb` que abren en Colab con un clic, sin clonar el repo y sin tocar paquetes preinstalados (ver `notebooks-ds.md`).
- **Intervención humana** — Puntos de revisión entre cada etapa del pipeline.

## Casos de Uso Probados

El power se validó contra cinco pipelines reales entre el 16 y el 18 de mayo de 2026, cada uno con un dataset distinto, un tipo de problema distinto y una familia de modelo distinta. La traza completa de cada conversación vive en `validaciones/` (raíz del workspace).

| # | Validación | Dataset | Tipo de problema | Modelo | Resultado clave |
|---|---|---|---|---|---|
| 1 | Credit Card Fraud | `alenc123/credit-card-fraud` (1.29M filas, 0.58% fraudes) | Clasificación binaria, desbalance severo (~172:1) | XGBoost + `scale_pos_weight` | F1 (clase fraude) = 0.81 con umbral calibrado por curva PR; ROC-AUC = 0.998. Detonó la regla de no usar SMOTE en datasets > 500K filas. |
| 2 | USA Housing | `gusdelact/USA_Housing` (5000 filas, 7 columnas) | Regresión continua | Linear Regression + benchmarking vs Ridge/Lasso/ElasticNet | R² = 0.9146, RMSE = $102K, MAPE = 7.42%. Aclaró que en regresión no aplica SMOTE: el equivalente es split estratificado por bins del target. |
| 3 | Wine Fraud | `gusdelact/wine_fraud` (6497 filas, 3.8% fraudes) | Clasificación binaria con desbalance | RF + SMOTE vs XGBoost + `scale_pos_weight` | RF gana por margen estrecho (F1 = 0.365 vs 0.351). Primer pipeline con `notes/0N_design_*.md` consultando RAG antes de codificar. |
| 4 | Mouse Viral Study | `gusdelact/mouse_viral_study` (400 filas, balanceado) | Clasificación binaria | SVM con kernel RBF | F1 = 1.0. Detonó cinco hallazgos (ver `validaciones/hallazgos_mousevirus_pipeline.md`) que crearon `notebooks-ds.md`, `matplotlib-headless.md` y la Regla 0 de `gradio-interfaces.md`. |
| 5 | Penguins (v1 + v2) | `gusdelact/penguins` (344 filas, 3 especies) | Clasificación multiclase | Decision Tree (max_depth=3) | F1-weighted = 0.988. La interrupción del usuario en v1 (*"hiciste el código sin diseñar"*) detonó `theory-driven-design.md`. v2 documentó los dos errores recurrentes de despliegue a HF Spaces (puerto 7860, `huggingface-hub>=0.28.1`). |

Cada falla concreta de estos cinco pipelines se promovió a regla obligatoria en un steering file. El power final es la destilación de esas cinco validaciones encima del material de los notebooks de clase.

**Casos de uso anteriores al refinamiento (con power viejo, pre-13 abril):**

| Proyecto | Tipo | Algoritmo | Resultado |
|----------|------|-----------|-----------|
| Iris Classifier | Clasificación multiclase | XGBoost (17 features engineered) | 96.67% accuracy |
| [Salary Predictor](https://huggingface.co/spaces/gusdelact/salary-predictor-lr) | Regresión | Regresión Lineal | R² = 0.89 |

## Requisitos

- [Kiro IDE](https://kiro.dev)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [hf CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) — Para publicar datasets, modelos y Spaces
- Node.js (para `npx` con `mcp-remote`, si usas el RAG vía HF Space)
- `brew install libomp` en macOS (si usas XGBoost)

## Autor

Gustavo De la Cruz Tovar

## Licencia

Apache 2.0
