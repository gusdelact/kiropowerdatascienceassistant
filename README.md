# Data Science Assistant — Kiro Power

Asistente integral para desarrollo de aplicaciones de ciencia de datos y machine learning. Cubre desde ingesta de datos hasta despliegue de modelos.

## Qué hace

- Guía el desarrollo de proyectos ML con una **arquitectura modular por etapas** (9 scripts independientes)
- Conecta directamente con **Hugging Face** (datasets, modelos y Spaces) para publicación y despliegue
- Incluye MCP servers para consultar documentación de librerías Python y Gradio
- Promueve **intervención humana** entre cada etapa del pipeline
- Genera dos aplicaciones Gradio: un dashboard de resultados (local) y una de inference (HF Spaces)
- Genera **dos notebooks autocontenidos para Google Colab** (entrenamiento + inferencia) publicados en HF Hub con badge "Open in Colab"

## Proceso de Diseño

El Power no se diseñó en abstracto. Se construyó a partir de material real y experiencia práctica.

### Fuentes de Referencia

**Libros base:**

1. **An Introduction to Statistical Learning with Applications in Python (ISLP)** — James, Witten, Hastie, Tibshirani & Taylor. Springer, 2nd Edition (2023). Disponible gratuitamente en https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html. Cubre los fundamentos de aprendizaje estadístico, validación cruzada, regularización, árboles de decisión, SVM, clustering y más. Es la referencia principal para la estructura del pipeline de validación y selección de modelos.

2. **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani & Friedman. Springer, 2nd Edition (2009). Disponible gratuitamente en https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf. Tratamiento más profundo y matemático de los mismos temas. Se usó como referencia para las decisiones de Feature Engineering, boosting (base teórica de XGBoost), y métricas de evaluación.

Ambos libros son de los mismos autores de Stanford y se complementan: ISL es accesible y práctico, ESL es riguroso y profundo.

**Material de referencia:**
Las heurísticas y patrones de código están basados en:
- La secuencia lógica de un proyecto de ML (EDA → preprocesamiento → entrenamiento → evaluación)
- Patrones probados con pandas, sklearn, matplotlib y seaborn
- Buenas prácticas de validación cruzada y manejo de data leakage
- Estructura de scripts reproducibles

### Metodología de Construcción

1. **Análisis de mejores prácticas** — Se revisaron los libros de referencia y proyectos reales para extraer el flujo de trabajo común en proyectos de datos
2. **Abstracción en etapas** — Se identificaron las 8 etapas recurrentes y se formalizaron como scripts independientes
3. **Integración de herramientas** — Se conectaron los MCP servers de Gradio Docs y RAG Books para que el asistente pudiera operar directamente con la documentación oficial y la teoría de referencia
4. **Documentación como código** — Los patrones extraídos se codificaron en steering files que el asistente carga bajo demanda
5. **Validación con casos reales** — Se probó el Power con proyectos completos para verificar que el flujo funcionaba end-to-end
6. **Iteración con feedback** — Se refinó el POWER.md incorporando gotchas y limitaciones descubiertas durante la implementación real (licencias, XGBoost en macOS, versiones de Gradio en HF Spaces)

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
| `theory-driven-design.md` | Protocolo obligatorio de uso del RAG **antes** de codificar. Exige producir `notes/01_design_fe.md`, `notes/02_design_modeling.md` y `notes/03_design_validation.md` con consultas reales y decisiones documentadas. Es el **cuándo** y el entregable. |

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
