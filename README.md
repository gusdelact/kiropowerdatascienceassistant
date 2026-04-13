# Data Science Assistant — Kiro Power

Asistente integral para desarrollo de aplicaciones de ciencia de datos y machine learning. Cubre desde ingesta de datos hasta despliegue de modelos.

## ¿Qué hace este Power?

Guía la construcción de proyectos de datos end-to-end combinando documentación de workflows, guías detalladas por etapa y conexiones directas a Kaggle, Hugging Face, Gradio y Tavily mediante MCP.

El Power promueve una **arquitectura modular por etapas** donde cada fase del pipeline es un script independiente, con intervención humana entre cada etapa.

```
Kaggle Dataset → Ingesta → EDA → Feature Engineering → Entrenamiento
    → Validación → Publicar Dataset → Publicar Modelo → Deploy HF Spaces
```

## MCP Servers

| Server | Función | Requiere Token |
|--------|---------|----------------|
| **Tavily** | Documentación actualizada de pandas, numpy, sklearn, TensorFlow, PyTorch, etc. | Sí (`YOUR_TAVILY_API_KEY`) |
| **Kaggle** | Buscar/descargar datasets, explorar competencias, gestionar notebooks | Sí (`YOUR_KAGGLE_TOKEN` — formato KGAT) |
| **Hugging Face** | Buscar modelos/datasets/Spaces, consultar docs de HF, gestionar Spaces | Sí (`YOUR_HF_TOKEN`) |
| **Gradio Docs** | Documentación oficial de Gradio con schemas exactos de componentes | No |

## Steering Files

Guías detalladas que se cargan bajo demanda según la etapa del proyecto:

| Archivo | Etapas | Contenido |
|---------|--------|-----------|
| `eda-feature-engineering.md` | 02, 03 | EDA con pandas/matplotlib/seaborn, Feature Engineering con sklearn ColumnTransformer |
| `model-training-validation.md` | 04, 05 | Entrenamiento con sklearn, XGBoost, TensorFlow, PyTorch. Validación con métricas |
| `kaggle-workflows.md` | 01, 06, 07 | Ingesta desde Kaggle, publicación de datasets con Data Card, modelos con Model Card |
| `gradio-interfaces.md` | Apps | Dos apps Gradio: training (local, 7 tabs) e inference (HF Spaces, predicción) |
| `mlops-deployment.md` | 08 | Inicialización con uv, despliegue a HF Spaces, checklist de producción |

## Configuración

### 1. Obtener tokens

| Token | Dónde obtenerlo |
|-------|-----------------|
| Tavily API Key | https://app.tavily.com → crea cuenta → copia API key |
| Kaggle KGAT Token | https://www.kaggle.com/settings → API → Generate New Token |
| Hugging Face Token | https://huggingface.co/settings/tokens → New token (lectura mínimo) |

### 2. Reemplazar placeholders en `mcp.json`

Abre `mcp.json` y reemplaza:
- `YOUR_TAVILY_API_KEY` → tu API key de Tavily
- `YOUR_KAGGLE_TOKEN` → tu token KGAT de Kaggle
- `YOUR_HF_TOKEN` → tu token de Hugging Face

### 3. Configurar Kaggle CLI (para publicar)

El MCP de Kaggle sirve para buscar y descargar. Para **publicar** datasets y modelos necesitas además el CLI:

```bash
# Descargar kaggle.json desde Kaggle Settings → API → Create New API Token
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

> El token KGAT (MCP) y el API Key (CLI) son credenciales distintas. Necesitas ambas.

## Estructura del Power

```
data-science-assistant/
├── POWER.md              # Documento rector: principios, arquitectura, referencia
├── ARQUITECTURA.md       # Diagramas y documentación de la arquitectura
├── mcp.json              # Configuración de los 4 MCP servers
├── README.md             # Este archivo
└── steering/
    ├── eda-feature-engineering.md
    ├── model-training-validation.md
    ├── kaggle-workflows.md
    ├── gradio-interfaces.md
    └── mlops-deployment.md
```

## Estructura de un Proyecto Generado

```
mi-proyecto-ml/
├── config.yaml                 # Configuración centralizada
├── pyproject.toml              # Dependencias (uv)
├── scripts/
│   ├── 01_ingest.py            # Ingesta de datos
│   ├── 02_eda.py               # Análisis Exploratorio
│   ├── 03_feature_engineering.py
│   ├── 04_train.py             # Entrenamiento
│   ├── 05_validate.py          # Validación
│   ├── 06_publish_dataset.py   # Dataset curado → Kaggle
│   ├── 07_publish_model.py     # Modelo → Kaggle
│   └── 08_deploy_hf.py         # App → HF Spaces
├── app_training/               # UI local (científico de datos)
├── app_inference/              # UI pública (usuario final → HF Spaces)
├── models/                     # Artefactos serializados
├── data/raw/ + data/processed/
├── outputs/figures/ + outputs/metrics/
├── cards/DATA_CARD.md + cards/MODEL_CARD.md
└── lib/                        # Funciones compartidas
```

## Principios Clave

- **uv obligatorio** — Nunca `pip install` directo. Siempre `uv add`, `uv run`, `uvx`.
- **Fuente de datos explícita** — El usuario debe proporcionar la referencia del dataset antes de generar código.
- **Licencia Apache 2.0** — Todo artefacto publicado usa esta licencia.
- **Dos apps Gradio** — Training (local) e Inference (HF Spaces). Audiencias y propósitos distintos.
- **Kaggle como fuente de verdad** — La app de inferencia descarga el modelo desde Kaggle al iniciar.
- **Intervención humana** — Puntos de revisión entre cada etapa del pipeline.

## Casos de Uso Probados

| Proyecto | Tipo | Algoritmo | Resultado |
|----------|------|-----------|-----------|
| Iris Classifier | Clasificación multiclase | XGBoost (17 features engineered) | 96.67% accuracy |
| Salary Predictor | Regresión | Regresión Lineal | R² = 0.89 |

## Requisitos

- [Kiro IDE](https://kiro.dev)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado
- Node.js (para `npx` en MCP servers de Tavily y Kaggle)
- `brew install libomp` en macOS (si usas XGBoost)

## Autor

Gustavo De la Cruz Tovar

## Licencia

Apache 2.0
