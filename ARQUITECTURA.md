# Data Science Assistant — Arquitectura del Power

**Autor**: Gustavo De la Cruz Tovar
**Licencia**: Apache 2.0

---

## ¿Qué es este Power?

El **Data Science Assistant** es un Kiro Power que guía la construcción de aplicaciones de ciencia de datos de principio a fin. No es una librería ni un framework — es un conjunto de documentación, workflows y conexiones a servicios externos (MCP servers) que le enseñan al asistente de IA cómo estructurar, generar y desplegar proyectos de datos de forma profesional y reproducible.

## Estructura del Power

```
data-science-assistant/
├── POWER.md                                # Documento principal: principios, arquitectura, referencia
├── mcp.json                                # Configuración de 4 MCP servers
└── steering/                               # Guías detalladas por workflow
    ├── eda-feature-engineering.md           # EDA y Feature Engineering
    ├── model-training-validation.md         # Entrenamiento y validación
    ├── kaggle-workflows.md                  # Ingesta y publicación en Kaggle
    ├── gradio-interfaces.md                # Dos apps Gradio (training + inference)
    └── mlops-deployment.md                 # Pipelines, despliegue a HF Spaces
```

### POWER.md
Documento central que define los principios obligatorios del Power:
- Uso exclusivo de `uv` y `uvx` para gestión de entornos y paquetes
- Arquitectura modular de 8 scripts independientes
- Intervención humana entre cada etapa
- Dataset de Kaggle explícito como requisito previo
- Licenciamiento Apache 2.0 para todo artefacto publicado
- Estructura de directorios estándar para todo proyecto

### mcp.json
Configura cuatro MCP servers que el Power utiliza:

| MCP Server | Función |
|------------|---------|
| **Tavily** | Consulta documentación actualizada de pandas, numpy, sklearn, TensorFlow, PyTorch, etc. |
| **Kaggle** | Buscar/descargar datasets, publicar datasets curados y modelos entrenados |
| **Hugging Face** | Buscar modelos/datasets/Spaces, consultar docs de HF, gestionar Spaces |
| **Gradio Docs** | Documentación oficial de Gradio con schemas exactos de componentes |

### Steering Files
Guías detalladas que el asistente carga bajo demanda según la etapa en la que se encuentre el usuario. Cada archivo contiene código de referencia, patrones y gotchas para una fase específica del pipeline.

---

## Arquitectura de una Aplicación de Datos

El Power implementa una arquitectura de **dos etapas** con **dos aplicaciones** separadas:

### Etapa 1: Entrenamiento (local)

El científico de datos ejecuta un pipeline secuencial de 7 pasos. Cada paso es un script independiente que produce artefactos. Entre cada paso hay un punto de intervención humana donde se revisan resultados y se ajustan parámetros.

```mermaid
flowchart TD
    subgraph ETAPA1["🏋️ ETAPA 1: ENTRENAMIENTO (ejecución local con uv)"]
        direction TB
        A["📥 01_ingest.py\nKaggle → data/raw/"] -->|"👤 Revisar datos"| B["📊 02_eda.py\nfiguras, reporte estadístico"]
        B -->|"👤 Decidir columnas y transformaciones"| C["⚙️ 03_feature_engineering.py\ndata/raw/ → data/processed/\npreprocessor.joblib"]
        C -->|"👤 Verificar features"| D["🧠 04_train.py\ndata/processed/ → model.joblib"]
        D -->|"👤 Revisar metadata"| E["✅ 05_validate.py\nConfusion Matrix, ROC, R²\nmétricas en JSON"]
        E -->|"👤 ¿Métricas aceptables?"| F["📦 06_publish_dataset.py\nDataset curado → Kaggle\n+ DATA_CARD.md"]
        F -->|"👤 Completar Data Card"| G["📦 07_publish_model.py\nModelo → Kaggle\n+ MODEL_CARD.md"]
    end

    style ETAPA1 fill:#f0f4ff,stroke:#4a6fa5,stroke-width:2px
    style A fill:#e8f5e9,stroke:#2e7d32
    style B fill:#fff3e0,stroke:#ef6c00
    style C fill:#e3f2fd,stroke:#1565c0
    style D fill:#fce4ec,stroke:#c62828
    style E fill:#f3e5f5,stroke:#6a1b9a
    style F fill:#e0f2f1,stroke:#00695c
    style G fill:#e0f2f1,stroke:#00695c
```

**Scripts y sus artefactos:**

| Script | Entrada | Salida | Descripción |
|--------|---------|--------|-------------|
| `01_ingest.py` | Kaggle dataset ref | `data/raw/` | Descarga datos desde Kaggle sin modificar |
| `02_eda.py` | `data/raw/` | `outputs/figures/`, `outputs/reports/` | Genera visualizaciones y reporte estadístico |
| `03_feature_engineering.py` | `data/raw/` | `data/processed/`, `models/preprocessor.joblib` | Transforma datos, guarda preprocessor |
| `04_train.py` | `data/processed/` | `models/model.joblib` | Entrena modelo, guarda serializado |
| `05_validate.py` | `models/model.joblib`, `data/processed/` | `outputs/metrics/`, `outputs/figures/` | Confusion matrix, ROC, R², métricas |
| `06_publish_dataset.py` | `data/processed/` | `cards/DATA_CARD.md` → Kaggle | Genera Data Card y publica dataset curado |
| `07_publish_model.py` | `models/` | `cards/MODEL_CARD.md` → Kaggle | Genera Model Card y publica modelo |

### Etapa 2: Inferencia (Hugging Face Spaces)

La app de inferencia se despliega como un Space público en Hugging Face. Descarga el modelo directamente desde Kaggle y expone una interfaz web para predicciones.

```mermaid
flowchart LR
    subgraph ETAPA2["🔮 ETAPA 2: INFERENCIA (Hugging Face Spaces)"]
        direction LR
        K["📦 Kaggle\nmodel.joblib\npreprocessor.joblib\nMODEL_CARD.md"] -->|"descarga al iniciar"| APP["🖥️ app_inference.py\nGradio UI"]
        APP --> U["👤 Usuario final"]
    end

    subgraph TABS["Tabs de la App"]
        T1["Predicción Individual\nFormulario → resultado"]
        T2["Predicción por Lotes\nUpload CSV → resultados"]
        T3["Info del Modelo\nTipo, features, fuente"]
    end

    APP --> TABS

    subgraph DEPLOY["Despliegue"]
        D["08_deploy_hf.py"] -->|"sube a"| HF["Hugging Face Spaces"]
    end

    style ETAPA2 fill:#fff8e1,stroke:#f9a825,stroke-width:2px
    style K fill:#e0f2f1,stroke:#00695c
    style APP fill:#e3f2fd,stroke:#1565c0
    style U fill:#fce4ec,stroke:#c62828
```

---

## Visión General: Dos Etapas Conectadas

```mermaid
flowchart TB
    subgraph LOCAL["🖥️ Ejecución Local (Científico de Datos)"]
        direction TB
        INGEST["01_ingest.py"] --> EDA["02_eda.py"]
        EDA --> FE["03_feature_eng.py"]
        FE --> TRAIN["04_train.py"]
        TRAIN --> VAL["05_validate.py"]
        VAL --> PUB_DS["06_publish_dataset.py"]
        VAL --> PUB_MOD["07_publish_model.py"]
    end

    subgraph KAGGLE["☁️ Kaggle (Repositorio Central)"]
        DS_CURADO["Dataset Curado\n+ DATA_CARD.md\nApache 2.0"]
        MODELO["Modelo Entrenado\n+ MODEL_CARD.md\n+ preprocessor\nApache 2.0"]
    end

    subgraph HF["☁️ Hugging Face Spaces"]
        APP_INF["app_inference.py\nGradio UI\nPredicción individual y lotes"]
    end

    KAGGLE_SRC["☁️ Kaggle\nDataset Original"] -->|"01_ingest.py"| INGEST
    PUB_DS -->|"publica"| DS_CURADO
    PUB_MOD -->|"publica"| MODELO
    MODELO -->|"descarga al iniciar"| APP_INF
    DEPLOY["08_deploy_hf.py"] -->|"despliega"| APP_INF
    APP_INF -->|"predicciones"| USUARIO["👤 Usuario Final"]

    style LOCAL fill:#f0f4ff,stroke:#4a6fa5,stroke-width:2px
    style KAGGLE fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style HF fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style KAGGLE_SRC fill:#e8f5e9,stroke:#2e7d32
    style USUARIO fill:#fce4ec,stroke:#c62828
```

---

## Dos Aplicaciones Gradio

### App Training (`app_training/app_training.py`)

- **Propósito**: Orquestar todo el pipeline de entrenamiento desde una interfaz visual
- **Usuario**: Científico de datos, ingeniero de datos
- **Ejecución**: Solo local — `uv run python app_training/app_training.py`
- **Nunca se despliega** a producción

La app tiene 7 tabs, uno por cada etapa del pipeline. El usuario ejecuta cada etapa secuencialmente, revisa los resultados en la misma interfaz, y decide cuándo avanzar.

| Tab | Etapa | Qué hace |
|-----|-------|----------|
| 1️⃣ Ingesta | `01_ingest` | Descarga dataset de Kaggle |
| 2️⃣ EDA | `02_eda` | Muestra distribuciones, correlaciones, faltantes |
| 3️⃣ Feature Engineering | `03_feature_eng` | Configura y ejecuta transformaciones |
| 4️⃣ Entrenamiento | `04_train` | Selecciona modelo, hiperparámetros, entrena |
| 5️⃣ Validación | `05_validate` | Muestra confusion matrix, ROC, métricas |
| 6️⃣ Publicar Dataset | `06_publish_ds` | Genera Data Card, prepara para Kaggle |
| 7️⃣ Publicar Modelo | `07_publish_mod` | Genera Model Card, prepara para Kaggle |

### App Inference (`app_inference/app_inference.py`)

- **Propósito**: Exponer el modelo entrenado como servicio de predicción
- **Usuario**: Usuario final (no técnico)
- **Ejecución**: Hugging Face Spaces (público)
- **Fuente del modelo**: Descarga desde Kaggle al iniciar

La app tiene 3 tabs:

| Tab | Qué hace |
|-----|----------|
| Predicción Individual | Formulario con inputs → predicción con probabilidades |
| Predicción por Lotes | Upload CSV → predicciones masivas → descarga resultados |
| Información del Modelo | Tipo de modelo, features, fuente en Kaggle |

---

## Rol de Kaggle y Hugging Face

### Kaggle: Repositorio de Datos y Modelos

Kaggle cumple dos funciones en esta arquitectura:

**1. Fuente de datos (entrada)**
- El pipeline comienza descargando un dataset de Kaggle (`01_ingest.py`)
- El usuario DEBE proporcionar la referencia explícita: `usuario/nombre-dataset`

**2. Repositorio de artefactos (salida)**
- El dataset curado (resultado de Feature Engineering) se publica en Kaggle con una **Data Card**
- El modelo entrenado se publica en Kaggle con una **Model Card**
- Kaggle es la **fuente de verdad** del modelo: la app de inferencia lo descarga de ahí

```mermaid
flowchart LR
    subgraph ENTRADA["Kaggle como Fuente de Datos"]
        DS_ORIG["📂 Dataset Original\nusuario/nombre-dataset"]
    end

    subgraph PIPELINE["Pipeline Local"]
        P["Ingesta → EDA → FE\n→ Train → Validate"]
    end

    subgraph SALIDA["Kaggle como Repositorio de Artefactos"]
        DS_CUR["📦 Dataset Curado\n+ DATA_CARD.md\nApache 2.0"]
        MOD["🧠 Modelo Entrenado\n+ MODEL_CARD.md\n+ preprocessor.joblib\n+ métricas\nApache 2.0"]
    end

    DS_ORIG -->|"01_ingest.py"| P
    P -->|"06_publish_dataset.py"| DS_CUR
    P -->|"07_publish_model.py"| MOD

    style ENTRADA fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style PIPELINE fill:#f0f4ff,stroke:#4a6fa5,stroke-width:2px
    style SALIDA fill:#e0f2f1,stroke:#00695c,stroke-width:2px
```

### Hugging Face: Plataforma de Despliegue

Hugging Face Spaces es donde se despliega la app de inferencia para el usuario final.

```mermaid
flowchart LR
    K["📦 Kaggle\nmodel.joblib\npreprocessor.joblib"] -->|"descarga al iniciar"| HF["🖥️ HF Spaces\napp_inference.py\nGradio UI"]
    HF -->|"predicciones vía browser"| U["👤 Usuario Final"]

    style K fill:#e0f2f1,stroke:#00695c
    style HF fill:#fff3e0,stroke:#ef6c00
    style U fill:#fce4ec,stroke:#c62828
```

**Actualización del modelo**: Publicar nueva versión en Kaggle → reiniciar el Space → la app usa el modelo actualizado automáticamente. No se necesita re-desplegar.

---

## Flujo Completo del Modelo

```mermaid
sequenceDiagram
    participant CD as 👨‍🔬 Científico de Datos
    participant LOCAL as 🖥️ Pipeline Local
    participant KAGGLE as ☁️ Kaggle
    participant HF as ☁️ HF Spaces
    participant USER as 👤 Usuario Final

    CD->>LOCAL: uv run python scripts/01_ingest.py
    LOCAL->>KAGGLE: Descarga dataset original
    KAGGLE-->>LOCAL: data/raw/

    CD->>CD: 👤 Revisa datos, ajusta config.yaml

    CD->>LOCAL: uv run python scripts/02_eda.py
    LOCAL-->>CD: Figuras, reporte estadístico

    CD->>CD: 👤 Decide columnas y transformaciones

    CD->>LOCAL: uv run python scripts/03_feature_engineering.py
    LOCAL-->>CD: data/processed/, preprocessor.joblib

    CD->>CD: 👤 Verifica features

    CD->>LOCAL: uv run python scripts/04_train.py
    LOCAL-->>CD: model.joblib

    CD->>LOCAL: uv run python scripts/05_validate.py
    LOCAL-->>CD: Métricas, confusion matrix, ROC

    CD->>CD: 👤 ¿Métricas aceptables?

    CD->>LOCAL: uv run python scripts/06_publish_dataset.py
    LOCAL->>KAGGLE: Dataset curado + DATA_CARD.md

    CD->>CD: 👤 Completa Data Card

    CD->>LOCAL: uv run python scripts/07_publish_model.py
    LOCAL->>KAGGLE: Modelo + MODEL_CARD.md

    CD->>CD: 👤 Completa Model Card

    CD->>LOCAL: uv run python scripts/08_deploy_hf.py
    LOCAL->>HF: Despliega app_inference.py

    USER->>HF: Accede al Space
    HF->>KAGGLE: Descarga modelo al iniciar
    KAGGLE-->>HF: model.joblib, preprocessor.joblib
    USER->>HF: Envía datos para predicción
    HF-->>USER: Resultado de predicción
```

---

## Gestión de Entorno

Todo el proyecto usa `uv` como gestor de paquetes y entornos virtuales:

| Acción | Comando |
|--------|---------|
| Crear proyecto | `uv init mi-proyecto-ml` |
| Agregar dependencia | `uv add pandas scikit-learn` |
| Ejecutar script | `uv run python scripts/01_ingest.py` |
| Ejecutar CLI tool | `uvx kaggle datasets list -s "query"` |
| Login en HF | `uvx huggingface-cli login` |

Nunca se usa `pip install` directamente ni `python -m venv`. El `pyproject.toml` y `uv.lock` garantizan reproducibilidad.

---

## Configuración Centralizada

Todos los parámetros del proyecto viven en un único `config.yaml`. Los scripts leen este archivo y nunca tienen valores hardcodeados. Esto permite que el científico de datos ajuste parámetros entre etapas sin modificar código.

Parámetros clave:
- **data.kaggle_ref**: Referencia del dataset en Kaggle (obligatorio)
- **features.target**: Columna objetivo
- **features.numeric / categorical**: Columnas a procesar
- **model.type**: Tipo de modelo (random_forest, xgboost, tensorflow, pytorch)
- **model.output_format**: Formato de serialización (joblib, pickle, keras, pth)
- **publish.kaggle_username**: Usuario de Kaggle para publicación
- **publish.hf_username**: Usuario de HF para despliegue

---

## Data Card y Model Card

Todo artefacto publicado a Kaggle incluye una card descriptiva:

### Data Card (`cards/DATA_CARD.md`)
Documenta el dataset curado:
- Descripción y fuente original
- Composición (filas, columnas, tipos)
- Preprocesamiento aplicado
- Valores faltantes del dataset original
- Uso previsto y limitaciones
- Licencia Apache 2.0

### Model Card (`cards/MODEL_CARD.md`)
Documenta el modelo entrenado:
- Tipo de modelo y framework
- Datos de entrenamiento y features
- Métricas de evaluación
- Hiperparámetros
- Uso previsto y limitaciones
- Consideraciones éticas
- Instrucciones de uso
- Licencia Apache 2.0

Ambas cards se generan automáticamente con secciones `[COMPLETAR]` que el científico de datos debe revisar y completar antes de publicar.

---

## Intervención Humana

La arquitectura está diseñada para que la automatización y la intervención humana coexistan. Cada etapa produce artefactos visibles (CSVs, figuras, JSONs, cards) que el equipo revisa antes de avanzar.

```mermaid
flowchart LR
    subgraph ROLES["Roles por Etapa"]
        direction TB
        IE["🔧 Ingeniero de Datos\nIngesta, Feature Engineering"]
        CD["👨‍🔬 Científico de Datos\nEDA, Entrenamiento, Validación"]
        AMBOS["🤝 Ambos\nRevisión de Cards, Despliegue"]
    end

    style IE fill:#e3f2fd,stroke:#1565c0
    style CD fill:#f3e5f5,stroke:#6a1b9a
    style AMBOS fill:#fff3e0,stroke:#ef6c00
```

| Entre etapas | Qué revisa el humano |
|-------------|---------------------|
| Ingesta → EDA | ¿Los datos descargados son correctos? ¿Formato esperado? |
| EDA → Feature Eng. | ¿Qué columnas usar? ¿Hay outliers que tratar? ¿Qué transformaciones? |
| Feature Eng. → Train | ¿El dataset procesado tiene sentido? ¿Las features son correctas? |
| Train → Validate | ¿El modelo converge? ¿Los parámetros son razonables? |
| Validate → Publish | ¿Las métricas son aceptables? ¿El modelo es suficientemente bueno? |
| Publish → Deploy | ¿Las cards están completas? ¿La app funciona localmente? |

Esta separación permite que diferentes roles participen en diferentes etapas: el ingeniero de datos en ingesta y feature engineering, el científico de datos en entrenamiento y validación, y ambos en la revisión de cards y despliegue.
