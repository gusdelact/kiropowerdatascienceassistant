# Kaggle Workflows

## Requisito Previo: Dataset Explícito

**ANTES de generar cualquier código de ingesta**, el usuario DEBE proporcionar:
- **Referencia del dataset**: formato `usuario/nombre-dataset` (ejemplo: `uciml/iris`, `zillow/zecon`)
- **Nombre del usuario/organización** que publica el dataset en Kaggle

Si el usuario no ha proporcionado esta información, PREGUNTAR explícitamente:
> "¿Cuál es el dataset de Kaggle que quieres usar? Necesito el formato `usuario/nombre-dataset` (ejemplo: `uciml/iris`)."

NUNCA inventar o asumir un dataset. El `config.yaml` se genera con estos datos.

## Etapa 1: Script 01_ingest.py — Ingesta de Datos

Descarga datos desde Kaggle (u otra fuente) y los guarda en `data/raw/`. NUNCA modifica los datos originales.

### Estructura del Script

```python
#!/usr/bin/env python3
"""01_ingest.py — Ingesta de datos desde Kaggle u otra fuente.

Descarga datos y los guarda en data/raw/ sin modificar.
Artefactos: data/raw/

Uso: uv run python scripts/01_ingest.py
"""
import yaml
import subprocess
import os
from pathlib import Path

# Cargar configuración
with open("config.yaml") as f:
    config = yaml.safe_load(f)

raw_path = config["data"]["raw_path"]
Path(raw_path).mkdir(parents=True, exist_ok=True)

source = config["data"]["source"]

if source == "kaggle":
    kaggle_ref = config["data"]["kaggle_ref"]
    print(f"Descargando dataset de Kaggle: {kaggle_ref}")

    # Usar uvx para ejecutar kaggle CLI sin instalarlo globalmente
    subprocess.run([
        "uvx", "kaggle", "datasets", "download",
        "-d", kaggle_ref,
        "--unzip",
        "-p", raw_path
    ], check=True)

elif source == "kaggle_competition":
    competition = config["data"]["kaggle_competition"]
    print(f"Descargando datos de competencia: {competition}")
    subprocess.run([
        "uvx", "kaggle", "competitions", "download",
        "-c", competition,
        "-p", raw_path, "--unzip"
    ], check=True)

elif source == "csv":
    csv_path = config["data"]["csv_path"]
    print(f"Copiando datos desde: {csv_path}")
    import shutil
    for f in Path(csv_path).glob("*.csv"):
        shutil.copy(f, raw_path)

else:
    print(f"Fuente no soportada: {source}")
    print("Fuentes válidas: kaggle, kaggle_competition, csv")
    exit(1)

# Listar archivos descargados
files = list(Path(raw_path).glob("*"))
print(f"\n✅ Ingesta completada. {len(files)} archivos en {raw_path}:")
for f in files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"   {f.name} ({size_mb:.1f} MB)")

print(f"\n👤 SIGUIENTE: Revisa los archivos descargados en {raw_path}")
print(f"   Ajusta config.yaml si necesitas cambiar columnas o target.")
print(f"   Cuando estés listo: uv run python scripts/02_eda.py")
```

### Alternativa: Usar el MCP de Kaggle

También puedes buscar y descargar datasets directamente desde el asistente usando el MCP de Kaggle:

```
# Buscar datasets
search_datasets(request={"query": "heart disease classification"})

# Descargar dataset
download_dataset(request={"ownerSlug": "username", "datasetSlug": "dataset-name"})
```

---

## Etapa 6: Script 06_publish_dataset.py — Publicar Dataset Curado

Publica el dataset procesado (resultado de Feature Engineering) a Kaggle, generando automáticamente una Data Card.

### Estructura del Script

```python
#!/usr/bin/env python3
"""06_publish_dataset.py — Publicar dataset curado a Kaggle.

Genera Data Card, prepara metadata y sube dataset procesado.
Artefactos: cards/DATA_CARD.md, data/processed/ → Kaggle

Uso: uv run python scripts/06_publish_dataset.py
"""
import yaml
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Cargar configuración
with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("cards").mkdir(parents=True, exist_ok=True)

username = config["publish"]["kaggle_username"]
dataset_slug = config["publish"]["kaggle_dataset_slug"]
project_name = config["project"]["name"]
description = config["project"]["description"]

# Cargar métricas del EDA para la Data Card
eda_report = {}
eda_path = Path("outputs/reports/eda_report.json")
if eda_path.exists():
    with open(eda_path) as f:
        eda_report = json.load(f)

# --- Generar DATA_CARD.md ---
data_card = f"""# Data Card: {project_name}

## Descripción del Dataset
{description}

## Información General
- **Autor**: {config['project']['author']}
- **Fecha de creación**: {datetime.now().strftime('%Y-%m-%d')}
- **Fuente original**: {config['data'].get('kaggle_ref', 'N/A')}
- **Licencia**: Apache-2.0

## Composición del Dataset
- **Filas**: {eda_report.get('shape', {}).get('rows', 'N/A')}
- **Columnas**: {eda_report.get('shape', {}).get('columns', 'N/A')}
- **Variables numéricas**: {', '.join(eda_report.get('numeric_columns', []))}
- **Variables categóricas**: {', '.join(eda_report.get('categorical_columns', []))}
- **Variable target**: {config['features']['target']}

## Preprocesamiento Aplicado
- Imputación de valores faltantes (mediana para numéricos, moda para categóricos)
- Escalado StandardScaler para variables numéricas
- One-Hot Encoding para variables categóricas
- Separación train/test con ratio {config['data']['test_size']}

## Valores Faltantes (Dataset Original)
{_format_missing_values(eda_report.get('missing_values', {}))}

## Uso Previsto
Este dataset curado está diseñado para entrenamiento de modelos de machine learning.
Las transformaciones aplicadas están documentadas y el preprocessor está disponible
como artefacto separado (preprocessor.joblib).

## Limitaciones y Sesgos
- [COMPLETAR: Describir limitaciones conocidas del dataset]
- [COMPLETAR: Describir posibles sesgos en los datos]

## Cómo Citar
```
@dataset{{{dataset_slug},
  author = {{{config['project']['author']}}},
  title = {{{project_name}}},
  year = {{{datetime.now().year}}},
  publisher = {{Kaggle}}
}}
```
"""

with open("cards/DATA_CARD.md", "w") as f:
    f.write(data_card)

print("📄 Data Card generada: cards/DATA_CARD.md")

# --- Preparar metadata de Kaggle ---
publish_dir = Path("data/publish_dataset")
publish_dir.mkdir(parents=True, exist_ok=True)

# Copiar archivos procesados
import shutil
for csv_file in Path("data/processed").glob("*.csv"):
    shutil.copy(csv_file, publish_dir)
shutil.copy("cards/DATA_CARD.md", publish_dir / "DATA_CARD.md")

# Crear dataset-metadata.json
metadata = {
    "title": project_name,
    "id": f"{username}/{dataset_slug}",
    "licenses": [{"name": "Apache 2.0"}],
    "keywords": config.get("publish", {}).get("keywords", ["machine-learning", "curated"]),
    "resources": []
}

for f in publish_dir.glob("*.csv"):
    metadata["resources"].append({
        "path": f.name,
        "description": f"Dataset procesado: {f.name}"
    })

with open(publish_dir / "dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n👤 REVISIÓN REQUERIDA:")
print(f"   1. Revisa cards/DATA_CARD.md y completa las secciones marcadas con [COMPLETAR]")
print(f"   2. Revisa los archivos en data/publish_dataset/")
print(f"   3. Cuando estés listo, ejecuta el upload:")
print(f"      uvx kaggle datasets create -p data/publish_dataset/ --dir-mode zip")
print(f"   O para actualizar versión existente:")
print(f"      uvx kaggle datasets version -p data/publish_dataset/ -m 'descripción del cambio'")


def _format_missing_values(missing_dict):
    if not missing_dict:
        return "No se detectaron valores faltantes."
    lines = []
    for col, pct in missing_dict.items():
        lines.append(f"- **{col}**: {pct:.1f}%")
    return "\n".join(lines)
```

---

## Etapa 7: Script 07_publish_model.py — Publicar Modelo a Kaggle

Publica el modelo entrenado a Kaggle. Hay dos opciones: como **Kaggle Model** (recomendado) o como **dataset**.

### Opción A: Publicar como Kaggle Model (Recomendado)

Usa la API de Kaggle Models. Requiere dos pasos: crear el modelo y luego crear una instancia con los archivos.

**Paso 1: Crear el modelo (una sola vez)**

Esto se puede hacer con el MCP de Kaggle:
```
create_model(request={
    "ownerSlug": "tu-username",
    "slug": "mi-modelo",
    "title": "Mi Modelo de Clasificación",
    "description": "Descripción del modelo...",
    "isPrivate": false
})
```

**Paso 2: Crear `model-instance-metadata.json`**

Este archivo es OBLIGATORIO para subir archivos al modelo con el CLI:

```json
{
  "ownerSlug": "tu-username",
  "modelSlug": "mi-modelo",
  "framework": "other",
  "instanceSlug": "default",
  "overview": "Descripción de esta variación del modelo.",
  "licenseName": "Apache 2.0",
  "fineTunable": false
}
```

**IMPORTANTE**: `"licenseName"` debe ser `"Apache 2.0"` (con espacio). Valores como `"Apache-2.0"` o `"CC0-1.0"` causan el error `"Specify an existing license"`.

Valores válidos de `framework`: `"tensorflow"`, `"pytorch"`, `"jax"`, `"other"`.

**Paso 3: Subir archivos con el CLI**

```bash
# Estructura de la carpeta
kaggle-model/
├── model-instance-metadata.json   # OBLIGATORIO
├── model.joblib                   # Modelo serializado
├── label_encoder.joblib           # Encoder (si aplica)
├── model_info.json                # Metadata con métricas
└── notebook.ipynb                 # Notebook reproducible (opcional)

# Subir
uv run kaggle models instances create -p kaggle-model/
```

**NOTA**: El MCP de Kaggle NO puede subir archivos a model instances. Solo el CLI puede hacerlo.

**Descargar modelo desde código (para app de inferencia)**

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# Firma: model_instance_version_download(model_instance_version, path, ...)
# model_instance_version es UN SOLO STRING con formato: "owner/model/framework/variation/version"
api.model_instance_version_download(
    "tu-username/mi-modelo/Other/default/1",
    path="model_artifacts/",
    untar=True,
)
```

**CUIDADO con la firma**: El método `model_instance_version_download` recibe UN SOLO string con el path completo, NO argumentos separados. Esto es un error común.

### Opción B: Publicar como Dataset en Kaggle

Alternativa más simple. El modelo se publica como dataset, lo cual facilita la descarga con `api.dataset_download_files()`.

**Patrón probado** (basado en implementación real de salary-predictor):

### Estructura del Script

```python
#!/usr/bin/env python3
"""07_publish_model.py — Publicar modelo entrenado a Kaggle.

Publica el modelo como dataset en Kaggle. Esto permite que la app de
inferencia en HF Spaces lo descargue fácilmente con la API de Kaggle.

Artefactos: cards/MODEL_CARD.md, models/ → Kaggle

Uso: uv run python scripts/07_publish_model.py
"""
import yaml
import json
import shutil
import subprocess
import os
from pathlib import Path
from datetime import datetime

# Cargar configuración
with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("cards").mkdir(parents=True, exist_ok=True)

username = config["publish"]["kaggle_username"]
model_slug = config["publish"]["kaggle_model_slug"]
project_name = config["project"]["name"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar métricas de validación
metrics = {}
metrics_path = Path("outputs/metrics/validation.json")
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

# Cargar metadata de entrenamiento
train_meta = {}
train_meta_path = Path("outputs/metrics/train_metadata.json")
if train_meta_path.exists():
    with open(train_meta_path) as f:
        train_meta = json.load(f)

# --- Generar MODEL_CARD.md ---
model_card = f"""# Model Card: {project_name}

## Información del Modelo
- **Tipo**: {train_meta.get('model_type', config['model']['type'])}
- **Framework**: {_detect_framework(config['model']['type'])}
- **Autor**: {config['project']['author']}
- **Fecha de entrenamiento**: {train_meta.get('timestamp', datetime.now().isoformat())}
- **Formato de serialización**: {config['model'].get('output_format', 'joblib')}

## Uso Previsto
- **Tarea**: {'Clasificación' if config['training']['scoring'] in ('accuracy', 'f1', 'roc_auc') else 'Regresión'}
- **Variable target**: {config['features']['target']}
- **Dominio**: [COMPLETAR: Describir el dominio de aplicación]

## Datos de Entrenamiento
- **Fuente**: {config['data'].get('kaggle_ref', 'N/A')}
- **Samples de entrenamiento**: {train_meta.get('n_samples', 'N/A')}
- **Features**: {train_meta.get('n_features', 'N/A')}
- **Preprocesamiento**: StandardScaler + OneHotEncoder (ver preprocessor.joblib)

## Métricas de Evaluación
{_format_metrics(metrics)}

## Hiperparámetros
```json
{json.dumps(train_meta.get('params', config['model']['params']), indent=2, default=str)}
```

## Features Utilizadas
{_format_features(train_meta.get('feature_names', []))}

## Limitaciones
- [COMPLETAR: Describir limitaciones del modelo]
- [COMPLETAR: Describir escenarios donde el modelo NO debe usarse]
- [COMPLETAR: Describir sesgos conocidos]

## Consideraciones Éticas
- [COMPLETAR: Describir consideraciones éticas relevantes]

## Cómo Usar

```python
import joblib

# Cargar modelo y preprocessor
model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

# Predecir
X_new_processed = preprocessor.transform(X_new)
predictions = model.predict(X_new_processed)
```

## Cómo Citar
```
@model{{{model_slug},
  author = {{{config['project']['author']}}},
  title = {{{project_name}}},
  year = {{{datetime.now().year}}},
  publisher = {{Kaggle}}
}}
```
"""

with open("cards/MODEL_CARD.md", "w") as f:
    f.write(model_card)

print("📄 Model Card generada: cards/MODEL_CARD.md")

# --- Preparar directorio de publicación ---
# Patrón probado: carpeta dedicada con solo los artefactos necesarios
publish_dir = Path("data/publish_model")
if publish_dir.exists():
    shutil.rmtree(publish_dir)
publish_dir.mkdir(parents=True)

# Copiar artefactos del modelo
for artifact in ["model.joblib", "preprocessor.joblib"]:
    src = Path("models") / artifact
    if src.exists():
        shutil.copy2(src, publish_dir)
        print(f"  ✓ {artifact}")

# Copiar metadata de métricas
if Path("outputs/metrics/validation.json").exists():
    shutil.copy2("outputs/metrics/validation.json", publish_dir / "metrics.json")
    print("  ✓ metrics.json")

# Copiar Model Card
shutil.copy2("cards/MODEL_CARD.md", publish_dir / "MODEL_CARD.md")
print("  ✓ MODEL_CARD.md")

# --- Crear dataset-metadata.json ---
# El modelo se publica como dataset en Kaggle (facilita descarga desde app de inferencia)
metadata = {
    "title": project_name,
    "id": f"{username}/{model_slug}",
    "licenses": [{"name": "Apache 2.0"}],
    "keywords": config.get("publish", {}).get("keywords", ["machine-learning"]) + [
        config["model"]["type"],
        "trained-model",
    ],
    "resources": [
        {
            "path": "model.joblib",
            "description": f"Modelo {config['model']['type']} serializado (scikit-learn/joblib)"
        },
        {
            "path": "preprocessor.joblib",
            "description": "Preprocessor (ColumnTransformer) para transformar datos de entrada"
        },
        {
            "path": "metrics.json",
            "description": "Métricas de evaluación del modelo"
        },
        {
            "path": "MODEL_CARD.md",
            "description": "Model Card con documentación completa del modelo"
        },
    ]
}

with open(publish_dir / "dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n📦 Publicando modelo a Kaggle como dataset...")

# Subir a Kaggle con --dir-mode zip
result = subprocess.run(
    ["kaggle", "datasets", "create", "-p", str(publish_dir), "--dir-mode", "zip"],
    capture_output=True, text=True
)

if result.returncode == 0:
    print(f"✅ Modelo publicado en Kaggle:")
    print(f"   https://www.kaggle.com/datasets/{username}/{model_slug}")
    print(f"\n👤 SIGUIENTE:")
    print(f"   1. Verifica el dataset en Kaggle")
    print(f"   2. Revisa que MODEL_CARD.md esté completa")
    print(f"   3. Cuando estés listo, despliega la app de inferencia:")
    print(f"      uv run python scripts/08_deploy_hf.py")
else:
    print(f"❌ Error: {result.stderr}")
    if "already exists" in result.stderr.lower() or "403" in result.stderr:
        print("\n💡 El dataset ya existe. Para actualizar, usa:")
        print(f"   kaggle datasets version -p {publish_dir} -m 'descripción del cambio' --dir-mode zip")
    else:
        print("\nAsegúrate de tener ~/.kaggle/kaggle.json con un token válido.")
        print("Descárgalo desde: https://www.kaggle.com/settings → API → Create New Token")


def _detect_framework(model_type):
    if model_type in ("tensorflow",):
        return "TensorFlow/Keras"
    elif model_type in ("pytorch",):
        return "PyTorch"
    elif model_type.startswith("xgboost"):
        return "XGBoost"
    return "scikit-learn"

def _format_metrics(metrics):
    if not metrics:
        return "No hay métricas disponibles."
    lines = []
    skip_keys = ("classification_report",)
    for key, value in metrics.items():
        if key in skip_keys:
            continue
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
    if lines:
        return "| Métrica | Valor |\n|---------|-------|\n" + "\n".join(lines)
    return "No hay métricas numéricas."

def _format_features(features):
    if not features:
        return "No disponible."
    return "\n".join(f"- `{f}`" for f in features[:20])
```

### Patrón clave: Modelo como Dataset en Kaggle

El modelo se publica como **dataset** (no como "Kaggle Model") porque:
1. La API de datasets es más simple para descarga programática
2. `kaggle datasets download` funciona directamente en la app de inferencia
3. Permite incluir artefactos adicionales (preprocessor, métricas, Model Card)
4. Usa `--dir-mode zip` para comprimir automáticamente

### Actualizar versión existente

```bash
# Primera publicación
kaggle datasets create -p data/publish_model/ --dir-mode zip

# Actualizaciones posteriores
kaggle datasets version -p data/publish_model/ -m "v2: Reentrenado con más datos" --dir-mode zip
```

---

## Kaggle CLI vs MCP: Referencia Rápida

| Operación | MCP de Kaggle | CLI (`uv run kaggle`) |
|-----------|--------------|----------------------|
| Buscar datasets | ✅ `search_datasets` | ✅ `kaggle datasets list` |
| Descargar datasets | ✅ `download_dataset` | ✅ `kaggle datasets download` |
| Info de dataset | ✅ `get_dataset_info` | ✅ `kaggle datasets metadata` |
| **Crear dataset nuevo** | ❌ | ✅ `kaggle datasets create -p carpeta/` |
| Actualizar metadata | ✅ `update_dataset_metadata` | ✅ `kaggle datasets metadata -p carpeta/` |
| Crear modelo (vacío) | ✅ `create_model` | ✅ `kaggle models create` |
| **Subir archivos a modelo** | ❌ | ✅ `kaggle models instances create -p carpeta/` |
| Guardar notebook | ✅ `save_notebook` | ✅ `kaggle kernels push` |
| Buscar notebooks | ✅ `search_notebooks` | ✅ `kaggle kernels list` |

**Credenciales requeridas:**
- MCP: Token KGAT (`KGAT_xxxx`) en mcp.json
- CLI: API Key (`~/.kaggle/kaggle.json` con username+key)

## Gotchas

- Los tokens de Kaggle expiran. Si recibes errores 401, regenera tu token en https://www.kaggle.com/settings
- El token KGAT (para MCP) y el API Key (para CLI) son credenciales DIFERENTES. Necesitas ambas
- Usa `uv run kaggle` o `uvx kaggle` en lugar de `kaggle` directamente para evitar instalar globalmente
- El tamaño máximo de un dataset en Kaggle es 100GB
- `"licenseName"` en `model-instance-metadata.json` debe ser `"Apache 2.0"` (con espacio), NO `"Apache-2.0"`
- `"licenses"` en `dataset-metadata.json` debe ser `[{"name": "Apache 2.0"}]` (con espacio)
- Siempre completa las secciones [COMPLETAR] en las cards antes de publicar
- Versiona siempre con mensajes descriptivos
- La Data Card y Model Card son documentos vivos: actualízalas con cada nueva versión
