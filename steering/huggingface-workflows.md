# Hugging Face Workflows

## ⚠️ Fuente Exclusiva de Datasets

**Esta versión del Power SOLO acepta Hugging Face Hub como fuente de datasets.**

- NO se soportan: Kaggle, URLs directas, archivos locales arbitrarios, APIs externas, bases de datos.
- Si el usuario pide un dataset de otra fuente, responder:
  > "Esta versión del Power solo soporta datasets de Hugging Face Hub. ¿Quieres que busque un dataset equivalente en HF? Puedo buscar con `hf datasets ls --search 'tema'`."
- El campo `source` en `config.yaml` DEBE ser `"huggingface"`.

---

## Requisito Previo: Dataset Explícito

**ANTES de generar cualquier código de ingesta**, el usuario DEBE proporcionar:
- **Referencia del dataset**: formato `usuario/nombre-dataset` (ejemplo: `scikit-learn/iris`, `ucirvine/heart-disease`)
- **Nombre del usuario** que publica en Hugging Face

Si el usuario no ha proporcionado esta información, PREGUNTAR explícitamente:
> "¿Cuál es el dataset de Hugging Face que quieres usar? Necesito el formato `usuario/nombre-dataset` (ejemplo: `scikit-learn/iris`)."

NUNCA inventar o asumir un dataset. El `config.yaml` se genera con estos datos.

## Convención de Nombres (OBLIGATORIO)

Los slugs y nombres en `config.yaml` DEBEN derivarse del nombre real del proyecto y dataset del usuario. **NUNCA usar nombres genéricos como "dummy", "test", "example", "mi-proyecto", "my-dataset".**

### Reglas de Naming:
- `project.name`: Nombre descriptivo del proyecto (ejemplo: `"iris-classifier"`, `"salary-predictor"`)
- `publish.hf_dataset_slug`: Derivar del nombre del dataset + sufijo descriptivo (ejemplo: `"iris-curated"`, `"heart-disease-processed"`)
- `publish.hf_model_slug`: Derivar del proyecto + tipo de modelo (ejemplo: `"iris-xgboost"`, `"salary-linear-regression"`)
- `publish.hf_space_name`: Derivar del proyecto + propósito (ejemplo: `"iris-classifier-app"`, `"salary-predictor"`)

### Template de config.yaml

```yaml
project:
  name: "<nombre-descriptivo-del-proyecto>"  # Derivar del dataset/problema
  description: "<descripción breve del objetivo>"
  author: "<nombre del usuario>"
  version: "0.1.0"

data:
  source: "huggingface"
  hf_dataset_ref: "<usuario/nombre-dataset>"  # Exacto como está en HF Hub
  raw_path: "data/raw"
  processed_path: "data/processed"

features:
  target: "<nombre-columna-objetivo>"
  numeric: []   # Se llena después del EDA
  categorical: []  # Se llena después del EDA

publish:
  hf_username: "<usuario-de-huggingface>"
  hf_dataset_slug: "<nombre-proyecto>-curated"  # NUNCA "dummy" ni genéricos
  hf_model_slug: "<nombre-proyecto>-<tipo-modelo>"  # Ejemplo: "iris-xgboost"
  hf_space_name: "<nombre-proyecto>-app"  # Ejemplo: "salary-predictor"
```

**Ejemplo concreto** para un proyecto con el dataset `scikit-learn/iris`:
```yaml
project:
  name: "iris-classifier"
  description: "Clasificador multiclase de especies de Iris"
  author: "gusdelact"
  version: "0.1.0"

data:
  source: "huggingface"
  hf_dataset_ref: "scikit-learn/iris"
  raw_path: "data/raw"
  processed_path: "data/processed"

features:
  target: "species"
  numeric: ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  categorical: []

publish:
  hf_username: "gusdelact"
  hf_dataset_slug: "iris-curated"
  hf_model_slug: "iris-xgboost"
  hf_space_name: "iris-classifier-app"
```

## Buscar Datasets en HF Hub

```bash
# Buscar datasets por nombre
hf datasets ls --search "iris"

# Buscar con filtros
hf datasets ls --search "classification" --sort downloads --limit 10

# Ver info de un dataset específico
hf datasets info scikit-learn/iris
```

---

## Etapa 1: Script 01_ingest.py — Ingesta de Datos

Descarga datos desde Hugging Face Hub (u otra fuente) y los guarda en `data/raw/`. NUNCA modifica los datos originales.

### Estructura del Script

```python
#!/usr/bin/env python3
"""01_ingest.py — Ingesta de datos desde Hugging Face Hub u otra fuente.

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

if source == "huggingface":
    hf_ref = config["data"]["hf_dataset_ref"]
    print(f"Descargando dataset de Hugging Face: {hf_ref}")

    # Usar hf CLI para descargar
    subprocess.run([
        "hf", "download", hf_ref,
        "--repo-type", "dataset",
        "--local-dir", raw_path
    ], check=True)

else:
    print(f"⚠️ Fuente no soportada: {source}")
    print("Esta versión del Power SOLO acepta 'huggingface' como fuente.")
    print("Configura config.yaml con source: huggingface y hf_dataset_ref: usuario/dataset")
    exit(1)

# Listar archivos descargados
files = list(Path(raw_path).glob("*"))
print(f"\n✅ Ingesta completada. {len(files)} archivos en {raw_path}:")
for f in files:
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {f.name} ({size_mb:.1f} MB)")

print(f"\n👤 SIGUIENTE: Revisa los archivos descargados en {raw_path}")
print(f"   Ajusta config.yaml si necesitas cambiar columnas o target.")
print(f"   Cuando estés listo: uv run python scripts/02_eda.py")
```

### Descargar solo archivos específicos

```bash
# Solo archivos CSV
hf download usuario/dataset --repo-type dataset --include "*.csv" --local-dir data/raw/

# Solo un archivo
hf download usuario/dataset train.csv --repo-type dataset --local-dir data/raw/

# Ver qué se descargaría (dry-run)
hf download usuario/dataset --repo-type dataset --dry-run
```

### ⚠️ Validación del nombre del target (OBLIGATORIO post-ingesta)

Después de descargar el dataset, el agente DEBE verificar que el nombre de la columna target en `config.yaml` coincide **exactamente** (case-sensitive) con el nombre real en el dataset descargado. Los datasets de HF Hub pueden tener columnas con mayúsculas (`Species`, `Target`, `Class`) que no coinciden con lo que el agente asumió al generar el `config.yaml`.

**Patrón de validación** (incluir al final de `01_ingest.py` o al inicio de `02_eda.py`):

```python
import pandas as pd

# Cargar una muestra para verificar columnas
sample = pd.read_csv(f"{raw_path}/train.csv", nrows=5)  # o .parquet
target_in_config = config["features"]["target"]

if target_in_config not in sample.columns:
    # Buscar coincidencia case-insensitive
    matches = [c for c in sample.columns if c.lower() == target_in_config.lower()]
    if matches:
        print(f"⚠️  Target '{target_in_config}' no encontrado, pero existe '{matches[0]}'")
        print(f"   Corrigiendo config.yaml: target = '{matches[0]}'")
        # El agente DEBE corregir config.yaml antes de continuar
    else:
        print(f"❌ Target '{target_in_config}' no existe en el dataset.")
        print(f"   Columnas disponibles: {list(sample.columns)}")
```

Si hay mismatch, **corregir `config.yaml` inmediatamente** antes de continuar con el EDA. No asumir que el nombre en minúsculas es correcto.

---

## Etapa 6: Script 06_publish_dataset.py — Publicar Dataset Curado a HF Hub

Publica el dataset procesado (resultado de Feature Engineering) a Hugging Face Hub, generando automáticamente una Data Card.

### Estructura del Script

```python
#!/usr/bin/env python3
"""06_publish_dataset.py — Publicar dataset curado a Hugging Face Hub.

Genera Data Card, prepara archivos y sube dataset procesado usando hf CLI.
Artefactos: cards/DATA_CARD.md, data/processed/ → HF Hub

Uso: uv run python scripts/06_publish_dataset.py
"""
import yaml
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Cargar configuración
with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("cards").mkdir(parents=True, exist_ok=True)

username = config["publish"]["hf_username"]
dataset_slug = config["publish"]["hf_dataset_slug"]
repo_id = f"{username}/{dataset_slug}"
project_name = config["project"]["name"]
description = config["project"]["description"]

# Cargar métricas del EDA para la Data Card
eda_report = {}
eda_path = Path("outputs/reports/eda_report.json")
if eda_path.exists():
    with open(eda_path) as f:
        eda_report = json.load(f)

# --- Generar DATA_CARD.md (será el README.md del dataset en HF) ---
data_card = f"""---
license: apache-2.0
task_categories:
  - tabular-classification
language:
  - en
size_categories:
  - n<1K
---

# {project_name}

## Descripción del Dataset
{description}

## Información General
- **Autor**: {config['project']['author']}
- **Fecha de creación**: {datetime.now().strftime('%Y-%m-%d')}
- **Fuente original**: {config['data'].get('hf_dataset_ref', 'N/A')}
- **Licencia**: Apache-2.0

## Composición del Dataset
- **Filas**: {eda_report.get('shape', {}).get('rows', 'N/A')}
- **Columnas**: {eda_report.get('shape', {}).get('columns', 'N/A')}
- **Variable target**: {config['features']['target']}

## Preprocesamiento Aplicado
- Imputación de valores faltantes (mediana para numéricos, moda para categóricos)
- Escalado StandardScaler para variables numéricas
- One-Hot Encoding para variables categóricas
- Separación train/test con ratio {config['data']['test_size']}

## Uso Previsto
Este dataset curado está diseñado para entrenamiento de modelos de machine learning.

## Limitaciones y Sesgos
- [COMPLETAR: Describir limitaciones conocidas del dataset]
- [COMPLETAR: Describir posibles sesgos en los datos]

## Cómo Citar
```
@dataset{{{dataset_slug},
  author = {{{config['project']['author']}}},
  title = {{{project_name}}},
  year = {{{datetime.now().year}}},
  publisher = {{Hugging Face}}
}}
```
"""

with open("cards/DATA_CARD.md", "w") as f:
    f.write(data_card)

print("📄 Data Card generada: cards/DATA_CARD.md")

# --- Preparar directorio de publicación ---
publish_dir = Path("data/publish_dataset")
if publish_dir.exists():
    shutil.rmtree(publish_dir)
publish_dir.mkdir(parents=True)

# Copiar archivos procesados
for csv_file in Path("data/processed").glob("*.csv"):
    shutil.copy(csv_file, publish_dir)

# Copiar Data Card como README.md (HF lo usa como portada del dataset)
shutil.copy("cards/DATA_CARD.md", publish_dir / "README.md")

print(f"\n📦 Archivos preparados en {publish_dir}/")

# --- Crear repo y subir ---
print(f"\nCreando repositorio de dataset: {repo_id}")
subprocess.run([
    "hf", "repos", "create", repo_id,
    "--repo-type", "dataset", "--exist-ok"
], check=True)

print(f"Subiendo archivos...")
result = subprocess.run([
    "hf", "upload", repo_id,
    str(publish_dir), ".",
    "--repo-type", "dataset",
    "--commit-message", f"Dataset curado v{config['project']['version']}"
], capture_output=True, text=True)

if result.returncode == 0:
    print(f"\n✅ Dataset publicado en:")
    print(f"   https://huggingface.co/datasets/{repo_id}")
    print(f"\n👤 SIGUIENTE:")
    print(f"   1. Verifica el dataset en HF Hub")
    print(f"   2. Completa las secciones [COMPLETAR] en la Data Card")
    print(f"   3. Cuando estés listo: uv run python scripts/07_publish_model.py")
else:
    print(f"❌ Error: {result.stderr}")
    print("Asegúrate de estar autenticado: hf auth login")
```

---

## Etapa 7: Script 07_publish_model.py — Publicar Modelo a HF Hub

Publica el modelo entrenado a Hugging Face Hub como repositorio de modelo.

### Estructura del Script

```python
#!/usr/bin/env python3
"""07_publish_model.py — Publicar modelo entrenado a Hugging Face Hub.

Genera Model Card, prepara archivos y sube modelo usando hf CLI.
Artefactos: cards/MODEL_CARD.md, models/ → HF Hub

Uso: uv run python scripts/07_publish_model.py
"""
import yaml
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Cargar configuración
with open("config.yaml") as f:
    config = yaml.safe_load(f)

Path("cards").mkdir(parents=True, exist_ok=True)

username = config["publish"]["hf_username"]
model_slug = config["publish"]["hf_model_slug"]
repo_id = f"{username}/{model_slug}"
project_name = config["project"]["name"]

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

# --- Detectar framework ---
def detect_framework(model_type):
    if model_type in ("tensorflow",):
        return "TensorFlow/Keras"
    elif model_type in ("pytorch",):
        return "PyTorch"
    elif "xgboost" in model_type:
        return "XGBoost"
    return "scikit-learn"

# --- Formatear métricas ---
def format_metrics(metrics):
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

# --- Generar MODEL_CARD.md (será el README.md del modelo en HF) ---
model_card = f"""---
license: apache-2.0
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - {config['model']['type']}
  - tabular
  - classification
---

# {project_name}

## Información del Modelo
- **Tipo**: {train_meta.get('model_type', config['model']['type'])}
- **Framework**: {detect_framework(config['model']['type'])}
- **Autor**: {config['project']['author']}
- **Fecha de entrenamiento**: {train_meta.get('timestamp', datetime.now().isoformat())}
- **Formato de serialización**: {config['model'].get('output_format', 'joblib')}

## Uso Previsto
- **Tarea**: {'Clasificación' if config['training']['scoring'] in ('accuracy', 'f1', 'roc_auc') else 'Regresión'}
- **Variable target**: {config['features']['target']}

## Datos de Entrenamiento
- **Fuente**: {config['data'].get('hf_dataset_ref', 'N/A')}
- **Samples de entrenamiento**: {train_meta.get('n_samples', 'N/A')}
- **Features**: {train_meta.get('n_features', 'N/A')}

## Métricas de Evaluación
{format_metrics(metrics)}

## Hiperparámetros
```json
{json.dumps(train_meta.get('params', config['model']['params']), indent=2, default=str)}
```

## Cómo Usar

```python
import joblib
from huggingface_hub import hf_hub_download

# Descargar modelo
model_path = hf_hub_download("{repo_id}", "model.joblib")
model = joblib.load(model_path)

# Predecir
predictions = model.predict(X_new)
```

## Limitaciones
- [COMPLETAR: Describir limitaciones del modelo]
- [COMPLETAR: Describir escenarios donde el modelo NO debe usarse]
"""

with open("cards/MODEL_CARD.md", "w") as f:
    f.write(model_card)

print("📄 Model Card generada: cards/MODEL_CARD.md")

# --- Preparar directorio de publicación ---
publish_dir = Path("data/publish_model")
if publish_dir.exists():
    shutil.rmtree(publish_dir)
publish_dir.mkdir(parents=True)

# Copiar artefactos del modelo
for artifact in ["model.joblib", "label_encoder.joblib", "preprocessor.joblib"]:
    src = Path("models") / artifact
    if src.exists():
        shutil.copy2(src, publish_dir)
        print(f"  ✓ {artifact}")

# Copiar metadata
if Path("outputs/metrics/validation.json").exists():
    shutil.copy2("outputs/metrics/validation.json", publish_dir / "metrics.json")
    print("  ✓ metrics.json")

if Path("outputs/metrics/train_metadata.json").exists():
    shutil.copy2("outputs/metrics/train_metadata.json", publish_dir / "train_metadata.json")
    print("  ✓ train_metadata.json")

# Copiar Model Card como README.md
shutil.copy2("cards/MODEL_CARD.md", publish_dir / "README.md")
print("  ✓ README.md")

# --- Crear repo y subir ---
print(f"\nCreando repositorio de modelo: {repo_id}")
subprocess.run([
    "hf", "repos", "create", repo_id, "--exist-ok"
], check=True)

print(f"Subiendo archivos...")
result = subprocess.run([
    "hf", "upload", repo_id,
    str(publish_dir), ".",
    "--commit-message", f"Modelo {config['model']['type']} v{config['project']['version']}"
], capture_output=True, text=True)

if result.returncode == 0:
    print(f"\n✅ Modelo publicado en:")
    print(f"   https://huggingface.co/{repo_id}")
    print(f"\n👤 SIGUIENTE:")
    print(f"   1. Verifica el modelo en HF Hub")
    print(f"   2. Completa las secciones [COMPLETAR] en la Model Card")
    print(f"   3. Cuando estés listo: uv run python scripts/08_deploy_hf.py")
else:
    print(f"❌ Error: {result.stderr}")
    print("Asegúrate de estar autenticado: hf auth login")
```

---

## Descargar Modelo desde HF Hub (para app de inferencia)

La app de inferencia descarga el modelo desde HF Hub al iniciar:

```python
from huggingface_hub import hf_hub_download
import joblib

# Descargar archivos del modelo
model_path = hf_hub_download("usuario/mi-modelo", "model.joblib")
model = joblib.load(model_path)

# Si hay label encoder
encoder_path = hf_hub_download("usuario/mi-modelo", "label_encoder.joblib")
encoder = joblib.load(encoder_path)
```

O usando el CLI:

```bash
hf download usuario/mi-modelo --local-dir ./model_cache
```

---

## Referencia Rápida de Comandos HF CLI

| Operación | Comando |
|-----------|---------|
| Buscar datasets | `hf datasets ls --search "nombre"` |
| Info de dataset | `hf datasets info usuario/dataset` |
| Descargar dataset | `hf download usuario/dataset --repo-type dataset --local-dir ./data` |
| Crear repo dataset | `hf repos create usuario/dataset --repo-type dataset` |
| Subir dataset | `hf upload usuario/dataset ./carpeta . --repo-type dataset` |
| Crear repo modelo | `hf repos create usuario/modelo` |
| Subir modelo | `hf upload usuario/modelo ./carpeta .` |
| Descargar modelo | `hf download usuario/modelo --local-dir ./modelo` |
| Crear Space | `hf repos create usuario/app --repo-type space --space-sdk gradio` |
| Subir a Space | `hf upload usuario/app ./carpeta . --repo-type space` |
| Ver quién soy | `hf auth whoami` |

## Gotchas

- Asegúrate de estar autenticado (`hf auth login`) antes de subir contenido
- El token necesita permisos de **escritura** para crear repos y subir archivos
- El README.md con YAML frontmatter es la "card" del dataset/modelo en HF Hub
- Usa `--exist-ok` en `hf repos create` para evitar errores si el repo ya existe
- Para actualizar un dataset/modelo existente, simplemente vuelve a hacer `hf upload` — sobreescribe los archivos
- La versión de `sdk_version` en el README del Space debe coincidir con una versión real de Gradio
- **Compatibilidad `huggingface-hub` ↔ `gradio`** — `gradio>=5.31` exige `huggingface-hub>=0.28.1`. NO pinear `huggingface-hub==0.25.0` en `app_inference/requirements.txt` ni en el `pyproject.toml` del proyecto: rompe el resolver de pip al construir el Space (`ResolutionImpossible`). Usar siempre rango `>=0.28.1`. Ver `gradio-interfaces.md` §"Errores Documentados en HF Spaces" → Error 2.
