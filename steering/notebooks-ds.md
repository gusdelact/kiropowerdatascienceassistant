# Notebooks Jupyter en proyectos de Data Science

Esta convención aplica a cualquier proyecto de DS / ML del workspace que entrene un modelo y lo entregue como artefacto reutilizable (`model.joblib` + metadata). Define cómo se estructuran los notebooks que acompañan al pipeline.

El objetivo es que un alumno (o un yo del futuro) pueda:
1. **Abrir el notebook en Google Colab con un solo clic en el badge "Open in Colab"** y reproducir el entrenamiento de extremo a extremo sin clonar el repo, sin `lib/`, sin `config.yaml`, sin secretos.
2. Usar el modelo ya entrenado para inferencia local o en Colab sin reentrenar nada, sin credenciales.

> **Destino primario de ejecución: Google Colab.** Cualquier notebook que genere este Power debe abrirse y correr de extremo a extremo en una runtime fresca de Colab (CPU, sin GPU, sin secretos). El soporte para Jupyter local se obtiene gratis si la Regla 5 se respeta (detección `IN_COLAB`).

## Regla 0: formato JSON correcto del `.ipynb`

Los notebooks generados **programáticamente** (con `nbformat`, scripts de generación, plantillas, etc.) deben respetar el formato canónico de Jupyter para que el `.ipynb` sea editable y diff-friendly:

- `nbformat: 4`, `nbformat_minor: 5`.
- En cada celda, el campo `source` es un **array de strings**, donde **cada línea de código es un elemento separado del array, terminando en `\n`** (excepto la última, que puede no tener `\n`). NUNCA concatenar todo el código en un único string con `\n` embebidos.

✅ Correcto:

```json
"source": [
  "import pandas as pd\n",
  "import numpy as np\n",
  "\n",
  "df = pd.read_csv('data.csv')\n",
  "df.head()"
]
```

❌ Incorrecto (rompe el diff de git, complica la edición y es lo que generan algunos scripts naïve):

```json
"source": ["import pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('data.csv')\ndf.head()"]
```

Cuando estés generando notebooks desde Python, usa `nbformat.v4.new_code_cell(source=lines)` con `lines` ya como **lista de strings terminadas en `\n`**, o pasa el `source` como lista al construir el dict de celda manualmente. Si abres el archivo en un editor y ves un único string gigantesco, está mal — regenera.

> Esta regla aplica solo a notebooks generados por código. Los notebooks creados directamente en Jupyter/VS Code ya cumplen el formato.

## Regla 1: dos notebooks separados, nunca uno solo

Cada proyecto debe tener al menos estos dos notebooks en su raíz:

- `01_entrenamiento.ipynb` — pipeline completo: carga, EDA, preprocesamiento, entrenamiento, evaluación, persistencia.
- `02_inferencia.ipynb` — carga el artefacto ya entrenado y muestra cómo predecir.

Nunca mezclar ambos flujos en un solo notebook. Razón: la celda de entrenamiento se vuelve a ejecutar sin querer, sobreescribe el `.joblib` y rompe la trazabilidad del modelo desplegado.

Si el proyecto tiene varios modelos candidatos (p. ej. baseline + modelo final), el patrón se generaliza así:

- `01_entrenamiento.ipynb`
- `02_inferencia.ipynb`
- `03_comparacion_modelos.ipynb` (opcional)

## Regla 2: estructura mínima de `01_entrenamiento.ipynb`

Secciones obligatorias, en este orden, cada una con celda markdown encabezando:

1. **Badge "Open in Colab"** — primera celda markdown del notebook, con el link directo al `.ipynb` publicado en HF Hub (ver Regla 7).
2. **Setup** — instala dependencias faltantes (Regla 8), fija semilla, define `PROJECT_ROOT` (Colab-aware, ver Regla 5), imprime versiones de librerías clave.
3. **Carga de datos** — descarga el dataset desde Hugging Face Hub con `hf_hub_download` (datasets públicos, sin credenciales). NO usar archivos locales en `data/`: los notebooks deben ser autocontenidos (Regla 9) y un Colab fresco no tiene esa carpeta.
4. **EDA** — gráficas inline con `%matplotlib inline`. Persiste también los `.png` en `OUTPUTS_DIR / "eda"` para que el dashboard los pueda consumir si el notebook se corre localmente.
5. **Preprocesamiento** — encoders, escalado, split estratificado. Cualquier transformador entrenado (LabelEncoder, StandardScaler, OneHotEncoder, ColumnTransformer, etc.) se persiste como artefacto aparte.
6. **Entrenamiento** — modelo final con sus hiperparámetros. Si se hizo búsqueda, dejar al menos una celda comentada o un markdown indicando cómo se llegó a esos hiperparámetros.
7. **Evaluación** — métricas en test, matriz de confusión y/o residuos, curvas relevantes. Las gráficas se guardan en `OUTPUTS_DIR / "modelos" / <nombre_modelo>`.
8. **Persistencia** — guarda `model.joblib`, los encoders/scalers como artefactos separados, y el `model_info.json` (ver Regla 4).
9. **Descarga de artefactos (solo Colab)** — celda final que en Colab dispara `google.colab.files.download(...)` para que el alumno se lleve los `.joblib` y el `model_info.json` a su máquina. En Jupyter local esta celda es no-op.
10. **Resumen final** — celda que imprime las rutas, tamaños de los artefactos generados y las métricas clave. Sirve como recibo de la corrida.

## Regla 3: estructura mínima de `02_inferencia.ipynb`

Debe correr de principio a fin **en Colab fresco** (sin clonar el repo) o localmente, sin volver a entrenar nada. Solo necesita acceso a internet para descargar el modelo desde HF Hub la primera vez.

Secciones:

1. **Badge "Open in Colab"** — primera celda markdown.
2. **Setup ligero** — instala solo lo que falte (típicamente `huggingface-hub`; ver Regla 8), imports mínimos para predecir (`joblib`, `pandas`, `pathlib`, librería del modelo).
3. **Descarga de artefactos desde HF Hub** — `hf_hub_download` del repo del modelo (`{HF_USER}/{MODEL_SLUG}`). Cachea en `~/.cache/huggingface/`, idempotente. NO depender de que los archivos ya estén en disco: en un Colab fresco no lo están.
4. **Verificación de compatibilidad** — lee `model_info.json` y compara versiones de librerías clave contra el entorno actual. Si hay mismatch, imprime un warning claro pero no falla. Esto evita el dolor típico de "el `.joblib` no carga porque cambió la versión de sklearn entre Colab y el entrenamiento".
5. **Carga de artefactos** — `model.joblib`, encoders/scalers, `model_info.json`.
6. **Función `predict`** — encapsula validación de input → transform → predict → decode. Firma sugerida:
   ```python
   def predict(input_dict: dict) -> dict:
       """Recibe un dict con las features crudas, devuelve la predicción decodificada."""
   ```
7. **Ejemplo feliz** — un input válido y su predicción.
8. **Ejemplo de error** — un input mal formado (feature faltante, categoría desconocida) para mostrar el manejo de errores.
9. **Bonus opcional: comparación con endpoint remoto** — si el modelo está desplegado (HF Space, API, etc.), hacer la misma predicción contra el endpoint y mostrar que coincide con la local. Esto valida que el deploy es fiel al artefacto.

## Regla 4: `model_info.json` obligatorio

Cada entrenamiento debe producir un `model_info.json` junto al `.joblib`. Campos mínimos:

```json
{
  "model_name": "iris-xgboost",
  "model_type": "XGBClassifier",
  "trained_at": "2026-05-17T12:34:56",
  "python_version": "3.11.9",
  "library_versions": {
    "scikit-learn": "1.5.2",
    "xgboost": "2.1.1",
    "pandas": "2.2.3"
  },
  "feature_order": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "target_classes": ["setosa", "versicolor", "virginica"],
  "metrics": {
    "accuracy_test": 0.967,
    "f1_macro_test": 0.965
  },
  "artifacts": {
    "model": "model.joblib",
    "label_encoder": "label_encoder.joblib"
  },
  "dataset": {
    "source": "kaggle:uciml/iris",
    "n_rows": 150,
    "hash_sha256": "abc123..."
  }
}
```

`feature_order` y `target_classes` son críticos: el notebook de inferencia los usa para construir el input correctamente y decodificar la salida sin depender de cómo esté ordenado el dict que pase el usuario.

## Regla 5: rutas ancladas a `PROJECT_ROOT` (Colab-aware)

Jupyter abre el notebook en el directorio que se le antoja según el cliente (VS Code, Kiro, JupyterLab, kernel remoto). Colab arranca en `/content` con el filesystem vacío. Nunca usar rutas relativas a secas. Patrón obligatorio en la celda de setup:

```python
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    PROJECT_ROOT = Path("/content")
else:
    PROJECT_ROOT = Path.cwd()
    # Asserción defensiva solo en local: en Colab el FS arranca vacío.
    assert any((PROJECT_ROOT / m).exists() for m in ("model.joblib", "train.py", "data", "scripts")), (
        f"PROJECT_ROOT no parece la raíz del proyecto: {PROJECT_ROOT}. "
        "Abre el notebook desde la carpeta del proyecto o ajusta PROJECT_ROOT manualmente."
    )

DATA_DIR    = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR  = PROJECT_ROOT / "models"
for d in (DATA_DIR, OUTPUTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
```

El `assert` con un check específico del proyecto hace que el error temprano sea legible en local. En Colab el assert no aplica porque el FS arranca vacío y los directorios se crean al vuelo.

## Regla 6: pinear versiones, sin tocar lo preinstalado en Colab

Los notebooks deben ser reproducibles, pero **no a costa de romper el kernel de Colab**. La regla es:

- **Local (Jupyter / VS Code)**: el `requirements.txt` del proyecto pinea con `==` las librerías que tocan el modelo (sklearn, xgboost, pandas) y con `>=` las auxiliares (matplotlib, seaborn). El alumno corre `uv pip install -r requirements.txt` (Regla 8) y obtiene el entorno exacto.
- **Colab**: NO instalar ni actualizar lo que el runtime ya trae preinstalado (numpy, pandas, scikit-learn, matplotlib, seaborn, joblib, scipy). Solo instalar lo que falte (típicamente `huggingface-hub`). Reinstalar numpy o pandas con `--upgrade` rompe los binarios C ya cargados por el kernel y dispara `ImportError: cannot import name '_center' from 'numpy._core.umath'` o errores equivalentes. Ver Troubleshooting al final.

La celda de setup del notebook de entrenamiento imprime las versiones efectivas y el notebook de inferencia las contrasta contra `model_info.json`. Si hay mismatch entre la versión de sklearn de Colab y la versión con la que se entrenó el modelo, el notebook avisa pero no falla — el `.joblib` suele cargar bien entre versiones cercanas (1.5 ↔ 1.6).

## Regla 7: badge "Open in Colab" y publicación en HF Hub

Los notebooks deben abrirse en Colab con un solo clic. Esto requiere dos cosas:

1. **Una URL pública del `.ipynb`**. La opción soportada por este Power es subirlos al repo del modelo en HF Hub (subcarpeta `notebooks/`). El Space `gradio_api` no aplica aquí. Otras opciones (GitHub, Drive) son válidas pero quedan a discreción del proyecto.
2. **Una primera celda markdown con el badge** apuntando a esa URL. Patrón canónico:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A%2F%2Fhuggingface.co%2F{HF_USER}%2F{MODEL_SLUG}%2Fresolve%2Fmain%2Fnotebooks%2F01_entrenamiento.ipynb)
```

Sustituye `{HF_USER}` y `{MODEL_SLUG}` por los valores reales del proyecto. El URL del badge codifica la ruta `resolve/main/notebooks/...` del repo del modelo.

**Workflow de publicación** (lo hace `08_deploy_hf.py` o un script auxiliar):
- Generar `notebooks/01_entrenamiento.ipynb` y `notebooks/02_inferencia.ipynb` con el script de generación.
- Ejecutarlos localmente como sanity check (deben correr de extremo a extremo sin errores).
- **Limpiar artefactos generados** (no subir al repo del modelo `.joblib`, `.png` ni `models/` que se hayan creado durante la prueba; HF Hub debe recibir solo los `.ipynb` y, opcionalmente, sus outputs limpios).
- Subir SOLO los `.ipynb` con `hf upload --include "notebooks/*.ipynb"` (no recursivo a subcarpetas) para evitar arrastrar carpetas locales.

## Regla 8: instalación de dependencias con `uv` en Colab y local

Colab tiene `uv` preinstalado y disponible en el kernel a través de `!uv pip ...`. Es 10-100× más rápido que `pip` resolviendo dependencias y la celda de setup se siente instantánea (ver [docs.astral.sh/uv](https://docs.astral.sh/uv/)).

**Patrón obligatorio en la primera celda de código del notebook:**

```python
# Setup de dependencias (Colab y local)
import importlib, subprocess, sys

def _ensure(pkg: str, import_name: str | None = None) -> None:
    """Instala pkg con uv si import_name no está disponible. Idempotente y silencioso."""
    name = import_name or pkg.split("[")[0].split("==")[0].split(">=")[0]
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call(
            ["uv", "pip", "install", "--system", "--quiet", pkg]
        )

# Solo declaramos lo que NO viene preinstalado en Colab.
_ensure("huggingface-hub>=0.28.1", "huggingface_hub")
# Si el modelo requiere xgboost/lightgbm/imbalanced-learn, agregar aquí:
# _ensure("xgboost>=2.1", "xgboost")
# _ensure("imbalanced-learn>=0.13", "imblearn")
```

Reglas duras:

- **NUNCA** `--upgrade` ni `-U` sobre paquetes que Colab ya trae (numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, joblib). Reemplazarlos en caliente rompe los binarios C cargados por el kernel.
- **SIEMPRE** chequear con `importlib.import_module` antes de instalar, para que la segunda corrida del notebook no reinstale nada.
- En Colab el flag `--system` es obligatorio (instala en el Python del runtime, no en un venv aislado). En local sin venv también funciona; con venv activo `uv pip install` solo (sin `--system`) instala en el venv.
- Para ambientes sin `uv` (algunos Jupyter locales viejos), el fallback es `python -m pip install --quiet`. La función puede detectarlo con `shutil.which("uv")`. Mantener el fallback es opcional para este Power: el cliente objetivo es Colab + entorno con `uv` (que es el que corre el pipeline completo).

## Regla 9: autocontención dura — un notebook, un archivo

Un notebook generado por este Power debe **funcionar al abrirse en un Colab fresco sin clonar el repo del proyecto**. Esto implica:

- **NO importar de `lib/`, `src/`, `utils/` ni cualquier módulo Python local del proyecto.** Toda la lógica que el notebook necesita debe estar inline en sus celdas (funciones de FE, helpers de plot, etc.). Si la lógica es muy larga, refactorizarla a una sola celda con funciones bien nombradas, no a un import externo.
- **NO leer `config.yaml`, `.env`, `params.json` ni archivos de configuración del repo.** Los hiperparámetros, nombres de columnas, slugs de HF Hub y demás se declaran en una **celda inline de configuración**, idealmente la segunda celda de código (después del setup de dependencias). Esto duplica un poco de información respecto al proyecto principal, pero es el costo de la autocontención.
- **NO leer datasets de `data/` local.** Descargar siempre con `hf_hub_download(repo_id="usuario/dataset", repo_type="dataset", filename="...")`. Datasets públicos no requieren credenciales.
- **NO leer modelos de `models/` local en el notebook de inferencia.** Descargar con `hf_hub_download(repo_id="usuario/modelo", filename="model.joblib")` desde el repo del modelo.

Si el proyecto evoluciona y el notebook queda desfasado respecto al pipeline en `scripts/`, regenerarlo desde cero con el script de generación. La autocontención es una promesa al alumno: "este `.ipynb` solo, sin nada más, te entrena el modelo en Colab".

## Regla 10: el notebook de inferencia es la fuente de verdad de la API

Si el proyecto expone el modelo vía Gradio / FastAPI / Lambda, la lógica de `predict()` en el backend debe ser equivalente a la del `02_inferencia.ipynb`. La forma práctica de garantizarlo es extraer esa función a un módulo Python (`predict.py` o similar) y que tanto el notebook como el deploy la importen. Como el notebook es autocontenido (Regla 9), no puede importar ese módulo dentro del `.ipynb`; lo que sí debe hacer es **copiar literalmente** la función `predict()` desde el módulo, con un comentario `# debe coincidir con app_inference/predict.py — regenerar el notebook si cambia`.

## Qué NO hacer

- No mezclar entrenamiento e inferencia en el mismo notebook.
- No depender de `lib/`, `src/`, `config.yaml`, `data/` ni `models/` locales (Regla 9).
- No usar `%pip install --upgrade` sobre paquetes preinstalados de Colab (rompe el kernel; ver Troubleshooting).
- No omitir `model_info.json`: sin él, el `.joblib` es una caja negra sin trazabilidad.
- No usar rutas relativas sin anclar a `PROJECT_ROOT` Colab-aware (Regla 5).
- No subir los `.joblib` ni los `.png` generados por la prueba local al repo del modelo en HF Hub junto con los `.ipynb` (Regla 7: subir solo `notebooks/*.ipynb`).
- No reentrenar dentro del notebook de inferencia ni siquiera "rapidito para probar". Si hace falta probar, va al de entrenamiento.
- No leer datasets de Kaggle, URLs arbitrarias, archivos locales o APIs externas: la única fuente válida es Hugging Face Hub (regla del Power).

## Plantilla mínima de celda de setup (entrenamiento, Colab + local)

Tres celdas iniciales. La primera markdown con el badge, las otras dos de código.

**Celda 1 (markdown)** — badge "Open in Colab":

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A%2F%2Fhuggingface.co%2F{HF_USER}%2F{MODEL_SLUG}%2Fresolve%2Fmain%2Fnotebooks%2F01_entrenamiento.ipynb)

# 01 — Entrenamiento ({PROJECT_NAME})

Notebook autocontenido. Corre de extremo a extremo en Colab CPU sin clonar el repo.
```

**Celda 2 (código)** — instalación de deps con `uv` (Regla 8):

```python
# Setup de dependencias (Colab y local)
import importlib, subprocess

def _ensure(pkg: str, import_name: str | None = None) -> None:
    name = import_name or pkg.split("[")[0].split("==")[0].split(">=")[0]
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call(["uv", "pip", "install", "--system", "--quiet", pkg])

_ensure("huggingface-hub>=0.28.1", "huggingface_hub")
# Agregar aquí cualquier dep que NO venga en Colab (xgboost, imbalanced-learn, etc.)
```

**Celda 3 (código)** — imports, semilla, PROJECT_ROOT, config inline:

```python
import sys, json, joblib, random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

SEED = 42
random.seed(SEED); np.random.seed(SEED)

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

PROJECT_ROOT = Path("/content") if IN_COLAB else Path.cwd()
DATA_DIR    = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR  = PROJECT_ROOT / "models"
for d in (DATA_DIR, OUTPUTS_DIR / "eda", MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Config inline (autocontención: NO leer config.yaml).
HF_USER       = "{HF_USER}"
DATASET_REPO  = f"{HF_USER}/{{DATASET_SLUG}}"
DATASET_FILE  = "data.csv"
MODEL_SLUG    = "{MODEL_SLUG}"
TARGET        = "{TARGET_COL}"
NUM_FEATURES  = [...]   # rellenar
CAT_FEATURES  = [...]   # rellenar

print(f"Colab: {IN_COLAB}")
print(f"python  = {sys.version.split()[0]}")
print(f"pandas  = {pd.__version__}")
print(f"numpy   = {np.__version__}")
print(f"sklearn = {sklearn.__version__}")
```

## Plantilla mínima de celda de setup (inferencia, Colab + local)

**Celda 1 (markdown)** — badge:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A%2F%2Fhuggingface.co%2F{HF_USER}%2F{MODEL_SLUG}%2Fresolve%2Fmain%2Fnotebooks%2F02_inferencia.ipynb)

# 02 — Inferencia ({PROJECT_NAME})

Descarga el modelo entrenado desde HF Hub y muestra cómo predecir.
```

**Celda 2 (código)** — instalación de deps:

```python
import importlib, subprocess

def _ensure(pkg: str, import_name: str | None = None) -> None:
    name = import_name or pkg.split("[")[0].split("==")[0].split(">=")[0]
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call(["uv", "pip", "install", "--system", "--quiet", pkg])

_ensure("huggingface-hub>=0.28.1", "huggingface_hub")
```

**Celda 3 (código)** — descarga, verificación y carga:

```python
import json, joblib, sklearn
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_USER    = "{HF_USER}"
MODEL_SLUG = "{MODEL_SLUG}"
REPO_ID    = f"{HF_USER}/{MODEL_SLUG}"

model_path = Path(hf_hub_download(REPO_ID, "model.joblib"))
info_path  = Path(hf_hub_download(REPO_ID, "model_info.json"))
prep_path  = Path(hf_hub_download(REPO_ID, "preprocessor.joblib"))

info = json.loads(info_path.read_text())
expected = info["library_versions"].get("scikit-learn")
if expected and expected != sklearn.__version__:
    print(f"[warn] sklearn instalado={sklearn.__version__}, modelo entrenado con={expected}. "
          "Si la carga falla, abre 01_entrenamiento.ipynb para reentrenar contra esta versión.")

try:
    model        = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
except Exception as e:
    raise RuntimeError(
        f"No se pudo cargar el modelo: {e}. Probable mismatch de sklearn entre Colab y "
        f"entrenamiento (Colab={sklearn.__version__}, modelo={expected}). "
        "Reentrena con 01_entrenamiento.ipynb en este mismo Colab."
    )

print(f"Modelo cargado: {info['model_name']} ({info['model_type']})")
print(f"Features esperadas: {info['feature_order']}")
```

## Troubleshooting

### `ImportError: cannot import name '_center' from 'numpy._core.umath'`
**Causa:** El notebook hizo `%pip install --upgrade pandas numpy ...`. pip instala una versión nueva de numpy en disco, pero el kernel ya tiene la antigua cargada en memoria; los binarios C de pandas (compilados contra la antigua) chocan con la nueva. Cualquier `import pandas` o `import sklearn` posterior explota.
**Solución:**
1. NO actualizar paquetes preinstalados en Colab (Regla 6 + Regla 8). Borrar el `--upgrade` y solo instalar lo que falte (`huggingface-hub` típicamente).
2. Si ya pasó: en Colab, `Runtime → Disconnect and delete runtime`, luego reabrir el notebook. La instalación contaminada queda en `/usr/local/lib/python3.X/dist-packages/` y persiste hasta que el runtime se recicla.
3. Regenerar el notebook con la plantilla de la Regla 8 (función `_ensure` que solo instala si falta).

### `OSError: Cannot find empty port in range: 7861-7861` al lanzar Gradio desde el notebook
**Causa:** El notebook tiene una celda que lanza una app Gradio embebida y hardcodea un puerto distinto de 7860.
**Solución:** Usar el patrón condicional documentado en `gradio-interfaces.md` (puerto 7860 si `SPACE_ID` está seteado, 7861 en local). En Colab, Gradio habilita `share=True` automáticamente; no hace falta puerto fijo.

### El badge "Open in Colab" abre el notebook pero falla en la primera celda
**Causa típica:** la URL del badge apunta a una versión vieja del `.ipynb` en HF Hub que aún tiene el `%pip install --upgrade`. HF Hub sirve `resolve/main/...` con caché agresiva.
**Solución:** Sobreescribir el `.ipynb` corregido (Regla 7), esperar 30-60s a que el CDN invalide, y refrescar el badge en Colab con `Runtime → Disconnect and delete runtime` antes de reabrirlo.

### `ModuleNotFoundError: No module named 'lib'` (o similar)
**Causa:** el notebook depende de un módulo local del proyecto (`lib/preprocessing.py`, `src/utils.py`, etc.). Viola la Regla 9 (autocontención).
**Solución:** Inline la lógica del módulo dentro del notebook (en una celda con funciones bien nombradas). Si la lógica es demasiado grande para inline, regenera el notebook desde el script `scripts/build_notebooks.py` con la lógica inyectada como string.

### Los `.joblib` se subieron al repo del modelo junto con los `.ipynb`
**Causa:** `hf upload notebooks/` recursivo arrastró los artefactos generados al ejecutar el notebook localmente como sanity check.
**Solución:** Subir explícitamente solo los `.ipynb` con `hf upload {repo_id} ./notebooks/01_entrenamiento.ipynb notebooks/01_entrenamiento.ipynb` (un archivo a la vez) o con `--include "*.ipynb"`. Limpiar los artefactos sobrantes del repo con `hf api delete-file` o desde la UI.
