# Construcción de Interfaces con Gradio

## ⚠️ Errores Documentados en HF Spaces (LEER ANTES DE GENERAR `app_inference`)

Los dos errores siguientes han ocurrido repetidamente al desplegar la app de inferencia. **Antes de escribir o modificar `app_inference/app_inference.py` y `app_inference/requirements.txt`, verificar explícitamente que el código a generar no incurre en ninguno de ellos.**

### Error 1 · `OSError: Cannot find empty port in range: 78XX-78XX`

**Causa**: HF Spaces solo expone el puerto `7860` dentro del contenedor. Cualquier `demo.launch(server_port=7861)` (o cualquier puerto distinto de 7860) falla en runtime con ese `OSError`.

**Por qué se cae el agente en esto**: durante el desarrollo local quiere correr `app_results` (puerto 7860 default) y `app_inference` (puerto distinto) al mismo tiempo, y termina hardcodeando 7861 en `app_inference`. Ese código sube tal cual al Space y revienta.

**Patrón correcto (OBLIGATORIO)**: detectar el entorno con la variable `SPACE_ID` que HF inyecta automáticamente en Spaces:

```python
import os, gradio as gr
# ... resto de la app ...
port = 7860 if os.environ.get("SPACE_ID") else 7861
demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
```

- En HF Spaces (`SPACE_ID` está definida) → usa 7860 (el único puerto válido).
- Localmente (`SPACE_ID` no existe) → usa 7861, evitando conflicto con `app_results` que corre en 7860.
- `server_name="0.0.0.0"` es obligatorio en Spaces para que el contenedor exponga la app.

**Prohibido**: hardcodear `server_port=7861` (o cualquier valor distinto de 7860) sin la rama condicional. Prohibido también omitir `server_name="0.0.0.0"`.

### Error 2 · `ResolutionImpossible` en `pip install` del Space (huggingface-hub vs gradio)

**Causa**: `gradio==5.31.0` requiere `huggingface-hub>=0.28.1`. Si el `requirements.txt` pinea `huggingface-hub==0.25.0` (versión común en proyectos que arrancaron con Gradio 5.x más antiguo), pip aborta con `ERROR: Cannot install gradio and huggingface-hub==0.25.0 because these package versions have conflicting dependencies`.

**Patrón correcto (OBLIGATORIO)**: en `app_inference/requirements.txt`, usar `huggingface-hub` con rango compatible o sin pin estricto, **no** con `==`:

```
gradio>=6.0,<7.0
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4
joblib==1.4.2
huggingface-hub>=0.28.1   # NO usar ==0.25.0; gradio 5.31+ exige >=0.28.1
```

**Verificación obligatoria antes de subir**: ejecutar localmente `uv pip compile app_inference/requirements.txt` (o `pip install --dry-run -r app_inference/requirements.txt` en el venv del proyecto) y confirmar que pip resuelve sin error. Si falla localmente, falla en Spaces.

**Regla de pinning para apps que se despliegan**:
- `scikit-learn`, `numpy`, `pandas`, `joblib` → pin estricto (`==`) para reproducibilidad de la deserialización del modelo.
- `gradio`, `huggingface-hub`, `fastapi`, `pydantic` → rango compatible (`>=X,<Y`) para que pip resuelva versiones consistentes entre sí. Estas tres librerías están fuertemente acopladas y un pin estricto cruzado siempre rompe.

### Checklist antes de `08_deploy_hf.py`

Antes de invocar el script de despliegue, el agente DEBE verificar explícitamente:

- [ ] `app_inference/app_inference.py` usa el patrón `port = 7860 if os.environ.get("SPACE_ID") else 7861` y `server_name="0.0.0.0"`.
- [ ] `app_inference/app_inference.py` NO contiene `server_port=7861` hardcoded fuera de la rama condicional.
- [ ] `app_inference/requirements.txt` usa `huggingface-hub>=0.28.1` (o rango compatible), NO `==0.25.0`.
- [ ] La versión de `gradio` en `requirements.txt` coincide con el `sdk_version` del `README.md` del Space (ambos del mismo rango / misma major).
- [ ] `pip install --dry-run -r app_inference/requirements.txt` resuelve sin error en local.
- [ ] Cualquier archivo de salida que la app genere (CSV, PNG, JSON descargables) se escribe a `tempfile.gettempdir()`, NO a `outputs/` ni a rutas relativas del proyecto (Error 3).
- [ ] Las llamadas a `model.predict(...)` reciben un `pd.DataFrame` con las columnas en el orden de `model_info.json["feature_order"]`, NO un `ndarray` ni un dict (Error 4).

Si cualquiera de estos puntos falla, **corregir antes de subir**. Resubir un Space después de un build error consume tiempo de cola en HF y confunde el feedback al usuario.

### Error 3 · `gradio.exceptions.InvalidPathError: Cannot move /outputs/predictions.csv to the gradio cache dir`

**Causa**: la app genera un archivo de salida (CSV de predicciones por lotes, PNG de un plot, JSON de un reporte) en una ruta relativa al proyecto local (`outputs/predictions.csv`, `PROJECT_ROOT / "outputs" / "..."`). En el contenedor de HF Spaces, el cwd es `/app` y `outputs/predictions.csv` se resuelve como `/outputs/predictions.csv`, que está **fuera de las rutas permitidas por Gradio**: solo se aceptan archivos en `/app` (cwd) o en el tempdir del sistema (`/tmp` en Linux, `/var/folders/.../T/` en macOS local). La app revienta cuando intenta devolver el archivo al usuario.

**Por qué se cae el agente en esto**: el código de `app_inference.py` se prueba primero localmente (donde `outputs/` es válido porque pertenece al cwd del proyecto) y se sube al Space sin re-evaluar la ruta. El bug solo se manifiesta en HF Spaces, no en local.

**Patrón correcto (OBLIGATORIO)**: usar `tempfile.gettempdir()` para cualquier archivo de salida que la app produzca y devuelva al usuario. Es la única ruta segura cross-platform (en HF Spaces es `/tmp`, en Linux local también, en macOS es `/var/folders/.../T/`).

```python
import tempfile
from pathlib import Path
import pandas as pd

def predict_batch(file):
    if file is None:
        return "Sube un CSV", None, None
    df = pd.read_csv(file.name)
    predictions = model.predict(df[FEATURE_NAMES])
    result_df = df.copy()
    result_df["prediction"] = predictions

    # ✅ Correcto: tempdir del sistema, válido en local y en HF Spaces.
    out_path = Path(tempfile.gettempdir()) / "predictions.csv"
    result_df.to_csv(out_path, index=False)
    return f"✅ {len(predictions)} predicciones", result_df.head(20), str(out_path)

# ❌ Incorrecto: ruta relativa al proyecto, falla en HF Spaces.
# out_path = PROJECT_ROOT / "outputs" / "predictions.csv"
# out_path = Path("outputs") / "predictions.csv"
```

**Prohibido**: escribir archivos descargables en rutas relativas al proyecto (`outputs/`, `data/`, `cache/`). Prohibido también construir la ruta con `Path(__file__).parent / "outputs" / ...` por la misma razón.

**Validación**: en local, imprime `tempfile.gettempdir()` al iniciar la app y confirma que es una ruta del SO (no del proyecto). Si la prueba local de "Predicción por lotes" funciona y el archivo descargable abre, en Spaces también va a funcionar.

### Error 4 · `UserWarning: X does not have valid feature names, but <Estimator> was fitted with feature names`

**Causa**: la app pasa un `np.ndarray` (o un dict convertido a array) a `model.predict(...)`, pero el modelo se entrenó con un `pd.DataFrame` que tenía nombres de columnas. sklearn ≥1.0 verifica que los inputs en inferencia coincidan con los del entrenamiento (incluidos los nombres) y emite el warning. **No es un error fatal, pero es síntoma de un riesgo real**: si pasas las columnas en orden distinto al del entrenamiento, el modelo predice con features cruzadas y devuelve resultados silenciosamente incorrectos.

**Por qué se cae el agente en esto**: en `predict_one(...)` recibe los valores de los sliders como argumentos sueltos (`income, age, rooms, ...`) y los junta en un `np.array([[income, age, rooms, ...]])`. El array funciona, pero pierde los feature names y depende de que el orden de los argumentos coincida con `feature_order`.

**Patrón correcto (OBLIGATORIO)**: envolver siempre los inputs en un `pd.DataFrame` con las columnas en el orden declarado en `model_info.json["feature_order"]` (Regla 4 de `notebooks-ds.md`). Esto elimina el warning, ancla el orden a una sola fuente de verdad y previene cruces silenciosos.

```python
import pandas as pd

# Cargado al iniciar la app, junto con model y preprocessor:
import json
info = json.load(open(hf_hub_download(MODEL_REPO, "model_info.json")))
FEATURE_NAMES = info["feature_order"]  # única fuente de verdad

def predict_one(**kwargs):
    # ✅ Correcto: DataFrame con columnas en el orden canónico.
    X = pd.DataFrame([{name: kwargs[name] for name in FEATURE_NAMES}])
    y_hat = float(model.predict(X)[0])
    return y_hat

def predict_batch(file):
    df = pd.read_csv(file.name)
    # ✅ Correcto: reordenar y subseleccionar para que coincida con el entrenamiento.
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        return f"❌ Faltan columnas: {missing}", None, None
    X = df[FEATURE_NAMES]
    df["prediction"] = model.predict(X)
    return "OK", df.head(20), None

# ❌ Incorrecto: ndarray, pierde nombres y depende del orden de los argumentos.
# X = np.array([[income, age, rooms, bedrooms, population]])
# y_hat = model.predict(X)[0]
```

**Prohibido**: pasar `np.ndarray`, `list[list]` o dict a `model.predict(...)` cuando el modelo se entrenó con DataFrame. Prohibido también hardcodear el orden de columnas como una constante en `app_inference.py`: leerlo siempre de `model_info.json["feature_order"]`.

**Validación**: ejecutar la app local con todos los flujos (`predict_one`, `predict_batch`) y confirmar que el log no muestra `UserWarning: X does not have valid feature names`. Si aparece el warning, hay un path que sigue pasando ndarray.

---

## Regla 0: cada app vive en su propia carpeta editable (NO inline en scripts)

Las apps Gradio del proyecto deben crearse como **archivos editables a mano**, dentro de carpetas dedicadas. Está prohibido generar el código de la app desde un script Python usando strings multilínea, plantillas dinámicas o `Path.write_text()` con todo el `app.py` embebido.

Razón: cuando el código de la app vive como string dentro de otro `.py`, se pierde el linter, la indentación deja de ser confiable, los diffs son ilegibles, los breakpoints no funcionan, el agente humano (o el siguiente agente IA) no puede modificar la app sin tocar el script generador, y cualquier error al regenerar borra los ajustes manuales del usuario.

Estructura obligatoria:

```
proyecto/
├── app_inference/
│   ├── app_inference.py     ← código principal, editable
│   ├── requirements.txt     ← dependencias del Space (versiones pinneadas)
│   └── README.md            ← metadata YAML del Space
├── app_results/
│   ├── app_results.py
│   └── (sin requirements.txt ni README.md: es solo local)
└── scripts/
    └── 08_deploy_hf.py      ← SOLO sube la carpeta. NO genera archivos.
```

Reglas adicionales:

- El script de deploy (`08_deploy_hf.py`) **solo sube** la carpeta `app_inference/` al Space con `hf upload`. No crea, no sobrescribe, no edita ningún archivo de la app.
- Si necesitas cambiar la UI, edita directamente `app_inference/app_inference.py`. No vuelvas a correr un "generador".
- El agente, al producir una app por primera vez, escribe los tres archivos directamente con `fs_write`, no a través de un script intermedio.
- La separación entre la lógica de la app y la lógica de deploy es estricta: si encuentras código que escribe `app.py` desde dentro de otro `.py`, refactorízalo a archivo plano antes de continuar.

---

El proyecto genera **una aplicación Gradio obligatoria** y una de visualización:

- **app_inference/app_inference.py** (OBLIGATORIA) — UI para el usuario final. Descarga el modelo publicado en HF Hub y expone una interfaz de predicción. Se despliega a Hugging Face Spaces con `hf upload`.
- **app_results/app_results.py** (SE GENERA SIEMPRE) — Dashboard local de visualización de resultados. Muestra las figuras del EDA y las métricas/gráficas de validación. No ejecuta nada, solo lee de `outputs/`. Se indica al usuario al final del entrenamiento que puede usarla.

---

## APP 1: Dashboard de Resultados (app_results/app_results.py)

Esta app es un **visor de resultados** que muestra las figuras y métricas generadas por los scripts del pipeline. No ejecuta ningún script ni modifica datos — solo lee y presenta.

**Se genera SIEMPRE como parte del proyecto.** Al finalizar la etapa de validación (script 05), indicar al usuario:
> "✅ Validación completada. Puedes visualizar todos los resultados de EDA y validación ejecutando: `uv run python app_results/app_results.py`"

### Estructura

La app tiene tres tabs, alineados con lo que el científico de datos necesita revisar después del entrenamiento:

- **Tab EDA**: Figuras de `outputs/figures/eda_*.png` (distribuciones, correlación, outliers, boxplots) + reporte JSON de `outputs/reports/eda_report.json`.
- **Tab Modelo**: **Hiperparámetros del modelo entrenado** (los `best_params_` resultantes del GridSearch/RandomizedSearch), tipo de estimador, features usadas, fecha de entrenamiento. Todo desde `outputs/metrics/train_metadata.json`.
- **Tab Validación**: Figuras de validación (`outputs/figures/val_*.png`: confusion matrix, ROC, residuales, feature importance) + métricas de `outputs/metrics/validation.json`.

### Reglas de visualización de figuras (OBLIGATORIO)

> Lección aprendida en producción: con Gradio 6, `gr.Gallery` reduce las figuras a miniaturas en grilla y al hacer clic abre un modal pequeño que NO respeta el aspect ratio original de matplotlib. Ejes, leyendas y anotaciones quedan ilegibles. **No usar `gr.Gallery` para mostrar figuras del pipeline.**

Patrón obligatorio para el dashboard de resultados:

1. **Una `gr.Image` por figura**, ocupando el ancho completo de la pestaña. Nunca galería.
2. **Altura fija explícita** (`height=600` por defecto). Sin altura, Gradio recorta o reduce según el contenedor padre.
3. **`container=False, show_label=False`** para evitar el marco gris y el label redundante. El título se renderiza arriba con un `gr.Markdown` propio (`#### nombre legible`) generado a partir del nombre del archivo.
4. **Listas explícitas y ordenadas** por sección (`EDA_FIGS`, `VAL_FIGS_OVERVIEW`, `VAL_FIGS_<MODELO>`), no `glob` directo. El orden alfabético casi nunca es el orden narrativo correcto (la distribución del target debe ir antes que las correlaciones, la comparación general antes del detalle por modelo, etc.).
5. **Helper `render_figure_block(filename, prefix)`** que centraliza: verifica existencia del PNG, formatea el título y emite el bloque markdown + image. Si una figura falta, se omite silenciosamente en lugar de romper la app.
6. **Reportes JSON en `gr.Accordion(open=False)`**. Los JSON ocupan mucho espacio vertical y empujan las figuras hacia abajo; deben abrirse bajo demanda.
7. **Tema vía `launch(theme=...)`, NO en el constructor de `Blocks`** — ver sección "Cambios obligatorios para Gradio 6" más abajo.

### Cambios obligatorios para Gradio 6

A partir de Gradio 6, varios parámetros de aplicación se movieron del constructor `gr.Blocks(...)` al método `launch(...)`. Esto incluye `theme`, `css`, `css_paths`, `js`, `head`, `head_paths`. Pasarlos al constructor en Gradio 6 emite warning y, en algunas combinaciones, no aplica el estilo.

```python
# ❌ Gradio 5.x — NO usar en proyectos nuevos
with gr.Blocks(theme=gr.themes.Soft(), title="...") as demo:
    ...
demo.launch()

# ✅ Gradio 6.x — patrón correcto
with gr.Blocks(title="...") as demo:
    ...
demo.launch(theme=gr.themes.Soft())
```

Fuente: guía oficial de migración Gradio 6 (`huggingface.co/spaces/multimodalart/gradio-6-migration-tool`, sección "App-level Changes"). Si el proyecto pinea `gradio>=6.0`, este patrón es obligatorio. Si pineas `gradio<6` (5.x), también es válido (aunque no estrictamente requerido), por lo que **siempre escribir el código en el patrón Gradio 6** para que sea forward-compatible.

### Código de Referencia

```python
#!/usr/bin/env python3
"""app_results.py — Dashboard de visualización de resultados del pipeline ML.

Muestra figuras de EDA y validación generadas por los scripts del pipeline.
No ejecuta nada, solo lee de outputs/.

Uso: uv run python app_results/app_results.py
"""
import json
from pathlib import Path

import gradio as gr

# --- Configuración de figuras (orden narrativo, no alfabético) ---
FIGURES_DIR = Path("outputs/figures")
FIGURE_HEIGHT = 600  # px, altura fija para legibilidad de matplotlib

# Listas explícitas por sección. El agente debe ajustar estos nombres a las
# figuras que realmente generen los scripts 02_eda.py y 05_validate.py.
EDA_FIGS = [
    "eda_target_distribution.png",
    "eda_numeric_distributions.png",
    "eda_outliers_boxplots.png",
    "eda_features_by_target.png",
    "eda_correlation_matrix.png",
]

VAL_FIGS_OVERVIEW = [
    "val_models_comparison.png",  # comparación general entre modelos
]

# Cuando el pipeline entrena varios modelos, una sublista por modelo en el
# orden: confusion matrix → ROC → PR → calibración umbral → feature importance.
VAL_FIGS_BY_MODEL = {
    "Random Forest": [
        "val_random_forest_confusion_matrix.png",
        "val_random_forest_roc_curve.png",
        "val_random_forest_pr_curve.png",
        "val_random_forest_threshold_calibration.png",
        "val_random_forest_feature_importance.png",
    ],
    # Agregar otros modelos aquí (XGBoost, ExtraTrees, etc.)
}


# --- Funciones de carga ---

def load_json(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    with open(path) as f:
        return json.dumps(json.load(f), indent=2, ensure_ascii=False)


def load_eda_report() -> str:
    return load_json(Path("outputs/reports/eda_report.json"), "No se encontró reporte de EDA.")


def load_fe_report() -> str:
    return load_json(Path("outputs/reports/fe_report.json"), "No se encontró reporte de FE.")


def load_train_metadata() -> str:
    return load_json(Path("outputs/metrics/train_metadata.json"), "No se encontró metadata de entrenamiento.")


def load_validation_metrics() -> str:
    return load_json(Path("outputs/metrics/validation.json"), "No se encontraron métricas de validación.")


# --- Helper para renderizar figuras (patrón obligatorio) ---

def render_figure_block(filename: str) -> None:
    """Renderiza una figura como Markdown + Image individual a tamaño completo.

    No usa gr.Gallery porque en Gradio 6 reduce a miniaturas y rompe la
    legibilidad de los plots de matplotlib.
    """
    fig_path = FIGURES_DIR / filename
    if not fig_path.exists():
        return  # se omite silenciosamente si la figura no fue generada

    title = (
        fig_path.stem
        .replace("eda_", "")
        .replace("val_", "")
        .replace("_", " ")
        .title()
    )
    gr.Markdown(f"#### {title}")
    gr.Image(
        value=str(fig_path),
        height=FIGURE_HEIGHT,
        container=False,
        show_label=False,
    )


# --- UI ---

with gr.Blocks(title="ML Results Dashboard") as demo:
    gr.Markdown("# 📊 Dashboard de Resultados ML")
    gr.Markdown("Visualización de resultados del pipeline de entrenamiento.")

    with gr.Tab("📈 EDA"):
        gr.Markdown("### Análisis Exploratorio de Datos")
        for fig in EDA_FIGS:
            render_figure_block(fig)

        with gr.Accordion("Reporte EDA (JSON)", open=False):
            eda_report_btn = gr.Button("Cargar Reporte EDA", variant="primary")
            eda_report_text = gr.Code(language="json", label="Reporte EDA")
            eda_report_btn.click(load_eda_report, outputs=eda_report_text)

        with gr.Accordion("Reporte Feature Engineering (JSON)", open=False):
            fe_report_btn = gr.Button("Cargar Reporte FE", variant="primary")
            fe_report_text = gr.Code(language="json", label="Reporte FE")
            fe_report_btn.click(load_fe_report, outputs=fe_report_text)

    with gr.Tab("🔧 Modelo"):
        gr.Markdown("### Modelo Entrenado")
        gr.Markdown(
            "Hiperparámetros (`best_params_`), tipo de estimador, features usadas "
            "y metadata del entrenamiento. Refleja el estado del modelo serializado en "
            "`models/model.joblib`."
        )
        meta_btn = gr.Button("Cargar Metadata del Modelo", variant="primary")
        meta_text = gr.Code(language="json", label="Train Metadata")
        meta_btn.click(load_train_metadata, outputs=meta_text)

    with gr.Tab("✅ Validación"):
        gr.Markdown("### Comparación general")
        for fig in VAL_FIGS_OVERVIEW:
            render_figure_block(fig)

        for model_name, figs in VAL_FIGS_BY_MODEL.items():
            gr.Markdown(f"### {model_name}")
            for fig in figs:
                render_figure_block(fig)

        with gr.Accordion("Métricas de validación (JSON)", open=False):
            val_btn = gr.Button("Cargar Métricas", variant="primary")
            val_text = gr.Code(language="json", label="Métricas")
            val_btn.click(load_validation_metrics, outputs=val_text)

# Patrón Gradio 6: theme se pasa a launch(), NO al constructor de Blocks
demo.launch(theme=gr.themes.Soft())
```

### Convención de nombres de figuras

Para que el dashboard funcione correctamente, los scripts de EDA y validación DEBEN guardar las figuras con estos prefijos:

| Script | Prefijo | Ejemplos |
|--------|---------|----------|
| `02_eda.py` | `eda_` | `eda_distributions.png`, `eda_correlation.png`, `eda_boxplots.png`, `eda_missing.png` |
| `05_validate.py` | `val_` | `val_confusion_matrix.png`, `val_roc_curve.png`, `val_residuals.png`, `val_feature_importance.png` |

### Contrato de `train_metadata.json` (alimenta el tab Modelo)

El script `04_train.py` DEBE escribir `outputs/metrics/train_metadata.json` con, al menos, estas claves para que el tab "Modelo" del dashboard tenga sentido:

```json
{
  "estimator": "XGBClassifier",
  "trained_at": "2026-05-17T18:32:11",
  "random_state": 42,
  "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "target": "species",
  "search_strategy": "RandomizedSearchCV",
  "cv_folds": 5,
  "scoring": "f1_macro",
  "best_params": {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8
  },
  "best_cv_score": 0.962,
  "training_seconds": 12.4,
  "model_path": "models/model.joblib"
}
```

Con esta estructura, el tab "Modelo" muestra al usuario exactamente **qué hiperparámetros quedaron en el modelo serializado**, no solo metadata genérica.

### Cuándo indicar al usuario

Al finalizar la validación (etapa 5), el agente DEBE informar:
> "✅ Validación completada. Para visualizar interactivamente los resultados de EDA y validación, ejecuta:
> ```
> uv run python app_results/app_results.py
> ```
> Se abrirá un dashboard local con las figuras y métricas generadas."

---

## APP 2: Inference (app_inference/app_inference.py)

Esta app se despliega a Hugging Face Spaces. **Descarga el modelo publicado en HF Hub** (la fuente de verdad del modelo) y expone una interfaz de predicción para el usuario final.

> **⚠️ REGLA DE PUERTO PARA HF SPACES:** La app de inferencia DEBE usar `demo.launch(server_name="0.0.0.0", server_port=7860)` cuando se despliega a HF Spaces. El contenedor de Spaces SOLO expone el puerto 7860. Usar cualquier otro puerto (7861, 8080, etc.) causa `OSError: Cannot find empty port in range`. Localmente se puede probar en 7861 para no chocar con la app de resultados, pero el código que se sube al Space DEBE tener 7860. Patrón recomendado:
> ```python
> # Detectar si estamos en HF Spaces o local
> import os
> port = 7860 if os.environ.get("SPACE_ID") else 7861
> demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
> ```

```python
#!/usr/bin/env python3
"""app_inference.py — UI de inferencia: consume modelo de HF Hub.

Descarga modelo publicado en Hugging Face Hub y expone interfaz de predicción.
Se despliega a Hugging Face Spaces.

Uso local:  uv run python app_inference/app_inference.py
Despliegue: uv run python scripts/08_deploy_hf.py
"""
import gradio as gr
import joblib
import json
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download

# --- Configuración ---
# Referencia al modelo en HF Hub
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "usuario/mi-modelo")


def load_model():
    """Descarga y carga el modelo + metadata desde HF Hub.

    Devuelve (model, encoder, info) donde info["feature_order"] es la
    única fuente de verdad del orden de columnas (Regla 4 notebooks-ds.md).
    """
    print(f"Descargando modelo desde HF Hub: {MODEL_REPO}")

    model_path = hf_hub_download(MODEL_REPO, "model.joblib")
    model = joblib.load(model_path)

    info_path = hf_hub_download(MODEL_REPO, "model_info.json")
    info = json.loads(Path(info_path).read_text())

    # Intentar cargar label encoder si existe
    encoder = None
    try:
        encoder_path = hf_hub_download(MODEL_REPO, "label_encoder.joblib")
        encoder = joblib.load(encoder_path)
    except Exception:
        pass

    return model, encoder, info


# --- Cargar modelo al iniciar ---
model, encoder, info = load_model()
FEATURE_NAMES = info["feature_order"]  # orden canónico, evita Error 4


def predict(**kwargs):
    """Predicción individual. Envolvemos siempre en DataFrame con columnas
    en el orden de FEATURE_NAMES (evita UserWarning + cruces silenciosos)."""
    input_df = pd.DataFrame([{name: kwargs[name] for name in FEATURE_NAMES}])
    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_df)[0]
        classes = model.classes_
        if encoder:
            classes = encoder.inverse_transform(range(len(classes)))
        return {str(c): float(p) for c, p in zip(classes, probas)}

    if encoder:
        prediction = encoder.inverse_transform([prediction])[0]

    return str(prediction)


def predict_batch(file):
    """Predicción por lotes desde CSV.

    Salida descargable en tempfile.gettempdir() (evita Error 3 en HF Spaces).
    Reordena las columnas a FEATURE_NAMES (evita Error 4).
    """
    if file is None:
        return "Sube un archivo CSV", None, None

    df = pd.read_csv(file.name)

    # Validar columnas y reordenar al canónico antes de predecir
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        return f"❌ Faltan columnas: {missing}. Esperadas: {FEATURE_NAMES}", None, None
    X = df[FEATURE_NAMES]

    predictions = model.predict(X)

    result_df = df.copy()
    if encoder:
        result_df["prediction"] = encoder.inverse_transform(predictions)
    else:
        result_df["prediction"] = predictions

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        classes = model.classes_
        if encoder:
            classes = encoder.inverse_transform(range(len(classes)))
        for i, cls in enumerate(classes):
            result_df[f"proba_{cls}"] = probas[:, i]

    # Salida en tempdir del SO (válido en local y HF Spaces)
    out_path = Path(tempfile.gettempdir()) / "predictions.csv"
    result_df.to_csv(out_path, index=False)

    return f"✅ {len(predictions)} predicciones generadas", result_df.head(20), str(out_path)


# --- UI ---
# Patrón Gradio 6: NO pasar theme/css al constructor de Blocks; van en launch().
with gr.Blocks(title="ML Predictor") as demo:
    gr.Markdown("# 🔮 ML Predictor")
    gr.Markdown(f"Modelo: `{MODEL_REPO}`")

    with gr.Tab("Predicción Individual"):
        gr.Markdown("Ingresa los valores para obtener una predicción.")
        # AJUSTAR: Definir inputs según las features del modelo
        # Ejemplo para Iris:
        sepal_length = gr.Number(label="Sepal Length", value=5.1)
        sepal_width = gr.Number(label="Sepal Width", value=3.5)
        petal_length = gr.Number(label="Petal Length", value=1.4)
        petal_width = gr.Number(label="Petal Width", value=0.2)

        predict_btn = gr.Button("Predecir", variant="primary")
        output = gr.Label(num_top_classes=5, label="Predicción")

        predict_btn.click(
            fn=lambda sl, sw, pl, pw: predict(
                sepal_length=sl, sepal_width=sw,
                petal_length=pl, petal_width=pw
            ),
            inputs=[sepal_length, sepal_width, petal_length, petal_width],
            outputs=output
        )

    with gr.Tab("Predicción por Lotes"):
        gr.Markdown("Sube un CSV con las mismas columnas que el dataset de entrenamiento.")
        file_input = gr.File(label="Archivo CSV", file_types=[".csv"])
        batch_btn = gr.Button("Predecir Lote", variant="primary")
        batch_status = gr.Textbox(label="Estado")
        batch_results = gr.DataFrame(label="Resultados (preview)")
        batch_download = gr.File(label="Descargar predicciones (CSV completo)")

        batch_btn.click(
            fn=predict_batch,
            inputs=file_input,
            outputs=[batch_status, batch_results, batch_download]
        )

    with gr.Tab("Información del Modelo"):
        gr.Markdown(f"""
        **Modelo**: `{type(model).__name__}`
        **Fuente**: HF Hub `{MODEL_REPO}`

        Este modelo fue entrenado, validado y publicado usando el pipeline
        de Data Science Assistant.
        """)

# Puerto 7860 obligatorio para HF Spaces; 7861 para pruebas locales
port = 7860 if os.environ.get("SPACE_ID") else 7861
demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
```

### app_inference/requirements.txt

> ⚠️ Pinear `huggingface-hub` con rango (`>=0.28.1`), NO con `==0.25.0`: rompe el resolver de pip junto a gradio 5.31+. Ver "Errores Documentados en HF Spaces" al inicio de este steering.

```
gradio>=6.0,<7.0
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4
joblib==1.4.2
huggingface-hub>=0.28.1
```

### app_inference/README.md (para HF Spaces)

> ⚠️ El `sdk_version` debe coincidir EXACTAMENTE con la versión real instalada por `pip` desde `requirements.txt`. Si `requirements.txt` usa `gradio>=6.0,<7.0` y pip instala `6.0.1`, pon `sdk_version: "6.0.1"`. Verificar con: `uv run python -c "import gradio; print(gradio.__version__)"` antes de publicar.

```markdown
---
title: Mi Predictor ML
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.0.1"
app_file: app_inference.py
pinned: false
license: apache-2.0
---

# Mi Predictor ML

Modelo de machine learning entrenado, validado y publicado en Hugging Face Hub.
Interfaz de predicción individual y por lotes.
```

---

## Conceptos de Gradio para Ambas Apps

### gr.Interface vs gr.Blocks

- **`gr.Interface`** — API de alto nivel, ideal para funciones simples (input → output). Útil para la app de inferencia si es simple.
- **`gr.Blocks`** — API de bajo nivel, control total sobre layout, eventos e interactividad. Necesario para la app de training con múltiples tabs.

### Componentes Clave

```python
# Inputs
gr.Number(label="Edad", value=30)
gr.Textbox(label="Nombre", placeholder="...")
gr.Dropdown(choices=["A", "B", "C"], label="Categoría")
gr.Slider(0, 100, value=50, step=1, label="Score")
gr.Checkbox(label="Activo", value=True)
gr.File(label="CSV", file_types=[".csv"])

# Outputs
gr.Label(num_top_classes=5)
gr.DataFrame(label="Resultados")
gr.Plot(label="Gráfica")
gr.Textbox(label="Info", lines=5)
gr.JSON(label="Métricas")
gr.Image(label="Figura")
```

### Tabs, Rows, Columns

```python
with gr.Blocks() as demo:
    with gr.Tab("Tab 1"):
        with gr.Row():
            with gr.Column(scale=2):
                input1 = gr.Textbox()
            with gr.Column(scale=1):
                output1 = gr.Label()

    with gr.Tab("Tab 2"):
        with gr.Accordion("Opciones Avanzadas", open=False):
            param1 = gr.Slider(...)
```

### Estado entre Interacciones

```python
with gr.Blocks() as demo:
    state = gr.State(value={})  # Persiste entre llamadas por usuario

    def update(data, current_state):
        current_state["last_input"] = data
        return current_state

    input1 = gr.Textbox()
    input1.change(update, [input1, state], state)
```

### Progreso

```python
def long_task(data, progress=gr.Progress()):
    for i in progress.tqdm(range(100), desc="Procesando"):
        # ... trabajo pesado
        pass
    return "Completado"
```

### Theming

En Gradio 6 el tema se pasa al `launch()`, no al constructor de `Blocks`:

```python
# Patrón correcto Gradio 6.x (forward-compatible con 5.x)
with gr.Blocks() as demo:
    ...
demo.launch(theme=gr.themes.Soft())     # Recomendado
# demo.launch(theme=gr.themes.Glass())
# demo.launch(theme=gr.themes.Monochrome())
```

Pasarlo al constructor (`gr.Blocks(theme=...)`) emite warning en 5.x y deja de funcionar en 6.x. Lo mismo aplica para `css`, `css_paths`, `js`, `head`, `head_paths`.

## Consultar Documentación de Gradio (MCP Server independiente)

El servidor MCP `gradio` (registrado en `~/.kiro/settings/mcp.json` con URL `https://gradio-docs-mcp.hf.space/gradio_api/mcp/`) expone exactamente dos tools:

| Tool real | Cuándo usarla |
|---|---|
| `docs_mcp_search_gradio_docs(query)` | Búsqueda por embeddings sobre la documentación oficial. **Default**: úsala primero. |
| `docs_mcp_load_gradio_docs()` | Devuelve toda la doc + ejemplos. Solo si necesitas contexto amplio (un componente entero, un guide). |

> ⚠️ El query NO debe contener la palabra "Gradio" — el server ya está embebido en su corpus. Usa términos del componente o feature que necesitas.

### Cuándo activar el server (triggers obligatorios)

Antes de escribir o modificar código que use Gradio, consulta el server si:

1. Vas a usar un componente que no usaste en este proyecto (`gr.File`, `gr.Dataframe`, `gr.Plot`, `gr.Chatbot`, `gr.Image`, etc.) → confirma su API actual.
2. Vas a usar `gr.Blocks` con layout no trivial (Tabs, Accordion, Row/Column anidados) → verifica props y eventos disponibles.
3. Vas a usar `gr.State` o eventos encadenados (`.then()`, `.success()`).
4. Vas a desplegar a HF Spaces y necesitas configurar `app.py` + `requirements.txt` + `README.md` con el `sdk_version` correcto.
5. Vas a integrar Gradio con HF Hub (`hf_hub_download`, autenticación, secrets en Spaces).
6. El usuario reporta un error de Gradio que no reconoces inmediatamente.

### Protocolo de consulta (3 pasos)

1. **Identifica** qué componente o feature necesitas.
2. **Llama** la tool más específica que aplique:
   - Para una pregunta concreta: `docs_mcp_search_gradio_docs(query="...")`
   - Para contexto general antes de escribir un app completa: `docs_mcp_load_gradio_docs()` (una sola vez al inicio)
3. **Cita** brevemente la fuente en un comentario del código si la decisión depende de la doc:
   ```python
   # gr.Dataframe acepta pandas.DataFrame directamente; tipo `headers` se infiere
   # (consultado vía gradio docs MCP)
   df_out = gr.Dataframe(value=df, label="Predicciones")
   ```

### Tabla de queries por decisión

| Lo que vas a escribir | Query sugerido |
|---|---|
| Layout con Tabs y Row | `"blocks tabs row column layout"` |
| Subir un CSV y procesarlo | `"file upload csv pandas dataframe"` |
| Mostrar un plot de matplotlib | `"plot matplotlib component output"` |
| Mostrar varias figuras PNG ya generadas a tamaño completo | `"image component height container show_label"` (usar `gr.Image` por figura, NO `gr.Gallery` — ver "Reglas de visualización de figuras") |
| Eventos encadenados | `"event listeners chain then success"` |
| State entre componentes | `"state component share data"` |
| Streaming de respuestas | `"chatbot streaming generator"` |
| Configurar Space en HF | `"deploy huggingface spaces sdk_version"` |
| Autenticación / OAuth en Space | `"oauth login huggingface spaces"` |
| Tema personalizado | `"themes launch parameter blocks migration"` (Gradio 6 movió `theme` a `launch()`) |
| Validación de inputs | `"input validation error message"` |
| File upload con tipos restringidos | `"file types accept extensions"` |
| Migrar app de Gradio 5 a 6 | `"gradio 6 migration guide breaking changes"` (incluye palabra "gradio" porque va a búsqueda web genérica) |

### Modo degradado (server no disponible)

Si las tools `docs_mcp_*` no están registradas o devuelven error:

1. Declara al usuario que el server `gradio` no está activo.
2. Cae a la búsqueda web del agente (`remote_web_search` / `web_fetch`) con query como `"gradio Blocks state management python"` (incluye la palabra "gradio" porque la web genérica lo necesita) y prefiere los dominios `gradio.app` y `huggingface.co`.
3. **No inventes** firmas de funciones ni nombres de props. Si no consultaste la doc y dudas, pregunta al usuario o marca el snippet como "verificar".

### Cuándo NO consultar el server

- Componentes triviales que ya usaste en el proyecto (`gr.Textbox`, `gr.Number`, `gr.Button`).
- Patrones repetidos del propio steering (galería de figuras, layouts, tema vía `launch()`, etc.).
- Decisiones puramente operativas (qué puerto usar, dónde guardar logs).

### Pinear versión de Gradio en `requirements.txt`

Las apps generadas (`app_inference/requirements.txt`, `app_results/requirements.txt`, y `pyproject.toml` del proyecto) deben pinear `gradio` con un rango compatible con el código del steering (Gradio 6 ready):

```
gradio>=6.0,<7.0
```

Y el `sdk_version` del `README.md` del Space debe coincidir exactamente con la versión instalada en el entorno (verificar con `uv run python -c "import gradio; print(gradio.__version__)"` antes de publicar). Un mismatch entre `sdk_version` y la versión real es causa común de fallos al construir el Space.
