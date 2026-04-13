# Construcción de Interfaces con Gradio

El proyecto genera **dos aplicaciones Gradio** con propósitos distintos:

- **app_training/app_training.py** — UI local para el científico de datos. Orquesta todo el pipeline: ingesta, EDA, Feature Engineering, entrenamiento, validación y publicación. Se ejecuta con `uv run python app_training/app_training.py`.
- **app_inference/app_inference.py** — UI para el usuario final. Descarga el modelo publicado en Kaggle y expone una interfaz de predicción. Se despliega a Hugging Face Spaces.

---

## APP 1: Training Pipeline (app_training/app_training.py)

Esta app orquesta todas las etapas del pipeline con tabs separados. El científico de datos ejecuta cada etapa desde la UI, revisa resultados y decide cuándo avanzar.

```python
#!/usr/bin/env python3
"""app_training.py — UI de entrenamiento: pipeline completo.

Orquesta: ingesta → EDA → Feature Engineering → entrenamiento → validación → publicación.
Cada tab es una etapa. El usuario revisa resultados antes de avanzar.

Uso: uv run python app_training/app_training.py
"""
import sys
sys.path.insert(0, ".")

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import joblib
import subprocess
from pathlib import Path

# --- Funciones de cada etapa ---

def run_ingest(kaggle_ref):
    """Etapa 1: Descarga datos desde Kaggle."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["uvx", "kaggle", "datasets", "download", "-d", kaggle_ref,
             "--unzip", "-p", "data/raw/"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return f"❌ Error: {result.stderr}", None

        files = list(Path("data/raw").glob("*"))
        file_list = "\n".join(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)" for f in files)
        return f"✅ Descargados {len(files)} archivos:\n{file_list}", files[0].name if files else None
    except Exception as e:
        return f"❌ Error: {e}", None


def run_eda(filename):
    """Etapa 2: EDA — genera estadísticas y visualizaciones."""
    if not filename:
        return "Selecciona un archivo primero", None, None, None

    df = pd.read_csv(f"data/raw/{filename}")
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    # Info básica
    info = f"Shape: {df.shape[0]} filas × {df.shape[1]} columnas\n\n"
    info += f"Columnas: {', '.join(df.columns)}\n\n"
    info += f"Tipos:\n{df.dtypes.to_string()}\n\n"
    info += f"Valores faltantes:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}"

    # Distribuciones
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n = min(len(numeric_cols), 6)
    if n > 0:
        fig_dist, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for i, col in enumerate(numeric_cols[:n]):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(col, fontsize=10)
        plt.tight_layout()
    else:
        fig_dist = None

    # Correlación
    if len(numeric_cols) > 1:
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(df[numeric_cols].corr(), dtype=bool))
        sns.heatmap(df[numeric_cols].corr(), mask=mask, annot=True,
                    fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlación")
        plt.tight_layout()
    else:
        fig_corr = None

    return info, df.head(20), fig_dist, fig_corr


def run_feature_engineering(filename, target_col, numeric_cols_str, cat_cols_str, test_size):
    """Etapa 3: Feature Engineering."""
    if not filename or not target_col:
        return "Completa los campos requeridos", None

    df = pd.read_csv(f"data/raw/{filename}")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split

    numeric_features = [c.strip() for c in numeric_cols_str.split(",") if c.strip()]
    categorical_features = [c.strip() for c in cat_cols_str.split(",") if c.strip()]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if y.dtype == "object" or y.nunique() < 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Nombres de features
    feature_names = list(numeric_features)
    if categorical_features:
        cat_names = preprocessor.named_transformers_["cat"] \
            .named_steps["onehot"].get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_names)

    train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    train_df[target_col] = y_train.values
    train_df.to_csv("data/processed/train.csv", index=False)

    test_df = pd.DataFrame(X_test_proc, columns=feature_names)
    test_df[target_col] = y_test.values
    test_df.to_csv("data/processed/test.csv", index=False)

    joblib.dump(preprocessor, "models/preprocessor.joblib")

    summary = (f"✅ Feature Engineering completado\n"
               f"Train: {train_df.shape}, Test: {test_df.shape}\n"
               f"Features: {len(feature_names)}\n"
               f"Preprocessor: models/preprocessor.joblib")
    return summary, train_df.head(10)


def run_train(target_col, model_type, n_estimators, max_depth):
    """Etapa 4: Entrenamiento."""
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv("data/processed/train.csv")
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    from lib.model_utils import build_model
    params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
    model = build_model(model_type, params)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.joblib")

    meta = {"model_type": model_type, "params": params,
            "n_features": X_train.shape[1], "n_samples": X_train.shape[0],
            "feature_names": list(X_train.columns)}
    with open("outputs/metrics/train_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return f"✅ Modelo entrenado: {model_type}\nFeatures: {X_train.shape[1]}, Samples: {X_train.shape[0]}\nGuardado: models/model.joblib"


def run_validate(target_col):
    """Etapa 5: Validación."""
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, f1_score, r2_score,
                                  mean_squared_error)

    model = joblib.load("models/model.joblib")
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    y_pred = model.predict(X_test)

    is_clf = y_test.dtype == "object" or y_test.nunique() < 20

    if is_clf:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        metrics_text = (f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n"
                        f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}\n\n"
                        f"{classification_report(y_test, y_pred)}")
    else:
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color="r", linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        plt.tight_layout()

        metrics_text = (f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}\n"
                        f"R²: {r2_score(y_test, y_pred):.4f}")

    return metrics_text, fig


def run_publish_dataset():
    """Etapa 6: Publicar dataset curado."""
    result = subprocess.run(
        ["uv", "run", "python", "scripts/06_publish_dataset.py"],
        capture_output=True, text=True, timeout=60
    )
    card_path = Path("cards/DATA_CARD.md")
    card_content = card_path.read_text() if card_path.exists() else "No generada aún"
    return result.stdout + result.stderr, card_content


def run_publish_model():
    """Etapa 7: Publicar modelo."""
    result = subprocess.run(
        ["uv", "run", "python", "scripts/07_publish_model.py"],
        capture_output=True, text=True, timeout=60
    )
    card_path = Path("cards/MODEL_CARD.md")
    card_content = card_path.read_text() if card_path.exists() else "No generada aún"
    return result.stdout + result.stderr, card_content


# --- UI ---

with gr.Blocks(theme=gr.themes.Soft(), title="ML Training Pipeline") as demo:
    gr.Markdown("# 🔬 ML Training Pipeline")
    gr.Markdown("Ejecuta cada etapa secuencialmente. Revisa resultados antes de avanzar.")

    with gr.Tab("1️⃣ Ingesta"):
        kaggle_ref = gr.Textbox(label="Kaggle Dataset (user/dataset)", placeholder="username/dataset-name")
        ingest_btn = gr.Button("Descargar", variant="primary")
        ingest_output = gr.Textbox(label="Resultado", lines=5)
        file_selector = gr.Textbox(label="Archivo principal", interactive=True)
        ingest_btn.click(run_ingest, inputs=kaggle_ref, outputs=[ingest_output, file_selector])

    with gr.Tab("2️⃣ EDA"):
        eda_file = gr.Textbox(label="Archivo CSV (en data/raw/)")
        eda_btn = gr.Button("Ejecutar EDA", variant="primary")
        eda_info = gr.Textbox(label="Información", lines=10)
        eda_preview = gr.DataFrame(label="Preview")
        with gr.Row():
            eda_dist = gr.Plot(label="Distribuciones")
            eda_corr = gr.Plot(label="Correlación")
        eda_btn.click(run_eda, inputs=eda_file, outputs=[eda_info, eda_preview, eda_dist, eda_corr])

    with gr.Tab("3️⃣ Feature Engineering"):
        with gr.Row():
            fe_file = gr.Textbox(label="Archivo CSV")
            fe_target = gr.Textbox(label="Columna Target")
        fe_numeric = gr.Textbox(label="Columnas numéricas (separadas por coma)")
        fe_cat = gr.Textbox(label="Columnas categóricas (separadas por coma)")
        fe_test_size = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="Test Size")
        fe_btn = gr.Button("Ejecutar Feature Engineering", variant="primary")
        fe_output = gr.Textbox(label="Resultado", lines=5)
        fe_preview = gr.DataFrame(label="Train Preview")
        fe_btn.click(run_feature_engineering,
                     inputs=[fe_file, fe_target, fe_numeric, fe_cat, fe_test_size],
                     outputs=[fe_output, fe_preview])

    with gr.Tab("4️⃣ Entrenamiento"):
        with gr.Row():
            train_target = gr.Textbox(label="Columna Target")
            train_model = gr.Dropdown(
                choices=["random_forest", "gradient_boosting", "xgboost", "logistic_regression"],
                value="random_forest", label="Tipo de Modelo"
            )
        with gr.Row():
            train_estimators = gr.Slider(50, 500, value=200, step=50, label="n_estimators")
            train_depth = gr.Slider(3, 20, value=10, step=1, label="max_depth")
        train_btn = gr.Button("Entrenar Modelo", variant="primary")
        train_output = gr.Textbox(label="Resultado", lines=4)
        train_btn.click(run_train,
                        inputs=[train_target, train_model, train_estimators, train_depth],
                        outputs=train_output)

    with gr.Tab("5️⃣ Validación"):
        val_target = gr.Textbox(label="Columna Target")
        val_btn = gr.Button("Validar Modelo", variant="primary")
        val_metrics = gr.Textbox(label="Métricas", lines=10)
        val_plot = gr.Plot(label="Visualización")
        val_btn.click(run_validate, inputs=val_target, outputs=[val_metrics, val_plot])

    with gr.Tab("6️⃣ Publicar Dataset"):
        pub_data_btn = gr.Button("Generar Data Card y Preparar", variant="primary")
        pub_data_output = gr.Textbox(label="Resultado", lines=5)
        pub_data_card = gr.Textbox(label="DATA_CARD.md (editar antes de subir)", lines=15)
        pub_data_btn.click(run_publish_dataset, outputs=[pub_data_output, pub_data_card])
        gr.Markdown("👤 **Revisa y edita la Data Card. Luego sube manualmente:**\n"
                    "`uvx kaggle datasets create -p data/publish_dataset/ --dir-mode zip`")

    with gr.Tab("7️⃣ Publicar Modelo"):
        pub_model_btn = gr.Button("Generar Model Card y Preparar", variant="primary")
        pub_model_output = gr.Textbox(label="Resultado", lines=5)
        pub_model_card = gr.Textbox(label="MODEL_CARD.md (editar antes de subir)", lines=15)
        pub_model_btn.click(run_publish_model, outputs=[pub_model_output, pub_model_card])
        gr.Markdown("👤 **Revisa y edita la Model Card. Luego sube manualmente:**\n"
                    "`uvx kaggle models create -p data/publish_model/`")

demo.launch()
```

---

## APP 2: Inference (app_inference/app_inference.py)

Esta app se despliega a Hugging Face Spaces. **Descarga el modelo y preprocessor publicados en Kaggle** (la fuente de verdad del modelo) y expone una interfaz de predicción para el usuario final. El flujo es: modelo entrenado → publicado en Kaggle → descargado por la app en HF Spaces → predicción.

```python
#!/usr/bin/env python3
"""app_inference.py — UI de inferencia: consume modelo de Kaggle.

Descarga modelo publicado en Kaggle y expone interfaz de predicción.
Se despliega a Hugging Face Spaces.

Uso local:  uv run python app_inference/app_inference.py
Despliegue: uv run python scripts/08_deploy_hf.py
"""
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os
import subprocess
from pathlib import Path

# --- Configuración ---
# Estas variables se configuran como secrets en HF Spaces
# o se leen de config.yaml en local
KAGGLE_MODEL_REF = os.environ.get("KAGGLE_MODEL_REF", "username/mi-modelo")
KAGGLE_TOKEN = os.environ.get("KAGGLE_TOKEN", "")
MODEL_DIR = Path("model_cache")

def download_model_from_kaggle():
    """Descarga modelo desde Kaggle si no está en cache."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "model.joblib"
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"

    if model_path.exists() and preprocessor_path.exists():
        print("Modelo en cache, no se descarga de nuevo.")
        return model_path, preprocessor_path

    print(f"Descargando modelo desde Kaggle: {KAGGLE_MODEL_REF}")

    # Intentar descargar desde Kaggle
    env = os.environ.copy()
    if KAGGLE_TOKEN:
        env["KAGGLE_KEY"] = KAGGLE_TOKEN

    try:
        # Para modelos publicados como dataset en Kaggle
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_MODEL_REF,
             "--unzip", "-p", str(MODEL_DIR)],
            env=env, check=True, capture_output=True, text=True, timeout=120
        )
    except Exception:
        # Fallback: buscar archivos locales
        for local_path in [Path("models"), Path(".")]:
            if (local_path / "model.joblib").exists():
                import shutil
                shutil.copy(local_path / "model.joblib", model_path)
                if (local_path / "preprocessor.joblib").exists():
                    shutil.copy(local_path / "preprocessor.joblib", preprocessor_path)
                break

    return model_path, preprocessor_path


# --- Cargar modelo al iniciar ---
model_path, preprocessor_path = download_model_from_kaggle()
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Leer feature names del preprocessor
feature_names = []
if hasattr(preprocessor, "transformers_"):
    for name, transformer, columns in preprocessor.transformers_:
        if isinstance(columns, list):
            feature_names.extend(columns)


def predict(**kwargs):
    """Predicción con el modelo cargado."""
    input_df = pd.DataFrame([kwargs])
    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)[0]

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_processed)[0]
        classes = model.classes_
        return {str(c): float(p) for c, p in zip(classes, probas)}

    return str(prediction)


def predict_batch(file):
    """Predicción por lotes desde CSV."""
    if file is None:
        return "Sube un archivo CSV", None

    df = pd.read_csv(file.name)
    X_processed = preprocessor.transform(df)
    predictions = model.predict(X_processed)

    result_df = df.copy()
    result_df["prediction"] = predictions

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_processed)
        for i, cls in enumerate(model.classes_):
            result_df[f"proba_{cls}"] = probas[:, i]

    output_path = "predictions_output.csv"
    result_df.to_csv(output_path, index=False)

    return f"✅ {len(predictions)} predicciones generadas", result_df.head(20)


# --- UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="ML Predictor") as demo:
    gr.Markdown("# 🔮 ML Predictor")
    gr.Markdown(f"Modelo: `{KAGGLE_MODEL_REF}`")

    with gr.Tab("Predicción Individual"):
        gr.Markdown("Ingresa los valores para obtener una predicción.")
        # AJUSTAR: Definir inputs según las features del modelo
        # Ejemplo genérico — reemplazar con los inputs reales
        inputs = []
        for feat in feature_names:
            inputs.append(gr.Number(label=feat))

        predict_btn = gr.Button("Predecir", variant="primary")
        output = gr.Label(num_top_classes=5, label="Predicción")

        predict_btn.click(
            fn=predict,
            inputs=inputs,
            outputs=output
        )

    with gr.Tab("Predicción por Lotes"):
        gr.Markdown("Sube un CSV con las mismas columnas que el dataset de entrenamiento.")
        file_input = gr.File(label="Archivo CSV", file_types=[".csv"])
        batch_btn = gr.Button("Predecir Lote", variant="primary")
        batch_status = gr.Textbox(label="Estado")
        batch_results = gr.DataFrame(label="Resultados (preview)")

        batch_btn.click(
            fn=predict_batch,
            inputs=file_input,
            outputs=[batch_status, batch_results]
        )

    with gr.Tab("Información del Modelo"):
        gr.Markdown(f"""
        **Modelo**: `{type(model).__name__}`
        **Features**: {len(feature_names)}
        **Fuente**: Kaggle `{KAGGLE_MODEL_REF}`

        Este modelo fue entrenado, validado y publicado usando el pipeline
        de Data Science Assistant.
        """)

demo.launch()
```

### app_inference/requirements.txt

```
gradio==5.0.1
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4
joblib==1.4.2
kaggle==1.6.17
```

### app_inference/README.md (para HF Spaces)

```markdown
---
title: Mi Predictor ML
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.12.0"
app_file: app_inference.py
pinned: false
license: apache-2.0
---

# Mi Predictor ML

Modelo de machine learning entrenado, validado y publicado en Kaggle.
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

```python
# Temas predefinidos
demo = gr.Blocks(theme=gr.themes.Soft())    # Recomendado
demo = gr.Blocks(theme=gr.themes.Glass())
demo = gr.Blocks(theme=gr.themes.Monochrome())
```

## Consultar Documentación de Gradio

```
# MCP de Gradio (más preciso)
search_gradio_docs(query="Blocks Tab Row Column layout")
search_gradio_docs(query="File upload CSV processing")
load_gradio_docs()

# Tavily
tavily_skill(query="Gradio Blocks State management events", library="gradio", language="python")
```
