# Guía de Teoría: Cómo usar `rag-books-mcp` para fortalecer el diseño del modelo

> Este es el ÚNICO steering de teoría del Power. No contiene fórmulas ni derivaciones embebidas. Su función es enseñar al agente **cuándo y cómo** consultar al servidor MCP independiente `rag-books-mcp` (vectoriza ESL, ISLP, FES, PDSH y R4DS en ChromaDB) para tomar mejores decisiones al diseñar el código de entrenamiento.

## Filosofía

Las heurísticas operacionales viven en los `workflow-*.md` y `models-*.md` (cómo escribir el código).
La justificación teórica vive en los **libros**, accesibles vía `rag-books-mcp` (por qué el código se escribe así).

El agente NO debe memorizar teoría; debe **consultarla en tiempo de uso** cuando una decisión de diseño lo amerite.

## Libros indexados

| Slug | Libro | Lenguaje de ejemplos | Rol |
|---|---|---|---|
| `esl` | The Elements of Statistical Learning (Hastie, Tibshirani, Friedman) | matemático / R | Teoría rigurosa. Demostraciones, propiedades asintóticas. |
| `islp` | An Introduction to Statistical Learning with Python | Python | Teoría accesible con ejemplos prácticos. |
| `fes` | Feature Engineering and Selection (Kuhn, Johnson) | R / agnóstico | Heurísticas de FE: encoding, transformaciones, selección. |
| `pdsh` | Python Data Science Handbook (VanderPlas) | Python | Implementación operativa: pandas, NumPy, matplotlib, sklearn. |
| `r4ds` | R for Data Science, 2e (Wickham, Çetinkaya-Rundel, Grolemund) | **R / tidyverse** | EDA iterativo y data wrangling. Filosofía agnóstica del lenguaje. |

> ⚠️ **Importante sobre R4DS.** Está escrito íntegramente en R con tidyverse (`dplyr`, `ggplot2`, `tidyr`). **NUNCA generes código en R para el usuario a partir de R4DS.** El valor de R4DS son sus principios de exploración: el ciclo iterativo de EDA (preguntas → visualización → refinar), las heurísticas de qué mirar primero (variación, covariación, valores faltantes, valores inusuales), y la disciplina de tratar los datos tabularmente. Eso se traduce 1-a-1 a pandas / seaborn:
>
> | tidyverse (R4DS) | Equivalente pandas |
> |---|---|
> | `df %>% filter(x > 0)` | `df[df["x"] > 0]` |
> | `df %>% mutate(z = x + y)` | `df.assign(z=df["x"] + df["y"])` |
> | `df %>% group_by(g) %>% summarise(m = mean(x))` | `df.groupby("g")["x"].mean()` |
> | `ggplot(df, aes(x, y)) + geom_point()` | `sns.scatterplot(data=df, x="x", y="y")` |
> | `count(df, var)` / `geom_bar()` | `df["var"].value_counts()` / `sns.countplot` |
> | `geom_histogram(binwidth=...)` | `sns.histplot(..., binwidth=...)` |
>
> Cuando cites R4DS, **parafrasea la idea** y muestra el equivalente Python. Nunca pegues código R como solución al usuario.

## Limitación de R4DS en el HF Space (v2)

R4DS está bajo **CC BY-NC-ND 3.0 US** (NoDerivatives). Por esa razón solo está indexado para uso local:

- **v1 (`mcp-servers/rag-books-mcp/`, stdio)**: R4DS **disponible**. La colección `r4ds_chapters` vive en el `chroma_db/` del repo y no se redistribuye fuera de tu equipo.
- **v2 (`mcp-servers/rag-books-mcp-v2/`, HF Space + dataset HF Hub)**: R4DS **disponible solo si el server lee `chroma_db/` local** (vía `RAG_CHROMA_DIR`). El script `publish_chroma_dataset.py` está configurado para **excluir las colecciones marcadas `local_only=True`** antes de subir al Hub, por lo que el Space público de v2 **NO** contiene R4DS.

Consecuencia práctica para el agente:

- Si el usuario está conectado a v1 (stdio local) o a v2 con `RAG_CHROMA_DIR` apuntando a su `chroma_db/` reciente, `search_theory(book="r4ds", ...)` y `book="all"` incluirán R4DS sin problema.
- Si el usuario está conectado al Space público de v2, `search_theory(book="r4ds", ...)` devolverá vacío. Cuando esto ocurra, el agente debe explicar la situación con una sola frase ("R4DS no se publica en el Space por su licencia ND, está disponible si corres el RAG localmente") y continuar con los otros libros.

## Cuándo consultar el RAG (triggers)

Activa el RAG **antes de escribir o modificar código de entrenamiento** cuando ocurra alguno de estos eventos:

1. **Decisión de algoritmo**: vas a elegir entre familias (lineal vs árboles vs SVM vs ensemble vs DL) y necesitas justificar técnicamente la elección.
2. **Manejo de desbalance**: vas a decidir entre SMOTE / `class_weight` / `scale_pos_weight` / undersampling / focal loss.
3. **Regularización**: vas a elegir entre Ridge / Lasso / Elastic Net, o vas a fijar `reg_alpha` / `reg_lambda` / dropout / weight decay.
4. **Hiperparámetros sensibles a teoría**: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `max_features`, `C` en SVM, número de capas/unidades en NN.
5. **Cross-validation**: vas a elegir K, estrategia (KFold vs Stratified vs Group), o explicar por qué CV no se hace después de SMOTE.
6. **Calibración del umbral**: vas a justificar por qué 0.5 no es óptimo o por qué calibras con la curva PR.
7. **Selección de features**: vas a justificar correlación, VIF, mutual information, Lasso path.
8. **Métrica objetivo**: vas a defender el uso de F1 / PR-AUC / log loss / Brier score sobre accuracy.
9. **Stacking o ensembles**: vas a defender la elección del meta-modelo o de los modelos base.
10. **EDA con propósito**: vas a justificar qué mirar primero, qué hipótesis de calidad de datos validar antes de modelar (R4DS y FES son las fuentes naturales aquí).
11. **El usuario pregunta "¿por qué?"**: cualquier pregunta del usuario que empiece con "por qué", "qué dice la teoría", "está bien que…", "tiene sentido…".

> Si NO ocurre ninguno de estos triggers, **no consultes el RAG**. Escribir un Random Forest con defaults razonables no necesita citar ESL.

## Tools disponibles del servidor

| Tool | Cuándo usarla |
|---|---|
| `search_theory(query, book?, top_k?)` | Búsqueda semántica abierta. Default: usa esta primero. |
| `get_section(book, chapter, section)` | Cuando ya sabes la referencia exacta (ej. ESL §10.10) y quieres el texto completo. |
| `cite_foundation(topic, detail_level?)` | Para fundamentación lista para incluir en respuesta o comentario. `detail_level` ∈ {`intuitive`, `rigorous`}. |
| `list_available_topics()` | Solo si dudas si un tema está indexado. |

## Protocolo de consulta (3 pasos)

1. **Antes de codificar**, identifica qué decisión teórica respalda la línea de código que vas a escribir.
2. **Consulta el RAG** con la tool más específica que aplique. Prefiere `cite_foundation` para incorporar al respuesta o comentario; `search_theory` si exploras; `get_section` si tienes la referencia.
3. **Integra la cita** en uno de estos formatos:
   - **En respuesta al usuario**: paráfrasis + `[ESL §X.Y]` o `[ISLP §X.Y]`.
   - **En el código**: comentario breve referenciando la sección, sin copiar bloques largos del libro.
   - **En la Model Card**: sección "Fundamento Teórico" con citas explícitas.

## Tabla de queries por decisión de código

| Línea de código que vas a escribir | Tool | Query / parámetros sugeridos |
|---|---|---|
| `train_test_split(..., stratify=y)` | `search_theory` | `"stratified split classification imbalance"` |
| `SMOTE(...).fit_resample(X_train, y_train)` (post-split) | `cite_foundation` | `topic="SMOTE data leakage post split"` |
| `XGBClassifier(scale_pos_weight=...)` (sin SMOTE) | `search_theory` | `"class weight loss reweighting imbalance"`, `book="esl"` |
| `RandomForestClassifier(n_estimators=500)` | `cite_foundation` | `topic="random forest more trees never overfit"`, `detail_level="rigorous"` |
| `XGBClassifier(learning_rate=0.05, n_estimators=...)` | `get_section` | `book="esl", chapter=10, section="10.12"` (shrinkage) |
| `early_stopping_rounds=20` | `search_theory` | `"early stopping boosting overfitting"` |
| `RidgeCV` vs `LassoCV` vs `ElasticNetCV` | `cite_foundation` | `topic="ridge lasso elastic net when to use"` |
| `StandardScaler` antes de SVM/lineal | `search_theory` | `"feature scaling support vector machine ridge"` |
| `cross_val_score(..., cv=StratifiedKFold(5))` | `search_theory` | `"k-fold cross validation choice of K"` |
| Pipeline con `ColumnTransformer` + `fit` solo en train | `cite_foundation` | `topic="data leakage preprocessing fit transform"` |
| Calibración del umbral con curva PR | `search_theory` | `"precision recall threshold calibration imbalance"` |
| Stacking con meta-modelo lineal | `get_section` | `book="esl", chapter=8, section="8.8"` |
| Dropout / BatchNorm en NN | `search_theory` | `"dropout batch normalization regularization"` |
| `class_weight='balanced'` en logreg/RF | `cite_foundation` | `topic="class weight balanced inverse frequency"` |
| `PCA` antes de KMeans | `search_theory` | `"PCA preprocessing clustering curse of dimensionality"` |
| `OneHotEncoder` vs target encoding | `search_theory` | `"high cardinality categorical encoding tree models"` |
| Iniciar EDA: qué mirar primero (variación, valores faltantes, outliers) | `search_theory` | `"exploratory data analysis variation values missing"`, `book="r4ds"` |
| Decidir el flujo iterativo del EDA (preguntas → viz → refinar) | `cite_foundation` | `topic="exploratory data analysis iterative cycle questions"` |
| Justificar cuándo usar `geom_histogram` / `geom_freqpoly` (vs scatter/box) | `search_theory` | `"histogram boxplot covariation continuous categorical"`, `book="r4ds"` |
| Decidir si una variable categórica con muchos niveles necesita reagruparse | `search_theory` | `"factor lump small categories rare"`, `book="r4ds"` |

> Esta tabla es una guía, no una camisa de fuerza. Si tu decisión no aparece, formula el query tú mismo siguiendo el patrón.

## Cómo integrar la cita en el código

**Mal** (cita sin contexto, copia textual larga):
```python
# Según ESL §10.10: "Friedman (2001) introduced gradient boosting as a generalization
# of AdaBoost to arbitrary differentiable loss functions, where each new weak learner
# is fit to the negative gradient of the loss..."
xgb = XGBClassifier(learning_rate=0.05, n_estimators=500)
```

**Bien** (paráfrasis breve + referencia):
```python
# learning_rate bajo + n_estimators alto = shrinkage; reduce overfitting
# de boosting al precio de más cómputo. Ver ESL §10.12.
xgb = XGBClassifier(learning_rate=0.05, n_estimators=500, early_stopping_rounds=20)
```

**Bien** (en Model Card):
```markdown
## Fundamento Teórico

Se eligió `scale_pos_weight` sobre SMOTE por dos razones:
1. El dataset (1.29M filas, desbalance 172:1) sufriría una explosión a >2.5M filas con SMOTE [ESL §16.3].
2. `scale_pos_weight` reweighta la función de pérdida sin inyectar muestras sintéticas
   que pueden alterar la distribución real de la clase minoritaria [ESL Cap 16].
```

**Bien** (cita de R4DS para guiar EDA, pero con código en pandas):
```python
# Hadley en R4DS Cap. 10 propone que el EDA es un ciclo iterativo: hacer preguntas,
# buscar respuestas con visualización/transformación, refinar las preguntas. Empezamos
# por la variación de cada variable y la covariación con el target. [R4DS §10]
import seaborn as sns
sns.histplot(data=df, x="bill_length_mm", hue="species", element="step")  # variación
sns.boxplot(data=df, x="species", y="bill_length_mm")                      # covariación
```

> ❌ NUNCA generes la solución en R aunque la cita venga de R4DS. La cita es la idea, el código es Python.

## Modo degradado: cuando el RAG no responde

Si las tools `search_theory` / `cite_foundation` / `get_section` no están registradas o devuelven error:

1. **Declara explícitamente al usuario** que el servidor `rag-books-mcp` no está activo y que la respuesta no incluye citas verificables.
2. **No inventes números de sección** (§10.10, §7.2…). Si no consultaste el libro, no cites.
3. Procede con el conocimiento general del modelo, **etiquetando** la respuesta como "sin verificación bibliográfica".
4. Sugiere al usuario activar el servidor. El Power soporta dos variantes (idénticas en tools); ofrécelas en este orden:
   - **Default · HF Space** — sin instalación local, basta con una URL en `mcp.json`.
   - **Alternativa · Git Repo (stdio)** — para uso offline o auditoría de código.

   Pregunta cuál prefiere antes de generar la configuración. Detalles en `POWER.md` §"RAG sobre los Libros".

## Cuándo NO usar el RAG

- Para escribir código boilerplate (split, scaler, encoder, fit).
- Para correr un baseline rápido con defaults.
- Para tareas operativas (descargar dataset, publicar a HF Hub, desplegar Space).
- Cuando la decisión ya está estandarizada por los `workflow-*.md` o `models-*.md`.
- Cuando el costo de la consulta supera el valor de la cita (decisiones triviales).

## Conexión con el resto del Power

- `workflow-eda.md`, `workflow-feature-engineering.md`, `workflow-model-training.md`, `workflow-validation.md` → describen **cómo** ejecutar el pipeline.
- `models-linear-regression.md`, `models-trees-rf.md`, `models-ensemble.md`, `models-svm.md` → heurísticas específicas por familia.
- **Este steering** → cuándo y cómo consultar la teoría que respalda esas decisiones.

Si una pregunta del usuario es puramente teórica y no requiere generar código (ej. "explícame el bias-variance tradeoff"), responde con `cite_foundation` o `get_section` directamente, sin invocar workflows.
