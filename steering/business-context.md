---
inclusion: always
---

# Business Context (paso 0 obligatorio)

Antes de cualquier ingesta, EDA, FE, modelado o validación, el agente DEBE capturar el **contexto de negocio** del problema y materializarlo en `notes/00_business_context.md`. Sin ese documento, el resto del pipeline no tiene heurística válida para elegir métrica, costo asimétrico de errores ni umbral de utilidad.

> Este steering es **bloqueante** y **antecede** a `theory-driven-design.md`. La nota 01 de EDA (`notes/01_design_eda.md`) y el resto de notas de diseño (`02`, `03`, `04`) referencian este documento como insumo. Es la fuente de verdad para "qué error duele más" y "qué métrica reporta el éxito real".

---

## Regla 1: el cuestionario corto (5 preguntas obligatorias)

Cuando el usuario indique el dataset (después del saludo del power), el agente DEBE hacer las cinco preguntas siguientes, **en orden**, y esperar respuesta antes de continuar. No se asumen respuestas.

> **Antes de la pregunta 1**, el agente identifica el **régimen del problema** a partir del target del dataset y se lo confirma al usuario en una frase: *"Por la columna `<target>`, este es un problema de **regresión** / **clasificación binaria** / **clasificación multiclase**. Si me confirmas, te hago las cinco preguntas adaptadas a ese régimen."* El régimen determina cómo se redactan las preguntas 3, 4 y 5; las preguntas 1 y 2 son idénticas en todos los casos.

### Preguntas neutrales (todos los regímenes)

1. **¿Qué problema de negocio resolvemos?** Una o dos oraciones, en lenguaje del usuario, sin tecnicismos de ML.
   - *Clasificación:* "Detectar transacciones fraudulentas para bloquearlas antes del settlement."
   - *Regresión:* "Estimar el precio de venta de una vivienda para fijar el listing inicial sin sobre-pagar ni dejar dinero en la mesa."

2. **¿Qué decisión o acción dispara la predicción?** Qué se hace con el output del modelo, quién lo consume y con qué frecuencia.
   - *Clasificación:* "Si el score > umbral, se bloquea la transacción y se notifica al cardholder."
   - *Regresión:* "El precio estimado se muestra al broker como sugerencia inicial; el broker lo acepta o ajusta antes de publicar."

### Preguntas dependientes del régimen

> **Importante — escalera de fidelidad para las preguntas 3 y 4.** Pocas veces el usuario tiene USD/unidad por error. Eso está bien. Las preguntas 3 y 4 admiten **cuatro niveles de respuesta**, en orden de preferencia. El agente acepta el nivel más alto al que el usuario pueda llegar y lo declara en el documento. Subir o bajar de nivel después es válido (basta con editar la sección 3 del documento).
>
> 1. **Cuantitativo (ideal):** USD por evento (clasificación) o USD por unidad de error (regresión). Permite calcular ratio exacto y costo esperado en validación.
> 2. **Ratio cualitativo:** "1:1", "1:3", "1:10". Una oración que justifique la magnitud. Suficiente para fijar `scale_pos_weight`, `class_weight` o τ en pinball loss.
> 3. **Solo dirección:** "me duele más X que Y" o "simétrico". El agente deriva un ratio default razonable (ver tabla en Regla 6) y lo deja anotado para que el usuario lo confirme antes de pasar a EDA.
> 4. **No tengo idea:** el agente **propone** un default basado en la pregunta 1 (problema) y patrones típicos (ver Regla 7), explica el supuesto en una frase, y pide confirmación explícita. Si el usuario confirma, el supuesto queda **declarado como tal** en la sección 3 del documento, con la frase: *"Asunción no validada con el negocio: <ratio>. Revisar contra dueño del proceso antes de producción."*
>
> El nivel de respuesta se anota como `nivel: 1 | 2 | 3 | 4 (asunción)` en la tabla de la sección 3 del documento. Eso permite que la Model Card final declare honestamente sobre qué evidencia se calibró el modelo.

#### Si el problema es **clasificación**

3. **¿Cuál es el costo de un falso positivo (FP)?** Predecir la clase positiva cuando es negativa. El usuario responde al nivel más alto al que pueda llegar (ver escalera arriba).
   - *Nivel 1:* "Bloquear una transacción legítima cuesta ~$15 en fricción + 0.5% de churn."
   - *Nivel 2:* "Bloquear una transacción legítima duele **3× menos** que dejar pasar un fraude."
   - *Nivel 3:* "Me duele menos un FP que un FN."
   - *Nivel 4:* "No tengo idea." → el agente propone el default de la Regla 7 según el dominio.
4. **¿Cuál es el costo de un falso negativo (FN)?** No predecir la clase positiva cuando es positiva. Misma escala que la pregunta 3.
   - El **ratio costo_FN / costo_FP** se deriva de las respuestas; en niveles 2-3 ese ratio es la respuesta directa del usuario.
5. **¿Qué umbral mínimo de éxito hace que el modelo sea útil en producción?** Métrica primaria + valor mínimo, opcionalmente con métrica secundaria de soporte.
   - Ejemplo: "Recall ≥ 0.80 sobre la clase fraude **con** precision ≥ 0.30" o "F1 ≥ 0.65 sobre la minoritaria".
   - Métricas válidas: `recall`, `precision`, `F1`, `F-beta`, `PR-AUC`, `ROC-AUC`, `balanced_accuracy`. **No** aceptar `accuracy` solo si las clases están desbalanceadas.
   - Si el usuario no sabe el valor mínimo, el agente propone uno **a partir del baseline trivial** (predecir siempre la mayoritaria) más un margen razonable, y pide confirmación.

#### Si el problema es **regresión**

3. **¿Cuál es el costo de sobre-estimar el target?** Predecir un valor mayor al real. Mismo nivel de respuesta que en clasificación.
   - *Nivel 1:* "Listar 10% por encima cuesta ~30 días extra en mercado y ~3% de descuento al cierre."
   - *Nivel 2:* "Sobre-estimar duele **la mitad** que sub-estimar."
   - *Nivel 3:* "Me duele más sub-estimar que sobre-estimar." (dirección)
   - *Nivel 3 — caso especial:* "**Simétrico**, las dos direcciones duelen igual." → ratio `1:1`, métrica primaria por default = MAE.
   - *Nivel 4:* "No tengo idea." → el agente asume **simétrico** como default (ver Regla 7) y lo declara como asunción.
4. **¿Cuál es el costo de sub-estimar el target?** Predecir un valor menor al real. Misma unidad y nivel que la pregunta 3.
   - El **ratio costo_sub / costo_sobre** se deriva igual que en clasificación.
5. **¿Qué umbral mínimo de éxito hace que el modelo sea útil en producción?** Métrica primaria + valor máximo aceptable de error.
   - Ejemplo: "MAE ≤ $20K sobre el precio de vivienda" o "MAPE ≤ 8% sobre la demanda mensual" o "RMSE ≤ 1.5°C sobre la temperatura prevista a 24h".
   - Métricas válidas:
     - `MAE` — error promedio en unidades del target. Robusto a outliers. Default si no hay razón para otra.
     - `RMSE` — penaliza más los errores grandes. Usar si los outliers de error son operativamente caros.
     - `MAPE` / `sMAPE` — error relativo. Usar si el target tiene rango amplio (precios, demanda) y el negocio piensa en porcentajes.
     - `RMSLE` — error logarítmico. Usar si target es estrictamente positivo, sesgado, y el negocio tolera mejor el sub-estimado que el sobre-estimado (o viceversa con signo).
     - `R²` solo como métrica de soporte, **nunca** como métrica primaria de negocio.
     - `Pinball loss` / `quantile loss` — usar si las preguntas 3 y 4 son fuertemente asimétricas.
   - Si el usuario no sabe el valor máximo aceptable, el agente propone uno **a partir del baseline trivial** (predecir la media o la mediana del train) más una mejora razonable (ej. "MAE del baseline = $35K, propongo MAE ≤ $25K como umbral mínimo de utilidad"), y pide confirmación.

#### Si el problema es **multiclase** o **multilabel**

Mismo patrón que clasificación binaria pero la pregunta 3-4 se reformula como **matriz de costos** (qué confusión entre pares de clases es la cara y cuál es barata) o, si la matriz completa es inviable de capturar, simplificar a "qué clase es la crítica" y aplicar el patrón binario sobre `clase_crítica vs resto`. Métricas: `f1_macro`, `f1_weighted`, `balanced_accuracy`.

> El agente NO empieza EDA ni ingesta hasta tener las cinco respuestas (o "simétrico" justificado en la 3 y 4 si el costo es simétrico, o "matriz simplificada" en multiclase).

---

## Regla 2: materializar en `notes/00_business_context.md`

Antes de ejecutar cualquier script, el agente debe escribir `notes/00_business_context.md` con esta estructura mínima. Hay **dos plantillas** según el régimen del problema; el agente elige la que corresponda.

### Plantilla A — Clasificación (binaria o multiclase)

```markdown
# Contexto de Negocio: <proyecto>

> Régimen: clasificación <binaria | multiclase | multilabel>. Target: `<columna>`. Clases: `<lista>`.

## 1. Problema
<respuesta a la pregunta 1, copiada o reformulada con permiso del usuario>

## 2. Decisión que dispara la predicción
<respuesta a la pregunta 2>
- **Consumidor del modelo:** <quién o qué sistema usa el output>
- **Frecuencia:** <batch / online / on-demand>

## 3. Costos asimétricos de error
| Error | Descripción concreta | Costo estimado | Nivel | Escala |
|---|---|---|---|---|
| Falso Positivo (FP) | <qué pasa cuando se predice positivo y el real es negativo> | <USD / horas / cualitativo / ratio> | <1\|2\|3\|4> | <alto/medio/bajo> |
| Falso Negativo (FN) | <qué pasa cuando se predice negativo y el real es positivo> | <USD / horas / cualitativo / ratio> | <1\|2\|3\|4> | <alto/medio/bajo> |

**Asimetría:** <cuál duele más y por qué>. **Ratio costo_FN / costo_FP ≈ <valor>** (ej. 10:1 si un FN cuesta 10× lo que un FP).

**Nivel de evidencia** (ver `business-context.md` Regla 1):
- 1 = cuantitativo (USD/evento real). 2 = ratio cualitativo justificado. 3 = solo dirección. 4 = asunción no validada (default por dominio, Regla 7).
- Si nivel = 4, agregar: *"Asunción no validada con el negocio: `<ratio_default>` (default por dominio "<dominio>"). Revisar contra dueño del proceso antes de producción."*

> Para multiclase: reemplazar la tabla por una **matriz de costos** entre pares de clases, o declarar "matriz simplificada: clase crítica = `<X>`" y aplicar el patrón binario sobre `<X> vs resto`.

## 4. Métrica primaria y umbral de utilidad
- **Métrica primaria:** <recall@k / precision / F1 / F-beta / PR-AUC / balanced_accuracy / f1_macro / f1_weighted>
- **Umbral mínimo de utilidad:** <valor numérico, p.ej. recall ≥ 0.80>
- **Restricción cruzada (si aplica):** <p.ej. precision ≥ 0.30 al alcanzar ese recall>
- **Métricas secundarias a reportar:** <2-3 métricas de soporte (ROC-AUC, confusion matrix, etc.)>

## 5. Implicaciones para el pipeline
| Fase | Implicación derivada del contexto |
|---|---|
| EDA (`workflow-eda.md`) | <qué mirar primero (distribución del target, balance de clases, segmentos críticos)> |
| FE (`workflow-feature-engineering.md`) | <split estratificado por clase, encoding de categóricas relevantes, tratamiento de la minoritaria> |
| Modelado (`workflow-model-training.md`) | <`scoring='f1'` / `scoring='recall'`; `class_weight='balanced'` o `scale_pos_weight = ratio_FN/FP`; SMOTE si la sección 4 lo justifica> |
| Validación (`workflow-validation.md`) | <calibrar umbral con curva PR para satisfacer la sección 4; reportar matriz de confusión con costos esperados> |

## 6. Restricciones (opcional, "no aplica" si el usuario lo confirma)
- **Interpretabilidad requerida:** <alta / media / baja>. Justificación si aplica (regulación, auditoría, explicación al cliente).
- **Latencia tolerable en inferencia:** <ms / segundos / no aplica>.
- **Restricciones regulatorias o de fairness:** <ej. GDPR, exclusión de variables protegidas, evaluación por subgrupo>.

## 7. Baseline contra el que comparamos (opcional)
<heurística actual o sistema previo, si existe; si no, "no aplica">.
```

### Plantilla B — Regresión

```markdown
# Contexto de Negocio: <proyecto>

> Régimen: regresión. Target: `<columna>` (`<unidad>`). Rango observado: `[<min>, <max>]`. Distribución: `<simétrica | sesgada-derecha | sesgada-izquierda>`.

## 1. Problema
<respuesta a la pregunta 1>

## 2. Decisión que dispara la predicción
<respuesta a la pregunta 2>
- **Consumidor del modelo:** <quién o qué sistema usa el output>
- **Frecuencia:** <batch / online / on-demand>
- **Tolerancia de la decisión al error:** <ej. "el broker ajusta el precio si difiere ±5% de su intuición; errores menores no se actúan">

## 3. Costos de error por dirección
| Dirección | Descripción concreta | Costo por unidad de error | Nivel | Escala |
|---|---|---|---|---|
| Sobre-estimar (`ŷ > y`) | <qué pasa al predecir un valor mayor al real> | <USD por unidad / % por punto / cualitativo / ratio> | <1\|2\|3\|4> | <alto/medio/bajo> |
| Sub-estimar (`ŷ < y`) | <qué pasa al predecir un valor menor al real> | <USD por unidad / % por punto / cualitativo / ratio> | <1\|2\|3\|4> | <alto/medio/bajo> |

**Asimetría:** <`simétrica` | `costo_sub / costo_sobre ≈ <valor>`>. Si es asimétrica, anotar la dirección dolorosa y el ratio.

**Nivel de evidencia** (ver `business-context.md` Regla 1):
- 1 = cuantitativo (USD/unidad real). 2 = ratio cualitativo justificado. 3 = solo dirección o "simétrico". 4 = asunción no validada (default por dominio, Regla 7).
- Si nivel = 4, agregar: *"Asunción no validada con el negocio: `<ratio_default>` (default por dominio "<dominio>"). Revisar contra dueño del proceso antes de producción."*

**Forma del costo:** <`lineal` (cada unidad cuesta lo mismo) | `cuadrática` (errores grandes duelen desproporcionadamente) | `umbral` (errores < X son gratis, errores ≥ X cuestan)>. Esto guía la elección entre MAE, RMSE y pérdidas custom.

## 4. Métrica primaria y umbral de utilidad
- **Métrica primaria:** <MAE | RMSE | MAPE | sMAPE | RMSLE | pinball_loss(τ=<valor>)>
- **Justificación de la elección:**
  - MAE si el costo es **lineal y simétrico** y los outliers de error no son especialmente caros.
  - RMSE si los **errores grandes son desproporcionadamente caros** (costo cuadrático).
  - MAPE/sMAPE si el negocio piensa en **error relativo (%)** y el target tiene rango amplio.
  - RMSLE si el target es **estrictamente positivo, sesgado**, y se justifica una transformación log.
  - Pinball loss con τ ≠ 0.5 si los costos son **asimétricos** (sobre vs sub-estimar).
- **Umbral mínimo de utilidad:** <ej. `MAE ≤ $20K`, `MAPE ≤ 8%`, `RMSE ≤ 1.5°C`>.
- **Restricción cruzada (si aplica):** <ej. `MAE ≤ $20K Y residuales sin sesgo sistemático por segmento>`.
- **Métricas secundarias a reportar:** <R² como soporte (NUNCA como primaria), residuales por segmento, error por bin del target>.

## 5. Implicaciones para el pipeline
| Fase | Implicación derivada del contexto |
|---|---|
| EDA (`workflow-eda.md`) | <skewness del target → ¿amerita `log1p`?>; <segmentos del target con cola pesada que merecen atención>; <outliers operativos vs erróneos>. |
| FE (`workflow-feature-engineering.md`) | <transformación del target (`log1p` si sección 4 usa MAPE/RMSLE o si la distribución es muy sesgada-derecha)>; <split estratificado por **bins del target** con `pd.qcut`, no aleatorio>; <clipping de outliers operativamente justificado>. |
| Modelado (`workflow-model-training.md`) | <`scoring='neg_mean_absolute_error'` / `'neg_root_mean_squared_error'` según sección 4>; <si asimétrico fuerte: `XGBRegressor(objective='reg:pseudohubererror')` o **quantile regression** con τ derivado del ratio costo_sub / costo_sobre>; <`np.clip(pred, 0, None)` si el target es físicamente acotado y el modelo lineal puede extrapolar negativo>. |
| Validación (`workflow-validation.md`) | <reportar la métrica primaria de la sección 4 + residuales por bin del target + heteroscedasticidad>; <criterio "pasa/no pasa" = umbral mínimo de la sección 4>; <si asimetría: reportar **error medio por dirección** (`mean(ŷ-y | ŷ>y)` vs `mean(y-ŷ | ŷ<y)`) además de la métrica global>. |

## 6. Restricciones (opcional, "no aplica" si el usuario lo confirma)
- **Interpretabilidad requerida:** <alta / media / baja>.
- **Latencia tolerable en inferencia:** <ms / segundos / no aplica>.
- **Cota física del target:** <ej. "precio ≥ 0", "porcentaje ∈ [0, 100]">; si aplica, el código de inferencia debe `np.clip` la predicción.
- **Restricciones regulatorias o de fairness:** <ej. evaluación de error por subgrupo demográfico>.

## 7. Baseline contra el que comparamos (opcional)
<heurística actual (ej. "media histórica del segmento", "regla del broker = comparable_price × 1.05"), si existe; si no, "no aplica">.
```

> El documento es **breve por diseño** (≤ 1 página renderizada). No es una propuesta comercial; es la fuente de verdad operativa para decisiones de modelado.

---

## Regla 3: cómo el contexto se propaga a cada fase

La propagación cambia según el régimen del problema. Las dos tablas siguientes son las decisiones concretas que el agente debe tomar en cada fase a partir del documento.

### Régimen: Clasificación

| Fase | Qué del contexto se usa | Decisión que cambia |
|---|---|---|
| EDA (`workflow-eda.md`) | Pregunta 1, distribución del target | Mirar **balance de clases** primero; identificar la clase crítica; revisar cardinalidad de categóricas relevantes para el negocio. |
| FE (`workflow-feature-engineering.md`) | Pregunta 1, clase crítica | Split estratificado **por clase**; encoding sensible a cardinalidad; evaluar SMOTE solo si la sección 4 lo justifica. |
| Modelado (`workflow-model-training.md`) | Pregunta 4 (métrica), Pregunta 3 vs 4 (ratio) | `scoring=` la métrica primaria literal (`'f1'`, `'recall'`, `'average_precision'` para PR-AUC); `class_weight='balanced'` o `scale_pos_weight = costo_FN / costo_FP`. |
| Validación (`workflow-validation.md`) | Pregunta 4 (umbral), Pregunta 3 vs 4 | Calibrar el **umbral** en la curva precision-recall hasta alcanzar el mínimo de la sección 4; reportar **costo esperado** = `FP × costo_FP + FN × costo_FN`. |

### Régimen: Regresión

| Fase | Qué del contexto se usa | Decisión que cambia |
|---|---|---|
| EDA (`workflow-eda.md`) | Pregunta 1, distribución y rango del target | Skewness y rango del target → ¿amerita `log1p`?; segmentos del target con cola pesada; outliers operativos vs erróneos. |
| FE (`workflow-feature-engineering.md`) | Pregunta 4 (métrica) y forma del costo | Transformar target con `np.log1p(y)` si la métrica es MAPE o RMSLE, o si la distribución es muy sesgada-derecha; **split estratificado por bins del target** con `pd.qcut(y, q=10)` (no aleatorio); clipping de outliers operativos. |
| Modelado (`workflow-model-training.md`) | Pregunta 4 (métrica), Pregunta 3 vs 4 (asimetría) | `scoring='neg_mean_absolute_error'` o `'neg_root_mean_squared_error'` o `'neg_mean_absolute_percentage_error'` según la sección 4. Si el ratio es asimétrico fuerte: `XGBRegressor(objective='reg:quantileerror', quantile_alpha=τ)` con τ derivado del ratio (`τ = costo_sub / (costo_sub + costo_sobre)`). Si el target es físicamente acotado: aplicar `np.clip(pred, low, high)` en inferencia. |
| Validación (`workflow-validation.md`) | Pregunta 4 (umbral) | Reportar la métrica primaria contra el umbral mínimo (criterio pasa/no pasa); analizar **residuales por bin del target** y por segmento; si asimetría, reportar **error medio por dirección** (`mean(residual | residual>0)` vs `mean(residual | residual<0)`). |

### Común a ambos regímenes

| Fase | Implicación |
|---|---|
| Publicación (`huggingface-workflows.md`) | Resumen de negocio (sección 1, 2 y 4 del documento) en la **Model Card** y **Data Card**. Métrica primaria destacada en el header de la Model Card. |

---

## Regla 4: relación con `theory-driven-design.md`

`business-context.md` se ejecuta **antes** que cualquier consulta al RAG. La razón:

- `theory-driven-design.md` (RAG) responde **cómo** se resuelve la decisión técnica (ej. "¿Ridge o Lasso para esta señal?", "¿logit o probit?").
- `business-context.md` responde **por qué** la decisión es importante (ej. "los falsos negativos cuestan 10× lo que los falsos positivos, así que recall manda").

Sin contexto de negocio, las consultas al RAG son genéricas y producen citas correctas pero descontextualizadas. Con contexto, las consultas se afinan: en vez de `"class imbalance metrics"` el agente pregunta `"recall vs precision tradeoff cost sensitive learning fraud detection"`. La calidad de la cita depende de la calidad del query, y la calidad del query depende del contexto.

Orden estricto:

```
01_ingest.py            (puede correr sin notas)
       │
       ▼
notes/00_business_context.md   ← BLOQUEANTE para todo lo siguiente
       │
       ▼
notes/01_design_eda.md    (parte pre-EDA: preguntas e hipótesis)
       │
       ▼
02_eda.py
       │
       ▼
notes/01_design_eda.md    (parte post-EDA: hallazgos)
       │
       ▼
notes/02_design_fe.md → 03_feature_engineering.py → ... (resto del flujo)
```

---

## Regla 5: excepciones (cuándo el contexto puede ser mínimo)

El cuestionario completo se puede acortar a la pregunta 1 + pregunta 5 en estos casos, declarándolo explícitamente en la sección 3 del documento:

- **Datasets pedagógicos canónicos** (Iris, Titanic, Boston Housing, Penguins, MNIST, Wine) usados con fines de demo o aprendizaje, no producción. Justificación: el problema de negocio es "ejemplificar el flujo del power"; no hay decisión real ni costos diferenciales.
- **Pruebas de smoke / benchmarking interno** del propio power. Documentar la finalidad ("validar que el pipeline corre end-to-end con dataset X").

En cualquier otro caso (proyecto real, dataset traído por el usuario, deploy con intención de uso), el cuestionario completo es obligatorio.

---

## Regla 6: cómo formula el agente las preguntas

El agente NO presenta el cuestionario como una tabla rígida. Sigue este patrón en dos pasos:

**Paso A — Identificar el régimen (una sola frase, no es pregunta):**

> "Por la columna `<target>` y por su distribución, este es un problema de **<regresión | clasificación binaria | clasificación multiclase>**. Si me confirmas, te hago las cinco preguntas adaptadas a ese régimen. Si crees que el régimen es otro (por ejemplo, quieres tratar un target numérico como categórico vía bins), avísame antes."

**Paso B — Cuestionario, una pregunta por turno:**

> "Antes de tocar los datos, déjame entender el contexto de negocio para elegir bien la métrica y el costo de los errores. Son cinco preguntas cortas. Las preguntas 3 y 4 son las que más cuesta responder, así que admiten cuatro niveles de respuesta — empezamos por el ideal, y si no tienes los datos, bajamos a un default razonable.
>
> **1.** En tus palabras, ¿qué problema de negocio quieres resolver con este dataset?"

Después de cada respuesta, parafrasea para confirmar y pasa a la siguiente. Si el usuario contesta varias preguntas a la vez, reorganiza la información en el documento y solo vuelve a preguntar lo que falte.

**Manejo del titubeo en preguntas 3-4** (lo más común):

Cuando el usuario diga *"no sé bien"*, *"no tengo datos"*, *"no soy yo el dueño del negocio"* o similar, el agente NO insiste pidiendo USD. Aplica la escalera de fidelidad de la Regla 1 en este orden:

1. **Ofrecer ratio cualitativo:** *"¿Sabrías al menos si uno duele más que el otro? Por ejemplo, ¿1:1, 1:3, 1:10?"*
2. **Si tampoco, ofrecer dirección:** *"¿Y al menos la dirección? ¿Cuál de los dos errores te incomoda más?"*
3. **Si nada de lo anterior, aplicar Regla 7:** proponer el default por dominio inferido de la pregunta 1, declarar que es **asunción no validada**, y pedir confirmación. Mostrar la fila de la tabla literal: *"Para tu dominio (`<dominio>`), el default típico es `<ratio>`. Lo anoto como asunción no validada y avanzamos. ¿Te parece bien?"*

**Para regresión simétrica** (los costos por sobre-estimar y sub-estimar son iguales), la pregunta 4 se colapsa: el usuario dice *"simétrico"* y se anota `1:1` en el documento, lo que dispara MAE como métrica primaria por default.

**Para la pregunta 5 cuando el usuario no sabe el valor mínimo/máximo**, el agente propone uno **a partir del baseline trivial** (mayoritaria en clasificación, media/mediana en regresión) más un margen razonable, y pide confirmación. La justificación va en el documento.

---

## Regla 7: defaults por dominio cuando el usuario no tiene datos (nivel 4)

Cuando el usuario contesta "no tengo idea" en las preguntas 3-4, el agente NO se rinde ni omite la información. Propone un default basado en patrones típicos del dominio inferido de la pregunta 1, lo declara como **asunción no validada** y pide confirmación. Esta tabla es el catálogo de defaults razonables.

### Defaults para clasificación

| Dominio típico (de la pregunta 1) | Default `costo_FN : costo_FP` | Justificación de una línea |
|---|---|---|
| Detección de fraude / anomalía / amenaza | `10 : 1` | Dejar pasar el caso crítico cuesta órdenes de magnitud más que una falsa alarma. |
| Diagnóstico médico (cribado) | `10 : 1` o más | No detectar la enfermedad cuesta más que una prueba confirmatoria adicional. |
| Mantenimiento predictivo / detección de fallas | `5 : 1` | Falla no detectada → daño operativo; falsa alarma → inspección barata. |
| Aprobación de crédito (default = clase positiva) | `3 : 1` | Aprobar un mal crédito duele más que rechazar uno bueno, pero no de forma extrema. |
| Marketing / propensión a compra | `1 : 3` | Mandar mail al que no iba a comprar es barato; perderse al que sí iba a comprar es relativamente caro pero el contacto cuesta. La asimetría se invierte. |
| Recomendación / ranking | `1 : 1` (o métrica de ranking) | Falsos positivos y negativos suelen pesar parecido; típicamente se reporta NDCG / MAP en vez de precision/recall. |
| Clasificación de imágenes / NLP general / problemas balanceados | `1 : 1` | Sin asimetría declarada, F1 macro o accuracy con clases balanceadas. |
| Dataset pedagógico canónico (Iris, Penguins, MNIST, Wine) | `1 : 1` | Sin contexto de negocio real; usar accuracy o F1 macro, declararlo en la sección 7 (Excepción). |

### Defaults para regresión

| Dominio típico (de la pregunta 1) | Default `costo_sub : costo_sobre` | Métrica primaria sugerida | Justificación de una línea |
|---|---|---|---|
| Precio de venta / valuación de activo | `1 : 1` (simétrico) | MAE | Sobre-pagar y sub-pagar duelen parecido en ausencia de info. Si el modelo se usa para listing inicial, puede inclinarse a sub-estimar para vender rápido (`1.5 : 1`). |
| Predicción de demanda / inventario | `2 : 1` | MAE o pinball(τ=0.66) | Sub-estimar la demanda → stockout (venta perdida + frustración). Sobre-estimar → costo de inventario (más barato). |
| Predicción de tiempo de entrega / ETA | `3 : 1` (sub-estimar duele más) | MAE o pinball(τ=0.75) | Llegar tarde respecto a lo prometido erosiona confianza; llegar antes es bonus. |
| Predicción meteorológica (temperatura, lluvia) | `1 : 1` | RMSE | Errores grandes son operativamente caros; la dirección suele ser simétrica. |
| Predicción de consumo energético / carga | `2 : 1` (sub-estimar duele más) | MAE | Sub-estimar la demanda → apagones / costo de balancing. Sobre-estimar → exceso de generación, más barato. |
| Predicción financiera / ROI esperado | `1 : 1` | MAE o RMSE | Default conservador; la asimetría se valida con el dueño del modelo. |
| Tiempo de respuesta de incidente / SLA | `3 : 1` (sub-estimar duele más) | MAE o pinball(τ=0.75) | Romper SLA cuesta penalización; cumplirlo con margen es gratis. |
| Salud (lectura de signos vitales, dosificación) | depende, **siempre confirmar con experto** | MAE | NO usar default sin confirmación clínica. Pedir explícitamente al usuario que valide con personal médico antes de fijar la métrica. |
| Dataset pedagógico canónico (Boston Housing, USA Housing, Salary, Diabetes) | `1 : 1` (simétrico) | MAE o RMSE | Sin contexto de negocio real; usar MAE como default y declararlo en la sección 7 (Excepción). |

### Cómo se declara la asunción en el documento

Cuando el agente usa un default de esta tabla, la sección 3 de `notes/00_business_context.md` debe contener literalmente:

```markdown
**Asimetría:** Asunción no validada con el negocio: `<ratio_default>` (default por dominio "<dominio>" según `business-context.md` Regla 7). Revisar contra dueño del proceso antes de producción.
```

Eso preserva la trazabilidad: cualquiera que abra la Model Card después sabe que el ratio de costos no salió de un análisis de negocio sino de un default razonable, y puede pedir validación.

### Lo que el agente dice al usuario (Nivel 4)

Si el usuario contesta "no sé":

> "Está bien. Para tu problema de `<dominio inferido de la pregunta 1>`, el patrón típico es `<ratio_default>` (`<justificación de una línea>`). Voy a anotarlo en el documento como **asunción no validada** y seguimos con eso. Si más adelante consigues los datos reales o validas con el dueño del proceso, basta con editar la sección 3 del documento y vuelvo a calibrar `class_weight` / `scoring` / `τ`. ¿Te parece bien proceder con `<ratio_default>` por ahora?"

El agente espera el "sí" antes de continuar. Si el usuario no acepta el default, ofrece dos alternativas: (a) bajar a nivel 3 con solo dirección, o (b) abandonar la asimetría y trabajar con `1:1` declarado como simétrico.

---

## Anti-patrones (qué NO hacer)

1. **Asumir el problema** a partir del nombre del dataset. `iris` no implica "queremos clasificar especies para el negocio X". Hay que preguntar.
2. **Saltarse el cuestionario** porque "es un dataset conocido". Solo aplica la excepción de la Regla 5 con declaración explícita.
3. **Producir `notes/00_business_context.md` después** de entrenar el modelo. Eso es decoración, no diseño.
4. **Usar la métrica por defecto del power** (accuracy, R²) sin verificar contra la pregunta 5. La métrica primaria sale del negocio, no de un default.
5. **Convertir el documento en un brief de marketing** de 5 páginas. Si pasa de una página, sobra contenido.
6. **Aplicar la plantilla de clasificación a un problema de regresión** (o viceversa). El régimen se confirma **antes** de redactar las preguntas 3-5; FP/FN no aplican a regresión y "umbral de decisión" no aplica salvo que el output se discretice.
7. **Reportar R² como métrica primaria de regresión.** R² es una métrica de bondad de ajuste relativa al baseline de la media, no un costo de negocio. Sirve como soporte. La métrica primaria es MAE, RMSE, MAPE, RMSLE o pinball loss según las preguntas 3 y 4.
8. **Usar `train_test_split` aleatorio en regresión** cuando el target tiene rango amplio o está sesgado. El split debe ser **estratificado por bins del target** (`pd.qcut(y, q=10)`) para que el train y el test tengan distribución comparable. Esta decisión sale del `notes/00_business_context.md` y se materializa en `notes/02_design_fe.md`.
