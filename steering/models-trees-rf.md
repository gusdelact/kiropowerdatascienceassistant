# Modelos: Árboles de Regresión y Random Forest

> Cubre: fundamentos de árboles de regresión (CART), criterio de splits, Random Forest Regressor con `GridSearchCV`, comparación contra modelos lineales, ExtraTrees como variante.
>
> Ejemplo: dataset Boston Housing (506 obs, 13 predictores).
>
> **Nota sobre paralelización:** Los ejemplos usan `n_jobs=-1` por simplicidad, pero en producción usar máximo 70% de los cores: `n_jobs=max(int(os.cpu_count() * 0.7), 1)`. Ver `workflow-model-training.md` §Heurísticas de Escalabilidad para la configuración completa.

Esta guía cubre **árboles individuales (CART) y Random Forest para regresión y clasificación**. Para modelos de boosting (XGBoost) y bagging aplicado a clasificación con desbalance, ver `models-ensemble.md`.

---

## Cuándo Usar Esta Familia

| Situación | Recomendación |
|---|---|
| Relaciones no lineales o con interacciones | Random Forest |
| Necesitas un modelo robusto sin tuning intensivo | Random Forest |
| Quieres interpretabilidad simple (un árbol visualizable) | Decision Tree (limitar `max_depth`) |
| Clases o features con pocos datos por categoría | Árboles (manejan bien) |
| Necesitas el mejor rendimiento posible | Cambiar a Boosting (`models-ensemble.md`) |
| Datos puramente lineales | Cambiar a `models-linear-regression.md` |

---

## 1. Decision Tree (Árbol de Regresión Individual)

### Idea Intuitiva

Un árbol de regresión particiona el espacio de features en regiones rectangulares. Cada hoja predice el promedio (regresión) o la clase mayoritaria (clasificación) de los datos que cayeron en ella.

### Criterio de Split

Para regresión, en cada nodo se elige la variable y el umbral que minimizan la **suma del error cuadrático** después del split:

$$\text{Split óptimo} = \arg\min_{j, s} \left[\sum_{i \in R_1} (y_i - \bar{y}_{R_1})^2 + \sum_{i \in R_2} (y_i - \bar{y}_{R_2})^2 \right]$$

### Implementación

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('boston_house_prices.csv')
X = df.loc[:, 'CRIM':'LSTAT']
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Búsqueda de hiperparámetros
param_grid = {
    'max_depth': [3, 5, 7, 9, 11, 13, 15, None],
    'min_samples_split': [2, 5, 10, 20, 30],
    'min_samples_leaf': [1, 5, 10, 20]
}

tree = DecisionTreeRegressor(random_state=42)
grid = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Mejores parámetros: {grid.best_params_}")
y_pred = grid.predict(X_test)
print(f"MSE Árbol: {mean_squared_error(y_test, y_pred):.4f}")
```

### Visualizar el Árbol

```python
plt.figure(figsize=(20, 10))
plot_tree(grid.best_estimator_, feature_names=X.columns, filled=True, max_depth=3, fontsize=10)
plt.savefig('outputs/figures/arbol_regresion.png', dpi=150, bbox_inches='tight')
```

### Hiperparámetros Clave

| Parámetro | Efecto al aumentar | Recomendación |
|---|---|---|
| `max_depth` | ↑ Profundidad → más complejidad → más varianza | 5-15. Sin límite tiende a overfittear |
| `min_samples_split` | ↑ → menos splits → menos overfit | 2-30 |
| `min_samples_leaf` | ↑ → hojas con más datos → predicciones más estables | 1-20 |
| `max_features` | ↓ → más aleatoriedad | `'sqrt'` o `'log2'` para clasificación |
| `criterion` | `'squared_error'` (regresión) / `'gini'`, `'entropy'` (clasificación) | Default suele bastar |

---

## 2. Random Forest (Bagging sobre Árboles)

### Idea (Notebooks 4 y 2)

```
1. Tomar B muestras bootstrap del dataset
2. Para cada muestra, entrenar un árbol usando solo m features aleatorias en cada split
3. Predicción final:
   - Regresión: promedio de los B árboles
   - Clasificación: voto mayoritario
```

**Razón clave:** Promediar muchos árboles reduce la varianza. Tomar features aleatorias reduce la correlación entre árboles, multiplicando el efecto del promedio.

### Implementación para Regresión

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 12, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 1.0]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train_s, y_train)

print(f"Mejores hiperparámetros: {grid.best_params_}")
y_pred = grid.predict(X_test_s)
print(f"MSE Random Forest: {mean_squared_error(y_test, y_pred):.4f}")
```

> **Nota:** Random Forest **no requiere estandarización** (los splits son invariantes a escala), pero estandarizar no daña y permite reusar el mismo `preprocessor.joblib`.

### Implementación para Clasificación

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    oob_score=True   # estimación gratis del test error
)
rf_clf.fit(X_train, y_train)

print(f"OOB Score: {rf_clf.oob_score_:.4f}")
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Out-of-Bag (OOB) Error

Cada muestra bootstrap usa ~63.2% de los datos. El ~36.8% restante sirve como validation set gratuito:

```python
rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")  # estimación insesgada del test error
```

**Ventaja:** No necesitas cross-validation para una primera estimación. Útil con datasets grandes donde CV es caro.

### Hiperparámetros Clave

| Parámetro | Efecto | Recomendación |
|---|---|---|
| `n_estimators` (B) | ↑ → menos varianza, nunca causa overfit | 100-500. Más es mejor pero rendimientos decrecientes |
| `max_features` (m) | ↓ → menos correlación entre árboles, pero más bias | `'sqrt'` para clasificación, `1.0` o `1/3` para regresión |
| `max_depth` | ↑ → más complejidad por árbol | 5-15. Profundo está bien si hay suficientes árboles |
| `min_samples_split` | ↑ → menos splits → más regularización | 2-10 |
| `min_samples_leaf` | ↑ → hojas más estables | 1-5 |
| `bootstrap` | `True` (default) habilita OOB | Mantener `True` |

> **Insight clave (ESL):** Random Forest **no puede sobreajustar al aumentar `n_estimators`**. Cada árbol adicional es solo un promedio más, no aumenta la complejidad efectiva del modelo.

---

## 3. ExtraTrees

**Variante de Random Forest con mayor aleatoriedad:**

| Aspecto | Random Forest | Extra Trees |
|---|---|---|
| Búsqueda de umbral | El **óptimo** dentro de las features candidatas | **Aleatorio** dentro del rango de la feature |
| Velocidad | Más lento | Más rápido |
| Sobreajuste | Riesgo bajo | Riesgo aún más bajo |
| Bias | Menor | Mayor |
| Varianza | Mayor | Menor |

```python
from sklearn.ensemble import ExtraTreesRegressor

et = ExtraTreesRegressor(
    n_estimators=200,
    max_features=1.0,    # más features candidatas porque el umbral es aleatorio
    bootstrap=False,     # default en ExtraTrees
    random_state=42,
    n_jobs=-1
)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
print(f"MSE ExtraTrees: {mean_squared_error(y_test, y_pred):.4f}")
```

**Cuándo elegir ExtraTrees sobre Random Forest:**
- Datasets grandes donde la velocidad importa
- Cuando RF muestra signos de overfitting
- Como segundo modelo en un benchmarking

---

## 4. Benchmarking

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

modelos = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=0.1),
    'Lasso': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
    'Árbol de regresión': DecisionTreeRegressor(max_depth=9, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestRegressor(max_depth=9, n_estimators=150, random_state=42, n_jobs=-1),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1),
}

def comparar_modelos(X_train, X_test, y_train, y_test, modelos):
    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        resultados[nombre] = mse
        print(f"{nombre:25s} - MSE: {mse:.4f}")
    return resultados

resultados = comparar_modelos(X_train, X_test, y_train, y_test, modelos)
```

**Resultado típico en Boston Housing:**

| Modelo | MSE |
|---|---|
| OLS, Ridge, Lasso, ElasticNet | 22-25 |
| Árbol de regresión | ~11 |
| Random Forest | ~10 |
| ExtraTrees | ~10 |

> **Conclusión:** Las relaciones en Boston Housing son no lineales — los árboles superan ampliamente a los modelos lineales.

---

## 5. Feature Importance

### Gini/MSE Importance (rápido pero sesgado)

```python
import pandas as pd

importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=True)

importances.tail(15).plot(kind='barh', figsize=(10, 6), color='steelblue')
plt.title('Feature Importance (Gini/MSE)')
plt.savefig('outputs/figures/feature_importance.png', dpi=150, bbox_inches='tight')
```

**Sesgo:** favorece features con muchos valores únicos (continuas vs categóricas con pocas categorías).

### Permutation Importance (más confiable)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=True)
```

---

## Pre-requisitos para esta Familia

1. **NO requiere estandarización** — los splits son invariantes a escala
2. **NO requiere imputación** — sklearn 1.4+ permite NaN en árboles, pero por consistencia mejor imputar
3. **One-Hot para categóricas** — los árboles de sklearn no aceptan strings directamente
4. **Para clasificación con desbalance** — preferir `class_weight='balanced'` antes que SMOTE para árboles

---

## Gotchas

- **`max_depth=None` (default)** — el árbol crece hasta hojas puras → overfitting. Siempre limitar.
- **Pocos `n_estimators`** — varianza alta. Subir a 200+ por defecto.
- **Olvidar `n_jobs=-1`** — entrenamiento mucho más lento de lo necesario. Pero usar máximo 70% de los cores (`n_jobs=int(os.cpu_count() * 0.7)`) para no saturar el sistema.
- **Confiar ciegamente en `feature_importances_`** — usar permutation importance para decisiones reales.

## Conexión con la Teoría

Para fundamentos (varianza de promedios correlacionados, OOB, comparación con boosting), consulta `rag-books-mcp` siguiendo `theory-rag-guide.md`. Queries útiles: `get_section(book="esl", chapter=15)` (Random Forests), `search_theory(query="out of bag error estimate")`, `cite_foundation(topic="bagging vs boosting bias variance")`.

## Siguiente paso

Si Random Forest no es suficiente, escalar a Boosting (XGBoost). Ver `models-ensemble.md`.

## Consultar Documentación

```
"sklearn RandomForestRegressor max_features oob_score" → docs sklearn (vía remote_web_search)
"sklearn ExtraTreesRegressor bootstrap default"        → docs sklearn (vía remote_web_search)
"sklearn permutation_importance n_repeats scoring"     → docs sklearn (vía remote_web_search)
```
