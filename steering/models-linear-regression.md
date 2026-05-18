# Modelos: Regresión Lineal y Regularización

> Cubre: OLS con `statsmodels`, motivación de Ridge (multicolinealidad e índice de condición), Lasso, ElasticNet. Ejemplo: dataset Boston Housing (506 obs, 13 predictores).

Esta guía es **específica para modelos lineales y regularizados**. Para el flujo común (split, CV, métricas), ver `workflow-model-training.md` y `workflow-validation.md`.

---

## Cuándo Usar Esta Familia

| Situación | Recomendación |
|---|---|
| Relaciones aproximadamente lineales | OLS / Ridge / Lasso |
| Alta interpretabilidad requerida | OLS o Lasso (este último selecciona variables) |
| Multicolinealidad alta | Ridge |
| Pocas variables relevantes (señal sparse) | Lasso |
| Grupos de variables correlacionadas | Elastic Net |
| `p > N` (más features que observaciones) | Ridge o Elastic Net |
| Relaciones claramente no lineales | Cambiar a árboles (`models-trees-rf.md`) |

---

## 1. OLS (Ordinary Least Squares)

### Formulación

$$y = X\beta + \epsilon \quad\Rightarrow\quad \hat\beta = (X^TX)^{-1}X^Ty$$

### Implementación con `statsmodels` (para diagnóstico estadístico)

```python
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('boston_house_prices.csv')
predictors = df.loc[:, 'CRIM':'LSTAT']
response = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    predictors, response, test_size=0.25, random_state=314
)

# statsmodels da p-values, intervalos de confianza, R², etc.
X_train_const = add_constant(X_train)
X_test_const = add_constant(X_test)

ols = OLS(y_train, X_train_const).fit()
print(ols.summary())

y_pred = ols.predict(X_test_const)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE OLS: {mse:.4f}")
```

### Implementación con `sklearn` (para producción)

```python
from sklearn.linear_model import LinearRegression

ols_sk = LinearRegression()
ols_sk.fit(X_train, y_train)
y_pred = ols_sk.predict(X_test)
print(f"R²: {ols_sk.score(X_test, y_test):.4f}")
print(f"Coeficientes: {dict(zip(X_train.columns, ols_sk.coef_))}")
```

### Diagnóstico de Colinealidad

**Índice de condición de la matriz $X^TX$:**

$$\kappa(X) = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}$$

```python
from numpy import linalg as la

XtX = X_train.T @ X_train
eigenvalues = la.eigvalsh(XtX)
condicion = np.sqrt(eigenvalues.max() / eigenvalues.min())
print(f"Índice de condición: {condicion:.2f}")
```

**Interpretación:**
- κ < 30 → multicolinealidad baja
- 30 ≤ κ < 100 → moderada
- κ ≥ 100 → severa → **usar Ridge o Elastic Net**

---

## 2. Ridge Regression (L2)

### Formulación

$$\hat\beta_{\text{ridge}} = \arg\min_\beta \left\{ \sum_i (y_i - x_i^T\beta)^2 + \lambda \sum_j \beta_j^2 \right\}$$

**Solución cerrada:** $\hat\beta_{\text{ridge}} = (X^TX + \lambda I)^{-1} X^T y$

### Propiedades

- **Shrinkage continuo** — todos los coeficientes se reducen pero **nunca son exactamente cero**
- **No hace selección de variables** — mantiene todas las features
- **Maneja multicolinealidad** — estabiliza coeficientes
- **Siempre tiene solución** — incluso con `p > N`

### Implementación

```python
from sklearn.linear_model import Ridge, RidgeCV

# Con λ fijo
ridge = Ridge(alpha=0.1)   # alpha = λ
ridge.fit(X_train, y_train)
print(f"R² Ridge: {ridge.score(X_test, y_test):.4f}")

# Con CV automático para elegir λ
alphas = np.logspace(-4, 4, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Mejor α: {ridge_cv.alpha_:.6f}")
print(f"R² Ridge CV: {ridge_cv.score(X_test, y_test):.4f}")
```

> **Importante:** Estandarizar las features ANTES de aplicar Ridge. La penalización es sensible a la escala. Esto se hace en `workflow-feature-engineering.md`.

---

## 3. Lasso (L1)

### Formulación

$$\hat\beta_{\text{lasso}} = \arg\min_\beta \left\{ \frac{1}{2N}\sum_i (y_i - x_i^T\beta)^2 + \lambda \sum_j |\beta_j| \right\}$$

### Propiedades

- **Selección de variables** — algunos coeficientes son **exactamente cero**
- **Modelo sparse** → más interpretable
- **No tiene solución cerrada** — usa coordinate descent
- **Inestable con features correlacionadas** — elige una arbitrariamente

### Implementación

```python
from sklearn.linear_model import Lasso, LassoCV

# Con CV
alphas = np.logspace(-4, 1, 100)
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

print(f"Mejor α: {lasso_cv.alpha_:.6f}")
print(f"R² Lasso CV: {lasso_cv.score(X_test, y_test):.4f}")

# Cuántas variables sobreviven
n_seleccionadas = (lasso_cv.coef_ != 0).sum()
print(f"Variables seleccionadas: {n_seleccionadas} de {X_train.shape[1]}")

# Cuáles
seleccionadas = X_train.columns[lasso_cv.coef_ != 0]
print(f"Variables: {list(seleccionadas)}")
```

### Regularization Path (visualización)

```python
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

alphas, coefs, _ = lasso_path(X_train, y_train, alphas=alphas)

plt.figure(figsize=(10, 6))
for i in range(coefs.shape[0]):
    plt.plot(np.log10(alphas), coefs[i], label=X_train.columns[i])
plt.xlabel('log10(α)')
plt.ylabel('Coeficientes')
plt.title('Lasso Path — variables que sobreviven a mayor α son las más importantes')
plt.axvline(np.log10(lasso_cv.alpha_), color='k', linestyle='--', label='α óptimo')
plt.legend(loc='best', fontsize=8)
plt.savefig('outputs/figures/lasso_path.png', dpi=150, bbox_inches='tight')
```

---

## 4. Elastic Net (L1 + L2)

### Formulación

$$\hat\beta_{\text{enet}} = \arg\min_\beta \left\{ \frac{1}{2N}\sum_i (y_i - x_i^T\beta)^2 + \lambda \left[\alpha \sum_j |\beta_j| + \frac{1-\alpha}{2}\sum_j \beta_j^2\right] \right\}$$

donde α controla la mezcla entre L1 (Lasso) y L2 (Ridge).

### Implementación

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# CV busca α (l1_ratio) y λ (alpha)
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=np.logspace(-4, 1, 100),
    cv=5,
    random_state=42
)
enet_cv.fit(X_train, y_train)

print(f"Mejor l1_ratio (α): {enet_cv.l1_ratio_}")
print(f"Mejor alpha (λ): {enet_cv.alpha_:.6f}")
print(f"R² ElasticNet: {enet_cv.score(X_test, y_test):.4f}")
```

---

## 5. Benchmarking de la Familia

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

modelos = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=0.1),
    'Lasso': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{nombre:12s} - MSE: {mse:.4f}")
```

**Resultado esperado en Boston Housing:**
- OLS, Ridge, Lasso, ElasticNet → MSE entre 22-25 (modelos lineales similares)
- Random Forest sobre el mismo dataset → MSE ~10 (significa que las relaciones NO son puramente lineales)

> **Conclusión práctica:** Si los modelos lineales tienen MSE similar y mucho peor que árboles, las relaciones son no lineales y debes cambiar de familia.

---

## Tabla de Decisión

| Aspecto | Ridge | Lasso | Elastic Net |
|---|---|---|---|
| Selección de variables | No | Sí | Sí |
| Coeficientes en cero | Nunca | Sí | Sí |
| Multicolinealidad | Maneja muy bien | Inestable | Bien |
| Solución cerrada | Sí | No | No |
| Funciona con `p > N` | Sí | Sí (≤ N vars) | Sí |
| Cuándo elegir | Señal distribuida | Señal sparse | Grupos correlacionados |

---

## Hiperparámetros Recomendados

| Modelo | Hiperparámetro principal | Heurística |
|---|---|---|
| Ridge | `alpha` | `np.logspace(-4, 4, 100)` con `RidgeCV` |
| Lasso | `alpha` | `np.logspace(-4, 1, 100)` con `LassoCV` |
| ElasticNet | `alpha`, `l1_ratio` | `[0.1, 0.5, 0.7, 0.9, 0.95, 1.0]` para `l1_ratio` |

**Regla:** SIEMPRE usar la versión `*CV` (RidgeCV, LassoCV, ElasticNetCV) en vez de fijar α a mano. Es más rápido que `GridSearchCV` y igual de robusto.

---

## Pre-requisitos para esta Familia

1. **Estandarización obligatoria** — Ridge/Lasso/ElasticNet son sensibles a escala. Hecha en `workflow-feature-engineering.md`.
2. **Multicolinealidad analizada** — si κ ≥ 100, prefiere Ridge sobre OLS.
3. **One-Hot Encoding** para categóricas.
4. **Imputación previa** — sklearn no acepta NaN.

---

## Gotchas

- **OLS con multicolinealidad alta** — coeficientes inestables, p-values poco confiables → usar Ridge.
- **Lasso con features muy correlacionadas** — selección arbitraria → usar Elastic Net.
- **No estandarizar antes** — los coeficientes con regularización quedan dominados por las variables de mayor escala.
- **Confundir el `alpha` de sklearn con el `lambda` de los libros** — son lo mismo. ESL usa λ, sklearn usa α.

## Conexión con la Teoría

Para fundamentos completos (geometría L1 vs L2, demostraciones, regularization path), consulta `rag-books-mcp` siguiendo `theory-rag-guide.md`. Queries útiles: `get_section(book="esl", chapter=3, section="3.4")`, `cite_foundation(topic="lasso L1 sparsity geometry")`, `search_theory(query="elastic net grouping correlated features")`.

## Consultar Documentación

```
"sklearn LassoCV n_alphas selection cyclic random"   → docs sklearn (vía remote_web_search)
"statsmodels OLS summary R-squared condition number" → docs statsmodels (vía remote_web_search)
```
