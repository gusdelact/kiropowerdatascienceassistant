---
inclusion: always
---

# Matplotlib en entornos headless (EC2 terminal, contenedores, CI)

Esta convención aplica a todo script de EDA, visualización o reporte que genere figuras con matplotlib / seaborn en este proyecto. Garantiza que el mismo código corra igual en macOS local y en una instancia EC2 sin display, sin abrir ventanas GUI ni colgarse por falta de `$DISPLAY`.

## Regla 1: backend `Agg` siempre

Antes de cualquier import de `matplotlib.pyplot` o `seaborn`, fija el backend no interactivo:

```python
import matplotlib
matplotlib.use("Agg")  # backend sin GUI; debe ir antes de importar pyplot
import matplotlib.pyplot as plt
import seaborn as sns
```

El orden importa. Si `pyplot` se importa antes del `matplotlib.use(...)`, el backend ya quedó decidido y el cambio no surte efecto.

En notebooks (`.ipynb`) sí está permitido usar el backend inline (`%matplotlib inline`); la regla `Agg` aplica a scripts `.py` que pueden ejecutarse en EC2/headless.

## Regla 2: guardar, no mostrar

Nunca uses `plt.show()` en scripts. Guarda la figura y libera memoria:

```python
fig_path = "outputs/eda/distribucion_clases.png"
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
```

`plt.close()` es obligatorio dentro de loops para evitar fugas de memoria.

## Regla 3: carpeta de salida estándar

Las figuras del EDA se guardan en `outputs/eda/` relativo a la raíz del proyecto. Para reportes de modelos, usa `outputs/modelos/<nombre_modelo>/`. Crea la carpeta antes de escribir:

```python
from pathlib import Path
Path("outputs/eda").mkdir(parents=True, exist_ok=True)
```

## Regla 4: red de seguridad por entorno

En EC2 o contenedores, exporta también la variable de entorno como respaldo, por si algún módulo importa pyplot antes de que se ejecute `matplotlib.use(...)`:

```bash
export MPLBACKEND=Agg
```

Recomendado dejarlo en el `~/.bashrc` o en el script de arranque del job.

## Plantilla mínima para un script de EDA

```python
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

OUT = Path("outputs/eda")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/midataset.csv")

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["columna"], ax=ax)
ax.set_title("Distribución de columna")
plt.savefig(OUT / "distribucion_columna.png", dpi=120, bbox_inches="tight")
plt.close()
```

## Qué NO hacer

- No usar `plt.show()` en scripts `.py`.
- No depender del backend por defecto del sistema (en macOS abre ventana GUI; en EC2 falla).
- No mezclar `plt.savefig` con `plt.show()` en el mismo flujo: el `show` consume la figura y el archivo puede quedar vacío según el backend.
- No dejar figuras abiertas dentro de loops sin `plt.close()`.

## Cómo revisar resultados desde EC2

Como no hay display, las figuras se inspeccionan descargándolas o sirviéndolas:

```bash
# Descarga local
scp -i mi-llave.pem ec2-user@<host>:~/proyecto/outputs/eda/*.png ./eda_local/

# O sincronización completa
rsync -avz -e "ssh -i mi-llave.pem" ec2-user@<host>:~/proyecto/outputs/ ./outputs/
```
