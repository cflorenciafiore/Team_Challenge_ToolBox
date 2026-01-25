# Team Challenge: Toolbox de EDA

Este repositorio incluye un **toolbox en Python** con utilidades para análisis exploratorio y selección de variables orientadas a **modelos de regresión**, junto con un script `run_test` para validar el comportamiento de las funciones.

## Qué incluye

### Funciones principales (toolbox)
- **`describe_df(df)`**
- **`tipifica_variables(df, umbral_categoria, umbral_continua)`**
- **`get_features_num_regression(df, target_col, umbral_corr=0.5, pvalue=None)`**
- **`plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None)`**
- **`get_features_cat_regression(df, target_col="", pvalue=0.05)`**
- **`plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False)`**

### Script de pruebas
- **`run_test.py`**: ejecuta tests/validaciones del toolbox.


## Requisitos

- Python 3.9+ (recomendado)
- Dependencias:
  - `pandas`
  - `numpy`
  - `scipy` (incluye `scipy.stats`: `pearsonr`, `ttest_ind`, `f_oneway`)
  - `matplotlib`
  - `seaborn`


## Estructura del repositorio

.
├── toolbox.py
├── run_test.py
├── README.md
└── data
    └──WA_Fn-UseC_-Telco-Customer-Churn.csv


## Uso

### 1) Importar y ejecutar funciones

```python
import pandas as pd

from toolbox import (
    describe_df,
    tipifica_variables,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression
)

df = pd.read_csv("data.csv")

# Calidad de datos
print(describe_df(df))

# Tipificación
print(tipifica_variables(df, umbral_categoria=20, umbral_continua=50.0))

# Selección numéricas por correlación
num_feats = get_features_num_regression(df, target_col="target", umbral_corr=0.5, pvalue=0.05)
print(num_feats)

# Pairplots (máx 5 columnas por bloque)
plot_features_num_regression(df, target_col="target", columns=num_feats, umbral_corr=0.5, pvalue=0.05)

# Selección categóricas por ANOVA
cat_feats = get_features_cat_regression(df, target_col="target", pvalue=0.05)
print(cat_feats)

# Visualización categóricas
plot_features_cat_regression(df, target_col="target", columns=cat_feats, pvalue=0.05, with_individual_plot=False)
```

## Ejecutar tests (run_test)

Ejecuta el script de pruebas desde la raíz del repo:

```bash
python run_test.py
```

