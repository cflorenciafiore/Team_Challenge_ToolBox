## Team Challenge: Toolbox

Este repositorio incluye un **toolbox en Python** con utilidades para **análisis exploratorio de datos (EDA) y selección de variables** orientadas a **modelos de regresión**, junto con un script `run_test.py` para validar el comportamiento de las funciones.

### Qué incluye

#### Funciones principales (`toolbox_ML.py`)
- **`describe_df(df)`**
    Genera un resumen del dataset con tipos de datos, valores faltantes y cardinalidad.
- **`tipifica_variables(df, umbral_categoria, umbral_continua)`**
    Sugiere el tipo de cada variable según sus características:
        - Binaria: dos valores únicos
        - Categórica: menos de `umbral_categoria` valores únicos
        - Numérica discreta: menos de `umbral_continua` valores únicos
        - Numérica continua: resto de las variables
 ⚠️ **Observación:** revisar manualmente la tipificación automática, porque, por ejemplo:
        - Columnas que son solo IDs (como `customerID`) pueden marcarse como "Numérica Continua", pero realmente no deberían usarse como feature.
        - Variables binarias pueden marcarse como "Categórica" si no se ajustan los umbrales.  
        - Variables discretas pequeñas pueden confundirse con categóricas.
- **`get_features_num_regression(df, target_col, umbral_corr, pvalue)`**
    Selecciona variables numéricas relevantes para regresión según correlación y/o p-value.
- **`plot_features_num_regression(df, target_col, columns, umbral_corr, pvalue)`**
    Genera gráficos de dispersión y distribución de variables numéricas seleccionadas.
- **`get_features_cat_regression(df, target_col, pvalue)`**
    Selecciona variables categóricas relevantes para regresión según ANOVA.
- **`plot_features_cat_regression(df, target_col, columns, pvalue, with_individual_plot)`**
    Genera gráficos comparativos de variables categóricas seleccionadas.

#### Script de pruebas (`run_test.py`)
    Ejecuta tests/validaciones automáticas de las funciones del toolbox.

#### Requisitos
    - Python 3.9+ (recomendado)
    - Dependencias:
        - `pandas`
        - `numpy`
        - `scipy` (incluye `scipy.stats`: `pearsonr`, `ttest_ind`, `f_oneway`)
        - `matplotlib`
        - `seaborn`

#### Ejemplo práctico
    Se recomienda revisar el notebook: `example_telco.ipynb`
    Contiene un ejemplo completo usando el dataset de Telco (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).  
    Incluye análisis exploratorio, tipificación de variables, selección de features numéricas y categóricas, y visualización.  
    También sirve como base para la presentación de 10 minutos del proyecto.

#### Requisitos

- Python 3.9+ (recomendado)
- Dependencias:
  - `pandas`
  - `numpy`
  - `scipy` (incluye `scipy.stats`: `pearsonr`, `ttest_ind`, `f_oneway`)
  - `matplotlib`
  - `seaborn`

#### Estructura del repositorio
```
.
├── toolbox_ML.py
├── run_test.py
├── example_telco.ipynb
├── README.md
└── data
    └──WA_Fn-UseC_-Telco-Customer-Churn.csv
```

#### Ejecución básica de las funciones (`toolbox_ML.py`)

```python
import pandas as pd
from toolbox_ML import (
    describe_df,
    tipifica_variables,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression
)

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
target = "MonthlyCharges"

# 1) Calidad de datos
print(describe_df(df))

# 2) Tipificación automática de variables
print(tipifica_variables(df, umbral_categoria=10, umbral_continua=15.0))

# 3) Selección de features numéricas
num_feats = get_features_num_regression(df, target_col=target, umbral_corr=0.5, pvalue=None)
print(num_feats)

# 4) Visualización de features numéricas
plot_features_num_regression(df, target_col=target, columns=num_feats, umbral_corr=0.5, pvalue=None)

# 5) Selección de features categóricas
cat_feats = get_features_cat_regression(df, target_col=target, pvalue=0.05)
print(cat_feats)

# 6) Visualización de features categóricas
plot_features_cat_regression(df, target_col=target, columns=cat_feats, pvalue=0.05, with_individual_plot=False)
```

#### Ejecución básica de los tests (`run_test.py`)
    Ejecuta el script de pruebas desde la raíz del repositorio:

```bash
python run_test.py
```