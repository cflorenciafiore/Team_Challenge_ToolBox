import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


def describe_df(dataframe: pd.DataFrame) -> pd.DataFrame: # {{{
	"""
	Calcula y describe la calidad de datos de cada columna de un dataframe.

	Entre los datos de calidad que se calculan encontramos el porcentaje de 
	cardinalidad, el porcentaje de valores faltantes, los valores únicos y
	el tipo de dato de las columnas.

	Args:
		dataframe: pd.DataFrame
			un dataframe

	Returns:
		pd.DataFrame
	"""
	if dataframe is None:
		raise ValueError("Dataframe sin especificar.")

	if dataframe.empty:
		raise ValueError("Dataframe vacío.")

	records = dataframe.shape[0]
	# print(f"Registros: {records}\nColumns: {attributes}\n")

	columns = dataframe.columns.values
	index = pd.Index(data=columns, dtype=str)


	data_type = pd.Series(data=dataframe.dtypes,
						  index=index,
						  name="DATA_TYPE")

	missing = (dataframe.isna().sum()/records) * 100
	missing = pd.Series(data=round(missing, 1).values,
						index=index,
						name="MISSINGS (%)")

	unique = pd.Series(data=dataframe.nunique().values,
					   index=index,
					   name="UNIQUE_VALUES")

	cardinality = pd.Series(data=round(unique/records * 100, 2),
							index=index,
							name="CARDIN (%)")

	data = {
		data_type.name: data_type,
		missing.name: missing,
		unique.name: unique,
		cardinality.name: cardinality
	}

	df = pd.DataFrame(data, index=index)
	return df.T
# }}}


def tipifica_variables(
    dataframe: pd.DataFrame,
    umbral_categoria: int,
    umbral_continua: float
) -> pd.DataFrame:
    """
    Calcula la cardinalidad de las columnas y categoriza las columnas del
    dataframe siguiendo estas reglas:

    - Si la cardinalidad (nº de valores únicos) de la columna es igual a 2,
      se categoriza como binaria.
    - Si la cardinalidad es menor que umbral_categoria, se asigna como categórica.
    - Si la cardinalidad es mayor o igual que umbral_categoria:
        - Se asigna numérica continua si el % de cardinalidad (valores únicos / filas * 100)
          es mayor o igual que umbral_continua.
        - En caso contrario, se asigna numérica discreta.

    Args:
        dataframe (pd.DataFrame): DataFrame de entrada.
        umbral_categoria (int): Umbral de cardinalidad para considerar una variable como categórica.
        umbral_continua (float): Umbral (en %) para considerar una variable numérica como continua.

    Returns:
        pd.DataFrame: DataFrame con columnas ["nombre_variable", "tipo_sugerido"].
    """

    # -----------------------------
    # Validaciones de entrada
    # -----------------------------
    if dataframe is None:
        raise ValueError("Dataframe sin especificar.")

    if dataframe.empty:
        raise ValueError("Dataframe vacío.")

    if umbral_categoria is None:
        raise ValueError("Umbral categoría sin especificar.")

    if umbral_continua is None:
        raise ValueError("Umbral continua sin especificar.")

    # Número de filas (para calcular el % de cardinalidad)
    n_rows = len(dataframe)

    # Cardinalidad por columna ignoramos los NaN
    unique_values = dataframe.nunique(dropna=True)

    # Porcentaje de cardinalidad por columna
    cardin_pct = (unique_values / n_rows) * 100

    tipos = []

    # Recorremos columnas del dataframe
    for col in dataframe.columns:
        card = int(unique_values[col])   # cardinalidad total
        pct = float(cardin_pct[col])     # cardinalidad relativa en %

        # binaria si tiene exactamente 2 valores únicos
        if card == 2:
            tipo = "Binaria"

        # categórica si la cardinalidad es menor al umbral de categoría
        elif card < umbral_categoria:
            tipo = "Categorica"

        # si no es categórica, se decide entre continua/discreta por % cardinalidad
        else:
            if pct >= umbral_continua:
                tipo = "Numerica Continua"
            else:
                tipo = "Numerica Discreta"

        tipos.append((col, tipo))

    # Devolvemos un dataframe con el resultado
    return pd.DataFrame(tipos, columns=["nombre_variable", "tipo_sugerido"])


def get_features_num_regression(dataframe, target_col, umbral_corr=0.5, pvalue=None):
	
    # Compruebo que df es un df
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: el argumento 'df' no es un DataFrame.")
        return None
    
    # Compruebo que el df no esté vacío
    if dataframe.empty:
        print("Error: el DataFrame está vacío.")
        return None
    
    # Compruebo que target_col existe en el df
    if target_col not in dataframe.columns:
        print("Error: la columna '{target_col}' no existe en el DataFrame.")
        return None

    # Compruebo que target_col es numérica
    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print("Error: la columna '{target_col}' no es numérica, no puede ser target de la regresión.")
        return None
    
    if not isinstance(umbral_corr, (int, float)) or not (0 <= float(umbral_corr) <= 1):
        print("Error: 'umbral_corr' debe ser un float entre 0 y 1 (incluidos).")
        return None
    umbral_corr = float(umbral_corr)

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < float(pvalue) <= 1):
            print("Error: 'pvalue' debe ser None o un float en el rango (0, 1].")
            return None
        pvalue = float(pvalue)
    # Selecciono las columnas numéricas
    numeric_cols = dataframe.select_dtypes(include="number").columns

    # Exluyo al target de las columnas numéricas
    numeric_features = numeric_cols.drop(target_col)

    # Compruebo que target_col no sea la única columna numérica
    if len(numeric_features) == 0:
        print("No hay columnas numéricas para analizar.")
        return None
    
    # Calculo correlaciones para filtrar
    selected_features = []

    for col in numeric_features:
        tmp = dataframe[[col, target_col]].dropna()
        corr = tmp[col].corr(tmp[target_col])

        if abs(corr) >= umbral_corr:
            selected_features.append(col)

    # Si me pasan pvalue, aplico test de hipótesis para filtrar aún más
    if pvalue is not None:
        final_features = []
        for col in selected_features:
            corr, pvalue_test = pearsonr(dataframe[col], dataframe[target_col])
            if pvalue_test <= pvalue:
                final_features.append(col)
        return final_features
    else:
        return selected_features

def plot_features_num_regression(dataframe, target_col="", columns=[], umbral_corr=0, pvalue=None):

    # Compruebo que df es un df
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: el argumento 'df' no es un DataFrame.")
        return None
    
    # Compruebo que el df no esté vacío
    if dataframe.empty:
        print("Error: el DataFrame está vacío.")
        return None
    
    # Compruebo que target_col existe en el df
    if target_col not in dataframe.columns:
        print("Error: la columna '{target_col}' no existe en el DataFrame.")
        return None
    # Compruebo que target_col existe en el df
    if target_col not in dataframe.columns:
        print("Error: la columna '{target_col}' no existe en el DataFrame.")
        return None

    # Compruebo que target_col es numérica
    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print("Error: la columna '{target_col}' no es numérica, no puede ser target de la regresión.")
        return None
    
    if not isinstance(umbral_corr, (int, float)) or not (0 <= float(umbral_corr) <= 1):
        print("Error: 'umbral_corr' debe ser un float entre 0 y 1 (incluidos).")
        return None
    umbral_corr = float(umbral_corr)

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < float(pvalue) <= 1):
            print("Error: 'pvalue' debe ser None o un float en el rango (0, 1].")
            return None
        pvalue = float(pvalue)

    # Selecciono las columnas numéricas
    numeric_cols = dataframe.select_dtypes(include="number")

    # Exluyo al target de las columnas numéricas
    numeric_features = numeric_cols.columns.drop(target_col)

    # Compruebo que target_col no sea la única columna numérica
    if len(numeric_features) == 0:
        print("No hay columnas numéricas para analizar.")
        return None
    
    # Si no me pasan ninguna columna, uso todas las columnas numéricas (menos el target)
    if not columns:
        columns = list(numeric_features)

    # Calculo correlaciones para filtrar
    selected_features = []

    for col in columns:
        corr = dataframe[col].corr(dataframe[target_col])
        if abs(corr) >= umbral_corr:
            selected_features.append(col)

    # Si me pasan pvalue, aplico test de hipótesis para filtrar aún más
    if pvalue is not None:
        final_features = []
        for col in selected_features:
            corr, pvalue_test = pearsonr(dataframe[col], dataframe[target_col])
            if pvalue_test <= pvalue:
                final_features.append(col)
    else:
        final_features = selected_features
    
    # Incluyo al target en las columnas finales
    final_features = [target_col] + final_features

    #Dibujo los pairplots
    max_cols = 5

    for i in range (0, len(final_features), max_cols):
        features_to_plot = final_features[i:i + max_cols]
        sns.pairplot(dataframe[features_to_plot])
        plt.show()

    return selected_features

def get_features_cat_regression(dataframe,
								target_col,
								pvalue=0.05):
	"""
	Esta función recibe como **argumentos un dataframe**, el nombre de **una de las
	columnas del mismo (argumento 'target_col')**, que **debería ser el target de un
	hipotético modelo de regresión**, es decir debe ser **una variable numérica
	continua o discreta pero con alta cardinalidad y una variable float "pvalue"
	cuyo valor por defecto será 0.05*.

	La función debe devolver una lista con las columnas categóricas del dataframe
	cuyo test de relación con la columna designada por 'target_col' supere en
	confianza estadística el test de relación que sea necesario hacer (es decir la
	función debe poder escoger cuál de los dos test que hemos aprendido tiene que
	hacer).

	La función debe hacer todas las comprobaciones necesarias para no dar error como
	consecuecia de los valores de entrada. Es decir hará un check de los valores
	asignados a los argumentos de entrada y si estos no son adecuados debe retornar
	None y printar por pantalla la razón de este comportamiento. Ojo entre las
	comprobaciones debe estar que "target_col" hace referencia a una variable
	numérica continua del dataframe.


	Args:
		dataframe: pd.DataFrame
			un dataframe

		target_col: str
			columna de dataframe. Debe ser una variable numérica continua o
			discreta pero con alta cardinalidad

		pvalue: float, default 0.05
			valor de significación en test de hipótesis


	Return:
		None
	"""
	pass



def plot_features_cat_regression(dataframe,
								 target_col="",
								 columns=[],
								 pvalue=0.05,
								 with_individual_plot=False):
	"""
	Esta función recibe un dataframe, una argumento "target_col" con valor por
	defecto "", una lista de strings ("columns") cuyo valor por defecto es la
	lista vacía, un argumento ("pvalue") con valor 0.05 por defecto y un
	argumento "with_individual_plot" a False.

	Si la lista no está vacía, la función pintará los histogramas agrupados de
	la variable "target_col" para cada uno de los valores de las variables
	categóricas incluidas en columns que cumplan que su test de relación con
	"target_col" es significatio para el nivel 1-pvalue de significación
	estadística. La función devolverá los valores de "columns" que cumplan con
	las condiciones anteriores.

	Si la lista está vacía, entonces la función igualará "columns" a las
	variables numéricas del dataframe y se comportará como se describe en el
	párrafo anterior.

	De igual manera que en la función descrita anteriormente deberá hacer un
	check de los valores de entrada y comportarse como se describe en el último
	párrafo de la función `get_features_cat_regression`.


	Args:
		dataframe: pd.DataFrame
			un dataframe

		target_col: str, default ""
			columna de dataframe. Debe ser una variable numérica continua o
			discreta pero con alta cardinalidad

		columns: list, default []
			listado de columnas categóricas del dataframe

		pvalue: float, default 0.05
			valor de significación en test de hipótesis entre target_col y
			columns

		with_individual_plot: bool, default False
			valor lógico para agrupar los plots de los histogramas o no

	Return:
		None
	"""