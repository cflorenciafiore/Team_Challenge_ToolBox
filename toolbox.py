import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


def describe_df(dataframe: pd.DataFrame) -> pd.DataFrame:
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

	records = dataframe.shape[0]
	attributes = dataframe.shape[1]
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


def tipifica_variables(dataframe, umbral_categorica, umbral_continua):
	"""
	Esta función debe recibir como argumento un dataframe, un entero
	(`umbral_categoria`) y un float (`umbral_continua`). La función debe
	devolver un dataframe con dos columnas "nombre_variable", "tipo_sugerido"
	que tendrá tantas filas como columnas el dataframe. En cada fila irá el
	nombre de una de las columnas y una sugerencia del tipo de variable.


	Args:
		dataframe: pd.DataFrame
			un dataframe

		umbral_categorica: int
			unmbral que permite identificar si una variables

			Tener en cuenta que:
			- Si la cardinalidad es 2, asignara "Binaria".
			- Si la cardinalidad es menor que `umbral_categoria` asignara
			  "Categórica"
			- Si la cardinalidad es mayor o igual que `umbral_categoria`,
			  entonces entra en juego el tercer argumento `umbral_continua`

		umbral_continua: float
			umbral que permite identificar si una variable es númerica continua
			o discreta.

			Tener en cuenta que:
			- Si además el porcentaje de cardinalidad es superior o igual a
			  `umbral_continua`, asigna "Numerica Continua"
			- En caso contrario, asigna "Numerica Discreta"


	Return:
		None
	"""
	pass


def get_features_num_regression(dataframe,
								target_col,
								umbral_corr=0.5,
								pvalue=None):
	"""
	Esta función recibe **como argumentos un dataframe**, el nombre de una de
	las columnas del mismo (argumento **'target_col'**), que debería ser **el
	target de un hipotético modelo de regresión**, es decir **debe ser una
	variable numérica continua o discreta pero con alta cardinalidad**, además
	de un **argumento 'umbral_corr'**, de tipo float que debe estar entre 0 y 1
	y **una variable float "pvalue"** cuyo valor debe ser por defecto "None".

	La función debe **devolver una lista con las columnas numéricas del
	dataframe cuya correlación con la columna designada por "target_col" sea
	superior en valor absoluto al valor dado por "umbral_corr"**. Además si la
	variable "pvalue" es distinta de None, **sólo devolvera las columnas
	numéricas cuya correlación supere el valor indicado y además supere el test
	de hipótesis con significación mayor o igual a 1-pvalue**.

	La función debe hacer todas las comprobaciones necesarias para no dar error
	como consecuecia de los valores de entrada. Es decir hará un check de los
	valores asignados a los argumentos de entrada y si estos no son adecuados
	debe retornar None y printar por pantalla la razón de este comportamiento.
	Ojo entre las comprobaciones debe estar que "target_col" hace referencia a
	una variable numérica continua del dataframe.


	Args:
		dataframe: pd.DataFrame
			un dataframe

		target_col: str
			columna de dataframe. Debe ser una variable numérica continua o
			discreta pero con alta cardinalidad

		umbral_corr: float, default 0.5
			umbral de correlación que se debe tener con target_col

		pvalue: float, default None
			valor de significación en test de hipótesis si cumple con
			umbral_corr


	Return:
		None
	"""
	pass


def plot_features_num_regression(dataframe,
								 target_col="",
								 columns=[],
								 umbral_corr=0,
								 pvalue=None):
	"""
	Esta función recibe un dataframe, una argumento "target_col" con valor por
	defecto "", una lista de strings ("columns") cuyo valor por defecto es la
	lista vacía, un valor de correlación ("umbral_corr", con valor 0 por
	defecto) y un argumento ("pvalue") con valor "None" por defecto.

	Si la lista no está vacía, la función pintará una pairplot del dataframe
	considerando la columna designada por "target_col" y aquellas incluidas en
	"column" que cumplan que su correlación con "target_col" es superior en
	valor absoluto a "umbral_corr", y que, en el caso de ser pvalue diferente de
	"None", además cumplan el test de correlación para el nivel 1-pvalue de
	significación estadística. La función devolverá los valores de "columns" que
	cumplan con las condiciones anteriores. 

	EXTRA: Se valorará adicionalmente el hecho de que si la lista de columnas a
	pintar es grande se pinten varios pairplot con un máximo de cinco columnas
	en cada pairplot (siendo siempre una de ellas la indicada por "target_col")

	Si la lista está vacía, entonces la función igualará "columns" a las
	variables numéricas del dataframe y se comportará como se describe en el
	párrafo anterior.

	De igual manera que en la función descrita anteriormente deberá hacer un
	check de los valores de entrada y comportarse como se describe en el último
	párrafo de la función `get_features_num_regresion`


	Args:
		dataframe: pd.DataFrame
			un dataframe

		target_col: str, default ""
			columna de dataframe. Debe ser una variable numérica continua o
			discreta pero con alta cardinalidad

		columns: list, default []
			listado de columnas de dataframe

		umbral_corr: float, default 0
			umbral de correlación que se debe tener entre columns y target_col

		pvalue: float, default None
			valor de significación en test de hipótesis si cumple con
			umbral_corr entre columns y target_col


	Return:
		None
	"""
	pass


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
