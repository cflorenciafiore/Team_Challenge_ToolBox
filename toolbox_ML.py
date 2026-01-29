import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr


def describe_df(dataframe: pd.DataFrame) -> pd.DataFrame:
        
	"""
	Calcula y describe la calidad de datos de cada columna de un dataframe.

	Entre los datos de calidad que se calculan, encontramos el porcentaje de
	cardinalidad, el porcentaje de valores faltantes, los valores únicos y
	el tipo de dato de las columnas.

	Args:
		dataframe: pd.DataFrame
			un dataframe

	Returns:
		pd.DataFrame
	"""

	# -----------------------------
	# Validaciones de entrada
	# -----------------------------
	if dataframe is None:
		raise ValueError("Dataframe sin especificar.")

	if dataframe.empty:
		raise ValueError("Dataframe vacío.")

	records = dataframe.shape[0]
	# print(f"Registros: {records}\nColumns: {attributes}\n")

	columns = dataframe.columns.values
	index = pd.Index(data=columns, dtype=str)

	data_type = pd.Series(
		data=dataframe.dtypes,
		index=index,
		name="DATA_TYPE"
	)

	missing = (dataframe.isna().sum() / records) * 100
	missing = pd.Series(
		data=round(missing, 1).values,
		index=index,
		name="MISSINGS (%)"
	)

	unique = pd.Series(
		data=dataframe.nunique().values,
		index=index,
		name="UNIQUE_VALUES"
	)

	cardinality = pd.Series(
		data=round(unique / records * 100, 2),
		index=index,
		name="CARDIN (%)"
	)

	data = {
		data_type.name: data_type,
		missing.name: missing,
		unique.name: unique,
		cardinality.name: cardinality
	}

	df = pd.DataFrame(data, index=index)
	return df.T

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
		card = int(unique_values[col])    # cardinalidad total
		pct = float(cardin_pct[col])      # cardinalidad relativa en %

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

	# Dibujo los pairplots
	max_cols = 5

	for i in range(0, len(final_features), max_cols):
		features_to_plot = final_features[i:i + max_cols]
		sns.pairplot(dataframe[features_to_plot])
		plt.show()

	return selected_features

def get_features_cat_regression(df, target_col='', pvalue=0.05):

	"""
	Identifica variables categóricas significativamente relacionadas con un target numérico usando ANOVA.

	Args:
	df (pd.DataFrame): DataFrame con los datos a analizar.
	target_col (str): Nombre de la columna objetivo (variable numérica continua o discreta con alta cardinalidad).
	pvalue (float): Nivel de significación estadística (por defecto 0.05).

	Returns:
	list: Lista con los nombres de las columnas categóricas que tienen relación estadísticamente significativa
		  con target_col. Retorna None si hay errores en los argumentos de entrada.
	"""
	
	# Verificar que df es un DataFrame
	if not isinstance(df, pd.DataFrame):
		print("Error: El primer argumento debe ser un pandas DataFrame.")
		return None

	# Verificar que df no está vacío
	if df.empty:
		print("Error: El DataFrame está vacío.")
		return None

	# Verificar que target_col es un string
	if not isinstance(target_col, str):
		print("Error: El argumento 'target_col' debe ser un string.")
		return None

	# Verificar que target_col no está vacío
	if target_col == '':
		print("Error: El argumento 'target_col' no puede estar vacío.")
		return None

	# Verificar que target_col existe en el DataFrame
	if target_col not in df.columns:
		print(f"Error: La columna '{target_col}' no existe en el DataFrame.")
		return None

	# Verificar que target_col es numérica
	if not pd.api.types.is_numeric_dtype(df[target_col]):
		print(f"Error: La columna '{target_col}' debe ser numérica (continua o discreta con alta cardinalidad).")
		return None

	# Verificar que pvalue está en el rango correcto
	if not isinstance(pvalue, (int, float)) or not (0 < float(pvalue) < 1):
		print("Error: El argumento 'pvalue' debe ser un número entre 0 y 1 (exclusivo).")
		return None

	# Verificar que target_col tiene suficiente variabilidad (no es constante)
	if df[target_col].nunique() <= 1:
		print(f"Error: La columna '{target_col}' debe tener más de un valor único para realizar análisis de regresión.")
		return None

	# Identificar columnas categóricas (excluyendo el target)
	categorical_columns = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
	# categorical_columns = [c for c in categorical_columns if c != target_col]

	# Si no hay columnas categóricas, retornar lista vacía
	if len(categorical_columns) == 0:
		print("Advertencia: No se encontraron columnas categóricas en el DataFrame.")
		return []

	# Lista para almacenar las columnas significativas
	significant_features = []

	# Realizar test ANOVA para cada columna categórica
	for col in categorical_columns:
		try:
			# Eliminar filas con valores nulos en la columna categórica o en el target
			df_clean = df[[col, target_col]].dropna()

			# Verificar que hay suficientes datos después de eliminar nulos
			if len(df_clean) < 3:
				continue

			# Obtener grupos (categorías únicas)
			groups = df_clean[col].unique()

			# Necesitamos al menos 2 grupos para ANOVA
			if len(groups) < 2:
				continue

			# Crear lista de arrays con los valores del target para cada grupo
			group_data = []
			for group in groups:
				group_values = df_clean.loc[df_clean[col] == group, target_col].values
				# Solo incluir grupos con al menos 1 observación
				if len(group_values) > 0:
					group_data.append(group_values)

			# Verificar que tenemos al menos 2 grupos con datos
			if len(group_data) < 2:
				continue

			# Realizar test ANOVA (one-way)
			# f_statistic: estadístico F
			# p_value: p-valor del test
			f_statistic, p_value = stats.f_oneway(*group_data)

			# Si el p-valor es menor que el nivel de significación, la relación es significativa
			if p_value < pvalue:
				significant_features.append(col)

		except Exception as e:
			# Si hay algún error con esta columna, continuar con la siguiente
			continue

	return significant_features

def plot_features_cat_regression(df, target_col='', columns=[], pvalue=0.05, with_individual_plot=False):

	"""
	Visualiza la relación entre variables categóricas y un target numérico mediante histogramas agrupados.

	Args:
	df (pd.DataFrame): DataFrame con los datos a analizar.
	target_col (str): Nombre de la columna objetivo (variable numérica).
	columns (list): Lista de nombres de columnas categóricas a visualizar. Si está vacía, se usan todas las categóricas (por defecto []).
	pvalue (float): Nivel de significación estadística para filtrar variables (por defecto 0.05).
	with_individual_plot (bool): Si True, crea un plot individual para cada variable. Si False, crea subplots (por defecto False).

	Returns:
	list: Lista con los nombres de las columnas categóricas que cumplen el criterio de significación estadística.
		  Retorna None si hay errores en los argumentos de entrada.
	"""
	
	# Verificar que df es un DataFrame
	if not isinstance(df, pd.DataFrame):
		print("Error: El primer argumento debe ser un pandas DataFrame.")
		return None

	# Verificar que df no está vacío
	if df.empty:
		print("Error: El DataFrame está vacío.")
		return None

	# Verificar que target_col es un string
	if not isinstance(target_col, str):
		print("Error: El argumento 'target_col' debe ser un string.")
		return None

	# Verificar que target_col no está vacío
	if target_col == '':
		print("Error: El argumento 'target_col' no puede estar vacío.")
		return None

	# Verificar que target_col existe en el DataFrame
	if target_col not in df.columns:
		print(f"Error: La columna '{target_col}' no existe en el DataFrame.")
		return None

	# Verificar que target_col es numérica
	if not pd.api.types.is_numeric_dtype(df[target_col]):
		print(f"Error: La columna '{target_col}' debe ser numérica (continua o discreta con alta cardinalidad).")
		return None

	# Verificar que columns es una lista
	if not isinstance(columns, list):
		print("Error: El argumento 'columns' debe ser una lista.")
		return None

	# Verificar que pvalue está en el rango correcto
	if not isinstance(pvalue, (int, float)) or pvalue <= 0 or pvalue >= 1:
		print("Error: El argumento 'pvalue' debe ser un número entre 0 y 1 (exclusivo).")
		return None

	# Verificar que with_individual_plot es booleano
	if not isinstance(with_individual_plot, bool):
		print("Error: El argumento 'with_individual_plot' debe ser un booleano (True o False).")
		return None

	# Verificar que target_col tiene suficiente variabilidad
	if df[target_col].nunique() <= 1:
		print(f"Error: La columna '{target_col}' debe tener más de un valor único para realizar análisis de regresión.")
		return None

	# Verificar que columns es una lista
	if not isinstance(columns, list):
		print("Error: El argumento 'columns' debe ser una lista.")
		return None

	# Si me pasan columns, primero compruebo que todas existen (test: columna inexistente -> None)
	if len(columns) != 0:
		for col in columns:
			if col not in df.columns:
				print(f"Error: La columna '{col}' especificada en 'columns' no existe en el DataFrame.")
				return None

	cat_cols = set(df.select_dtypes(include=["object", "category", "bool"]).columns)

	# Si columns está vacío, identificar todas las columnas categóricas
	if len(columns) == 0:
		columns = [col for col in df.columns if col in cat_cols and col != target_col]
	else:
	# Si columns no está vacío, nos quedamos SOLO con las categóricas (para cumplir el test)
		columns = [col for col in columns if col in cat_cols and col != target_col]

	# Verificar que todas las columnas en 'columns' existen en el DataFrame
	for col in columns:
		if col not in df.columns:
			print(f"Error: La columna '{col}' especificada en 'columns' no existe en el DataFrame.")
			return None

	# Si no hay columnas categóricas, retornar lista vacía
	if len(columns) == 0:
		print("Advertencia: No se encontraron columnas categóricas para visualizar.")
		return []

	# Filtrar columnas usando get_features_cat_regression
	significant_features = []

	for col in columns:
		try:
			# Eliminar filas con valores nulos
			df_clean = df[[col, target_col]].dropna()

			# Verificar que hay suficientes datos
			if len(df_clean) < 3:
				continue

			# Obtener grupos
			groups = df_clean[col].unique()

			# Necesitamos al menos 2 grupos
			if len(groups) < 2:
				continue

			# Crear lista de arrays con los valores del target para cada grupo
			group_data = []
			for group in groups:
				group_values = df_clean[df_clean[col] == group][target_col].values
				if len(group_values) > 0:
					group_data.append(group_values)

			# Verificar que tenemos al menos 2 grupos con datos
			if len(group_data) < 2:
				continue

			# Realizar test ANOVA
			f_statistic, p_value = stats.f_oneway(*group_data)

			# Si es significativo, agregarlo a la lista
			if p_value < pvalue:
				significant_features.append(col)

		except Exception as e:
			continue

	# Si no hay features significativas, informar y retornar lista vacía
	if len(significant_features) == 0:
		print(
			f"Advertencia: Ninguna de las columnas categóricas tiene relación estadísticamente significativa con '{target_col}' al nivel de significación {pvalue}."
		)
		return []

	# Visualizar las features significativas
	n_features = len(significant_features)

	if with_individual_plot:
		# Crear un plot individual para cada feature
		for col in significant_features:
			plt.figure(figsize=(10, 6))

			# Eliminar valores nulos
			df_clean = df[[col, target_col]].dropna()

			# Obtener categorías únicas
			categories = sorted(df_clean[col].unique())

			# Crear histogramas agrupados
			for category in categories:
				data = df_clean[df_clean[col] == category][target_col]
				plt.hist(data, alpha=0.6, label=str(category), bins=20)

			plt.xlabel(target_col)
			plt.ylabel('Frecuencia')
			plt.title(f'Distribución de {target_col} por {col}')
			plt.legend(title=col)
			plt.grid(True, alpha=0.3)
			plt.tight_layout()
			plt.show()

	else:
		# Crear subplots
		n_cols = min(3, n_features)
		n_rows = (n_features + n_cols - 1) // n_cols

		fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

		# Asegurar que axes sea siempre un array
		if n_features == 1:
			axes = np.array([axes])
		else:
			axes = axes.flatten() if n_features > 1 else np.array([axes])

		for idx, col in enumerate(significant_features):
			ax = axes[idx]

			# Eliminar valores nulos
			df_clean = df[[col, target_col]].dropna()

			# Obtener categorías únicas
			categories = sorted(df_clean[col].unique())

			# Crear histogramas agrupados
			for category in categories:
				data = df_clean[df_clean[col] == category][target_col]
				ax.hist(data, alpha=0.6, label=str(category), bins=20)

			ax.set_xlabel(target_col)
			ax.set_ylabel('Frecuencia')
			ax.set_title(f'Distribución de {target_col} por {col}')
			ax.legend(title=col, fontsize=8)
			ax.grid(True, alpha=0.3)

		# Ocultar ejes sobrantes
		for idx in range(n_features, len(axes)):
			axes[idx].axis('off')

		plt.tight_layout()
		plt.show()

	return significant_features
