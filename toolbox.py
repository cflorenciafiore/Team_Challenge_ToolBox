import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')



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


def tipifica_variables(dataframe: pd.DataFrame,
					   umbral_categoria: int,
					   umbral_continua: float) -> pd.DataFrame: # {{{
	"""
	Calcula la cardinalidad de las columnas y categoriza las columnas del
	dataframe siguiendo las siguientes reglas:
		- Sí la cardinalidad de la columna es igual a dos se categoriza como
		  binaria.
		- Sí la cardinalidad de la columna es menor a umbral_categoria se asigna
		  como categórica.
		- Sí la cardinalidad de la columna es mayor a umbral_categoria:
			- Se asigna numérica discreta si la cardinalidad es menor a
			  umbral_continua.
			- Se asigna numérica contínua sí la cardinalidad es mayor o igual a
			  umbral_continua.


	Args:
		dataframe: pd.DataFrame
			un dataframe

		umbral_categorica: int
			unmbral que permite identificar sí una variable es categórica.

		umbral_continua: float
			umbral que permite identificar sí una variable es númerica continua
			o discreta.

	Return:
		pd.DataFrame
	"""
	if dataframe is None:
		raise ValueError("Dataframe sin especificar.")

	if dataframe.empty:
		raise ValueError("Dataframe vacío.")
	
	if umbral_categoria is None:
		raise ValueError("Umbral categoría sin especificar.")

	if umbral_continua is None:
		raise ValueError("Umbral contínua sin especificar.")

	if umbral_categoria > umbral_continua:
		raise ValueError("Umbral categoría no puede ser mayor a umbral contínua.")

	size = dataframe.shape[0]

	unique = dataframe.nunique()
	unique.name = "unique"

	is_binary = unique == 2
	binary = pd.DataFrame(data={ "category": "Binaria" },
						  index=unique[is_binary].index)

	cardinality = pd.Series(data=unique/size * 100,
							name="cardinality")

	is_category = cardinality[~is_binary] < umbral_categoria
	is_continuous = cardinality[~is_binary] >= umbral_continua
	categorized = np.select(
			[is_category, is_continuous],
			["Categórica", "Numérica Continua"],
			"Numérica Discreta"
	)

	data = pd.concat([
			binary,
			pd.Series(categorized,
					  index=cardinality[~is_binary].index,
					  name="category")
	])
	data["cardinality"] = cardinality.round(2)


	return data.sort_index()
# }}}


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

    # Selecciono las columnas numéricas
    numeric_cols = dataframe.select_dtypes(include="number")

    # Exluyo al target de las columnas numéricas
    numeric_features = numeric_cols.columns.drop(target_col)

    # Compruebo que target_col no sea la única columna numérica
    if len(numeric_features) == 0:
        print("No hay columnas numéricas para analizar.")
        return None
    
    # Calculo correlaciones para filtrar
    selected_features = []

    for col in numeric_features:
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
        return final_features
    else:
        return selected_features

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

def plot_features_num_regression(dataframe, target_col, columns=[], umbral_corr=0, pvalue=None):

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



def get_features_cat_regression(df, target_col='', pvalue=0.05):
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
    
    Identifica variables categóricas significativamente relacionadas con un target numérico usando ANOVA.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos a analizar.
    target_col (str): Nombre de la columna objetivo (variable numérica continua o discreta con alta cardinalidad).
    pvalue (float): Nivel de significación estadística (por defecto 0.05).

    Retorna:
    list: Lista con los nombres de las columnas categóricas que tienen relación estadísticamente significativa 
          con target_col. Retorna None si hay errores en los argumentos de entrada.
    """
    #  Verificar que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: El primer argumento debe ser un pandas DataFrame.")
        return None
    
    #  Verificar que df no está vacío
    if df.empty:
        print("Error: El DataFrame está vacío.")
        return None
    
    #  Verificar que target_col es un string
    if not isinstance(target_col, str):
        print("Error: El argumento 'target_col' debe ser un string.")
        return None
    
    #  Verificar que target_col no está vacío
    if target_col == '':
        print("Error: El argumento 'target_col' no puede estar vacío.")
        return None
    
    # : Verificar que target_col existe en el DataFrame
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no existe en el DataFrame.")
        return None
    
    #  Verificar que target_col es numérica
    if df[target_col].dtype not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        print(f"Error: La columna '{target_col}' debe ser numérica (continua o discreta con alta cardinalidad).")
        return None
    
    #  Verificar que pvalue está en el rango correcto
    if not isinstance(pvalue, (int, float)) or pvalue <= 0 or pvalue >= 1:
        print("Error: El argumento 'pvalue' debe ser un número entre 0 y 1 (exclusivo).")
        return None
    
    #  Verificar que target_col tiene suficiente variabilidad (no es constante)
    if df[target_col].nunique() <= 1:
        print(f"Error: La columna '{target_col}' debe tener más de un valor único para realizar análisis de regresión.")
        return None
    
    # Identificar columnas categóricas (excluyendo el target)
    categorical_columns = []
    
    for col in df.columns:
        if col == target_col:
            continue
        
        # Una columna es categórica si:
        # 1. Es de tipo object o category
        # 2. Es numérica pero con pocos valores únicos (típicamente <= 10)
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_columns.append(col)
        elif df[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            if df[col].nunique() <= 10:
                categorical_columns.append(col)
    
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
                group_values = df_clean[df_clean[col] == group][target_col].values
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
	Esta función recibe un dataframe, una argumento "target_col" con valor por
	defecto "", una lista de strings ("columns") cuyo valor por defecto es la
	lista vacía, un argumento ("pvalue") con valor 0.05 por defecto y un
	argumento "with_individual_plot" a False.

	Si la lista no está vacía, la función pintará los histogramas agrupados de
	la variable "target_col" para cada uno de los valores de las variables
	categóricas incluidas en columns que cumplan que su test de relación con
	"target_col" es significativo para el nivel 1-pvalue de significación
	estadística. La función devolverá los valores de "columns" que cumplan con
	las condiciones anteriores.

	Si la lista está vacía, entonces la función igualará "columns" a las
	variables numéricas del dataframe y se comportará como se describe en el
	párrafo anterior.

	De igual manera que en la función descrita anteriormente deberá hacer un
	check de los valores de entrada y comportarse como se describe en el último
	párrafo de la función `get_features_cat_regression`.

    Visualiza la relación entre variables categóricas y un target numérico mediante histogramas agrupados.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos a analizar.
    target_col (str): Nombre de la columna objetivo (variable numérica).
    columns (list): Lista de nombres de columnas categóricas a visualizar. Si está vacía, se usan todas las categóricas (por defecto []).
    pvalue (float): Nivel de significación estadística para filtrar variables (por defecto 0.05).
    with_individual_plot (bool): Si True, crea un plot individual para cada variable. Si False, crea subplots (por defecto False).

    Retorna:
    list: Lista con los nombres de las columnas categóricas que cumplen el criterio de significación estadística.
          Retorna None si hay errores en los argumentos de entrada.
    """
    #  Verificar que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: El primer argumento debe ser un pandas DataFrame.")
        return None
    
    #  Verificar que df no está vacío
    if df.empty:
        print("Error: El DataFrame está vacío.")
        return None
    
    #  Verificar que target_col es un string
    if not isinstance(target_col, str):
        print("Error: El argumento 'target_col' debe ser un string.")
        return None
    
    #  Verificar que target_col no está vacío
    if target_col == '':
        print("Error: El argumento 'target_col' no puede estar vacío.")
        return None
    
    #  Verificar que target_col existe en el DataFrame
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no existe en el DataFrame.")
        return None
    
    #  Verificar que target_col es numérica
    if df[target_col].dtype not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        print(f"Error: La columna '{target_col}' debe ser numérica (continua o discreta con alta cardinalidad).")
        return None
    
    #  Verificar que columns es una lista
    if not isinstance(columns, list):
        print("Error: El argumento 'columns' debe ser una lista.")
        return None
    
    #  Verificar que pvalue está en el rango correcto
    if not isinstance(pvalue, (int, float)) or pvalue <= 0 or pvalue >= 1:
        print("Error: El argumento 'pvalue' debe ser un número entre 0 y 1 (exclusivo).")
        return None
    
    #  Verificar que with_individual_plot es booleano
    if not isinstance(with_individual_plot, bool):
        print("Error: El argumento 'with_individual_plot' debe ser un booleano (True o False).")
        return None
    
    #  Verificar que target_col tiene suficiente variabilidad
    if df[target_col].nunique() <= 1:
        print(f"Error: La columna '{target_col}' debe tener más de un valor único para realizar análisis de regresión.")
        return None
    
    # Si columns está vacío, identificar todas las columnas categóricas
    if len(columns) == 0:
        columns = []
        for col in df.columns:
            if col == target_col:
                continue
            
            # Una columna es categórica si es object, category o numérica con pocos valores únicos
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                columns.append(col)
            elif df[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                if df[col].nunique() <= 10:
                    columns.append(col)
    
    #  Verificar que todas las columnas en 'columns' existen en el DataFrame
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
        print(f"Advertencia: Ninguna de las columnas categóricas tiene relación estadísticamente significativa con '{target_col}' al nivel de significación {pvalue}.")
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