# run_tests.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from toolbox import (
    describe_df,
    tipifica_variables,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression
)
import io
from contextlib import redirect_stdout
import warnings

warnings.filterwarnings(
    "ignore",
    message="all input arrays have length 1.*f_oneway requires that at least one input has length greater than 1.*"
)
warnings.filterwarnings(
    "ignore",
    message="One or more sample arguments is too small; all returned values will be NaN.*"
)
    
DATA_PATH = "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def call_silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return fn(*args, **kwargs)
    
# ===== Helpers =====
def ok(msg): print(f"TEST OK  {msg}")
def fail(msg): raise AssertionError(f"Error {msg}")

def assert_is_df(x, name="La función"):
    if not isinstance(x, pd.DataFrame):
        fail(f"{name} debe devolver un DataFrame, recibido: {type(x)}")

def assert_is_list(x, name="La variable"):
    if not isinstance(x, list):
        fail(f"{name} debe ser una lista, recibido: {type(x)}")

def run_test(name, fn):
    try:
        fn()
        ok(name)
    except Exception as e:
        print(f"Error {name} -> {e}")
        raise


def main():
    # 0. Cargar dataset
    df = pd.read_csv(DATA_PATH)

    # 1. Tests: describe_df
    def test_describe_df_structure():
        res = describe_df(df)
        assert_is_df(res, "describe_df(df)")

        # Debe tener las mismas columnas que las columnas del df original
        if list(res.columns) != list(df.columns):
            fail("describe_df: las columnas deben coincidir EXACTAMENTE con las del dataframe original.")

        # Debe tener filas concretas, pueden estar en cualquier orden
        expected_rows = {"DATA_TYPE", "MISSINGS (%)", "UNIQUE_VALUES", "CARDIN (%)"}
        if not expected_rows.issubset(set(res.index)):
            fail(f"describe_df: el índice debe contener {expected_rows}. Tiene: {set(res.index)}")

        # Valores esperados: MISSINGS (%) numérico y entre 0 y 100
        miss = pd.to_numeric(res.loc["MISSINGS (%)"], errors="coerce")
        if miss.isna().any():
            fail("describe_df: MISSINGS (%) debe ser numérico para todas las columnas.")
        if (miss < 0).any() or (miss > 100).any():
            fail("describe_df: MISSINGS (%) debe estar entre 0 y 100.")

        # UNIQUE_VALUES debe ser entero >= 0
        uniq = pd.to_numeric(res.loc["UNIQUE_VALUES"], errors="coerce")
        if uniq.isna().any():
            fail("describe_df: UNIQUE_VALUES debe ser numérico para todas las columnas.")
        if (uniq < 0).any():
            fail("describe_df: UNIQUE_VALUES no puede ser negativo.")

        print(res)
    # 2. Tests: tipifica_variables
    def test_tipifica_variables_structure():
        res = tipifica_variables(df, umbral_categoria=10, umbral_continua=15.0)
        assert_is_df(res, "tipifica_variables(df,umbral_categoria, umbral_continua)")

        expected_cols = ["nombre_variable", "tipo_sugerido"]
        if list(res.columns) != expected_cols:
            fail(f"tipifica_variables: columnas esperadas {expected_cols}, recibidas {list(res.columns)}")

        if len(res) != df.shape[1]:
            fail("tipifica_variables: debe devolver tantas filas como columnas tenga el dataframe.")

        if set(res["nombre_variable"]) != set(df.columns):
            fail("tipifica_variables: la columna nombre_variable debe contener TODOS los nombres de columnas del df.")

        allowed = {"Binaria", "Categorica", "Numerica Continua", "Numerica Discreta"}
        if not set(res["tipo_sugerido"]).issubset(allowed):
            fail(f"tipifica_variables: tipo_sugerido solo puede contener {allowed}.")

    def test_tipifica_variables_telco_sanity():
        # Esto es muy adhoc y porque conocemos de antemano el dataframe, por eso se hace un sanity check para ver si realmente la función de tipicar es correcta también en su implementación
        res = tipifica_variables(df, umbral_categoria=10, umbral_continua=15.0)

        # Las siguientes columnas son binarias
        expected_binary = {"Churn", "Partner", "Dependents", "PhoneService"}
        # Las cogemos del dataset
        present_binary = expected_binary.intersection(set(df.columns))
        if present_binary:
            # Cogemos el valor que se le ha inyectado
            sub = res.set_index("nombre_variable").loc[list(present_binary), "tipo_sugerido"]
            # Para variables binarias, no lo hacemos muy estricto y dejamos que puedan salir como categóricas también, damos algo de manga
            # Pero para variables numéricas continuas o discretas, sí que no dejamos que pasen como categóricas, dará error
            if (sub == "Numerica Continua").any():
                fail("tipifica_variables: una variable binaria no debería ser 'Numerica Continua'.")
            if (sub == "Numerica Discreta").any():
                fail("tipifica_variables: una variable binaria no debería ser 'Numerica Discreta'.")

    # 3. Tests: get_features_num_regression
    def test_get_features_num_regression_basic():
        # Elegimos un target numérico razonable en Telco:
        target = "MonthlyCharges"

        res = get_features_num_regression(df, target_col=target, umbral_corr=0.5, pvalue=None)
        assert_is_list(res, "get_features_num_regression(...)")
  
        # Debe devolver SOLO nombres de columnas del df
        for col in res:
            if col not in df.columns:
                fail(f"get_features_num_regression: devolvió columna inexistente: {col}")

        # Debe devolver SOLO numéricas (excluyendo el target)
        num_cols = set(df.select_dtypes(include=[np.number]).columns)
        for col in res:
            if col not in num_cols:
                fail(f"get_features_num_regression: devolvió una columna no numérica: {col}")
            if col == target:
                fail("get_features_num_regression: no debe incluir target_col en la lista devuelta.")

    def test_get_features_num_regression_checks():
        # target_col inexistente => None
        out = get_features_num_regression(df, target_col="NO_EXISTE", umbral_corr=0.2, pvalue=None)
        if out is not None:
            fail("get_features_num_regression: si target_col no existe, debe retornar None.")

        # umbral_corr fuera de rango => None
        out = get_features_num_regression(df, target_col="MonthlyCharges", umbral_corr=1.5, pvalue=None)
        if out is not None:
            fail("get_features_num_regression: si umbral_corr no está en [0,1], debe retornar None.")

        # pvalue fuera de rango => None (si se usa)
        out = get_features_num_regression(df, target_col="MonthlyCharges", umbral_corr=0.2, pvalue=1.5)
        if out is not None:
            fail("get_features_num_regression: si pvalue no es None y no está en (0,1], debe retornar None.")

        # target_col no numérica => None (ej: customerID suele ser string)
        if "customerID" in df.columns:
            out = get_features_num_regression(df, target_col="customerID", umbral_corr=0.2, pvalue=None)
            if out is not None:
                fail("get_features_num_regression: si target_col no es numérica, debe retornar None.")

    # 4. Tests: plot_features_num_regression
    def test_plot_features_num_regression_basic():
        target = "MonthlyCharges"

        res = call_silent(
            plot_features_num_regression,
            df,
            target_col=target,
            columns=[],
            umbral_corr=0.1,
            pvalue=None
        )
        
        assert_is_list(res, "plot_features_num_regression(...)")

        # Las columnas devueltas deben existir y ser numéricas
        num_cols = set(df.select_dtypes(include=[np.number]).columns)
        for col in res:
            if col not in df.columns:
                fail(f"plot_features_num_regression: columna inexistente devuelta: {col}")
            if col not in num_cols:
                fail(f"plot_features_num_regression: columna no numérica devuelta: {col}")
            if col == target:
                fail("plot_features_num_regression: no debe incluir target_col en el retorno.")

    def test_plot_features_num_regression_checks():
        # target vacío
        out = call_silent(plot_features_num_regression, df)
        if out is not None:
            fail("plot_features_num_regression: target_col vacío debe retornar None.")

        # target inexistente
        out = call_silent(plot_features_num_regression, df, target_col="NO_EXISTE")
        if out is not None:
            fail("plot_features_num_regression: target_col inexistente debe retornar None.")

        # umbral_corr inválido
        out = call_silent(plot_features_num_regression, df, target_col="MonthlyCharges", umbral_corr=2)
        if out is not None:
            fail("plot_features_num_regression: umbral_corr fuera de rango debe retornar None.")

        # pvalue inválido
        out = call_silent(plot_features_num_regression, df, target_col="MonthlyCharges", pvalue=2)
        if out is not None:
            fail("plot_features_num_regression: pvalue fuera de rango debe retornar None.")

        # target no numérico
        if "customerID" in df.columns:
            out = call_silent(
                plot_features_num_regression,
                df,
                target_col="customerID"
            )
            if out is not None:
                fail("plot_features_num_regression: target no numérico debe retornar None.")
    
    # 5. Tests: get_features_cat_regression
    def test_get_features_cat_regression_basic():
        target = "MonthlyCharges"

        res = call_silent(
            get_features_cat_regression,
            df,
            target_col=target,
            pvalue=0.05
        )
        assert_is_list(res, "get_features_cat_regression(...)")

        # Todas deben ser categóricas
        cat_cols = set(df.select_dtypes(include=["object", "category", "bool"]).columns)
        for col in res:
            if col not in df.columns:
                fail(f"get_features_cat_regression: columna inexistente devuelta: {col}")
            if col not in cat_cols:
                fail(f"get_features_cat_regression: columna no categórica devuelta: {col}")

    def test_get_features_cat_regression_checks():
        # target inexistente
        out = call_silent(get_features_cat_regression, df, "NO_EXISTE")
        if out is not None:
            fail("get_features_cat_regression: target inexistente debe retornar None.")
        # pvalue inválido
        out = call_silent(get_features_cat_regression, df, "MonthlyCharges", pvalue=2)
        if out is not None:
            fail("get_features_cat_regression: pvalue fuera de rango debe retornar None.")

        # target no numérico
        if "customerID" in df.columns:
            out = call_silent(get_features_cat_regression, df, "customerID")
            if out is not None:
                fail("get_features_cat_regression: target no numérico debe retornar None.")
    
    # 6. Tests: plot_features_cat_regression
    def test_plot_features_cat_regression_basic():
        target = "MonthlyCharges"

        res = call_silent(
            plot_features_cat_regression,
            df,
            target_col=target,
            columns=[],
            pvalue=0.05,
            with_individual_plot=False
        )

        assert_is_list(res, "plot_features_cat_regression(...)")

        # Debe devolver solo columnas categóricas existentes
        cat_cols = set(df.select_dtypes(include=["object", "category", "bool"]).columns)
        for col in res:
            if col not in df.columns:
                fail(f"plot_features_cat_regression: devolvió columna inexistente: {col}")
            if col not in cat_cols:
                fail(f"plot_features_cat_regression: devolvió columna no categórica: {col}")

    def test_plot_features_cat_regression_checks():
        # target vacío
        out = call_silent(plot_features_cat_regression, df)
        if out is not None:
            fail("plot_features_cat_regression: target_col vacío debe retornar None.")

        # target inexistente
        out = call_silent(plot_features_cat_regression, df, target_col="NO_EXISTE")
        if out is not None:
            fail("plot_features_cat_regression: target_col inexistente debe retornar None.")

        # pvalue inválido
        out = call_silent(plot_features_cat_regression, df, target_col="MonthlyCharges", pvalue=2)
        if out is not None:
            fail("plot_features_cat_regression: pvalue fuera de rango debe retornar None.")

        # target no numérico
        if "customerID" in df.columns:
            out = call_silent(plot_features_cat_regression, df, target_col="customerID")
            if out is not None:
                fail("plot_features_cat_regression: target no numérico debe retornar None.")

        # columns con columna inexistente
        out = call_silent(plot_features_cat_regression, df, target_col="MonthlyCharges", columns=["NO_EXISTE"])
        if out is not None:
            fail("plot_features_cat_regression: columns con columna inexistente debe retornar None.")

    # ===== Ejecutar tests =====
    run_test("describe_df: estructura y rangos", test_describe_df_structure)
    run_test("tipifica_variables: estructura", test_tipifica_variables_structure)
    run_test("tipifica_variables: sanity check Telco", test_tipifica_variables_telco_sanity)
    run_test("get_features_num_regression: básico", test_get_features_num_regression_basic)
    run_test("get_features_num_regression: checks de entradas", test_get_features_num_regression_checks)
    run_test("plot_features_num_regression: básico", test_plot_features_num_regression_basic)
    run_test("plot_features_num_regression: checks de entradas", test_plot_features_num_regression_checks)
    run_test("get_features_cat_regression: básico", test_get_features_cat_regression_basic)
    run_test("get_features_cat_regression: checks de entradas", test_get_features_cat_regression_checks)
    run_test("plot_features_cat_regression: básico", test_plot_features_cat_regression_basic)
    run_test("plot_features_cat_regression: checks de entradas", test_plot_features_cat_regression_checks)

    print("\nHAN PASADO TODOS LOS TESTS")


if __name__ == "__main__":
    main()
