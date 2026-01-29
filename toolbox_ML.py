import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr, mannwhitneyu, f_oneway

#####################################################################

def describe_df(data:pd.DataFrame) -> pd.DataFrame:
    '''
    Genera una descripción esquemática de las variables de un DataFrame, devolviendo un nuevo DataFrame en el que cada columna corresponde a una variable
    del conjunto de datos original y cada fila resume una propiedad de dicha variable.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada del que se desea obtener un resumen descriptivo de sus variables.

    Returns
    -------
    pandas.DataFrame
        DataFrame resumen que contiene, para cada variable del DataFrame original, información relativa al tipo de dato, porcentaje de valores nulos,
        número de valores únicos y porcentaje de cardinalidad.
    '''

    salida = {columna: [] for columna in data.columns}
    for columna in salida.keys():
        salida[columna].append(data[columna].dtype)
        salida[columna].append(round(data[columna].isna().sum() / len(data) * 100, 1))
        salida[columna].append(data[columna].nunique())
        salida[columna].append(round(data[columna].nunique() / len(data) * 100, 2))

    salida['Columnas'] = ['DATA_TYPE', 'MISSINGS (%)', 'UNIQUE_VALUES', 'CARDIN (%)']

    return pd.DataFrame(salida).set_index('Columnas')


def tipifica_variables(data:pd.DataFrame, umbral_categoria:int, umbral_continua:float) -> pd.DataFrame:
    '''
    Asigna un tipo sugerido a cada variable de un DataFrame según su cardinalidad, indicando si debe considerarse categórica o numérica (continua o
    discreta).

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada con las variables a analizar.
    umbral_categoria : int
        Umbral de cardinalidad para diferenciar variables categóricas de numéricas.
    umbral_continua : float
        Porcentaje de cardinalidad mínimo para considerar una variable numérica como continua.

    Returns
    -------
    pandas.DataFrame
        DataFrame con dos columnas: 'nombre_variable' y 'tipo_sugerido', con una fila por cada variable del DataFrame original.
    '''

    describe = describe_df(data)

    salida = [[columna] for columna in data.columns]
    for columna in salida:
        tipo_sugerido = 'Numérica Discreta'
        if describe.loc['UNIQUE_VALUES', columna].values[0] == 2:
            tipo_sugerido = 'Binaria'

        elif describe.loc['CARDIN (%)', columna].values[0] < umbral_categoria:
            tipo_sugerido = 'Categórica'

        elif describe.loc['CARDIN (%)', columna].values[0] >= umbral_continua:
            tipo_sugerido = 'Numérica Continua'

        columna.append(tipo_sugerido)

    return pd.DataFrame(salida, columns=['nombre_variable', 'tipo_sugerido'])


def get_features_num_regression(data:pd.DataFrame, target_col:str, umbral_corr:float, pvalue:float=None) -> list:
    '''
    Devuelve una lista de columnas numéricas de un DataFrame que presentan una correlación significativa con una variable objetivo, según un umbral
    definido y, opcionalmente, un nivel de significación.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada que contiene las variables explicativas y la variable objetivo.
    target_col : str
        Nombre de la columna del DataFrame que se utilizará como variable objetivo.
    umbral_corr : float
        Umbral mínimo de correlación (en valor absoluto) para seleccionar una variable.
    pvalue : float, opcional
        Nivel de significación estadística para aceptar la variable. Por defecto es None.

    Returns
    -------
    list
        Lista de nombres de columnas numéricas que cumplen los criterios de correlación y significación estadística.
    '''
    
    pvalue = 1 if pvalue == None else pvalue

    if not(isinstance(data, pd.DataFrame) and isinstance(target_col, str) and isinstance(umbral_corr, (int, float)) and isinstance(pvalue, (int, float))):
        return print('Introduce el tipo de variable adecuado para cada parámetro.')
    
    columnas_numericas = data.select_dtypes(include=['number']).columns

    if not target_col in columnas_numericas:
        return print('La columna objetivo debe existir y ser numérica.')

    salida = []
    for columna in columnas_numericas:
        if columna != target_col:
            corr_calc, pvalue_calc = pearsonr(data[columna], data[target_col])
            if corr_calc >= umbral_corr and pvalue_calc <= pvalue:
                salida.append(columna)    

    return salida


def plot_features_num_regression(data:pd.DataFrame, target_col:str, columns:list=[], umbral_corr:float=0, pvalue:float=None) -> None:
    '''
    Genera gráficos pairplot para variables numéricas relacionadas con una variable objetivo, filtrando las columnas a representar según su correlación y,
    opcionalmente, un test de significación estadística.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada que contiene las variables a analizar.
    target_col : str, opcional
        Columna del DataFrame que se utilizará como variable objetivo.
    columns : list de str, opcional
        Lista de nombres de columnas candidatas a representar. Si está vacía, se utilizan todas las variables numéricas.
    umbral_corr : float, opcional
        Umbral mínimo de correlación para incluir una variable en los gráficos.
    pvalue : float, opcional
        Nivel de significación estadística para los tests de correlación.
        
    Returns
    -------
    list
    Lista de nombres de las columnas seleccionadas para generar los pairplots.
    '''

    columns = data.columns.to_list() if columns == [] else columns
    if not(isinstance(columns, list)):
        return print('Introduce el tipo de variable adecuado para cada parámetro.')

    columns_calc = get_features_num_regression(data, target_col, umbral_corr, pvalue)

    columnas_pintar = [col for col in columns if col in columns_calc]
    divisiones = math.ceil(len(columnas_pintar) / 4)
    for div in range(divisiones):
        col_div = [target_col] + columnas_pintar[4*div:4*(div+1)]
        sns.pairplot(data[col_div], )
    
    return columnas_pintar


def get_features_cat_regression(data:pd.DataFrame, target_col:str,  pvalue:float=0.05) -> list:
    '''
    Devuelve una lista de columnas categóricas de un DataFrame cuya relación con la variable objetivo supera un nivel de significación estadística
    definido.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada que contiene las variables explicativas y la variable objetivo.
    target_col : str
        Columna del DataFrame que se utilizará como variable objetivo (numérica continua o discreta de alta cardinalidad).
    pvalue : float, opcional
        Nivel de significación estadística para los tests de relación. Por defecto es 0.05.

    Returns
    -------
    list
        Lista de nombres de columnas categóricas cuya relación con la variable objetivo es significativa.
    '''

    data = data.dropna(axis=0)

    if not(isinstance(data, pd.DataFrame) and isinstance(target_col, str) and isinstance(pvalue, (int, float))):
        return print('Introduce el tipo de variable adecuado para cada parámetro.')
    
    columnas_categoricas = data.select_dtypes(exclude=['number']).columns

    if not target_col in data.select_dtypes(include=['number']).columns:
        return print('La columna objetivo debe existir y ser numérica.')

    salida = []
    for columna in columnas_categoricas:
        if data[columna].nunique() == 2:
            grupo_a, grupo_b = [data[data[columna] == valor][target_col] for valor in data[columna].unique()]
            _, p_valor = mannwhitneyu(grupo_a, grupo_b)
            if p_valor <= pvalue:
                salida.append(columna)
        else:
            lista = [data[data[columna] == valor][target_col] for valor in data[columna].unique()]
            _, p_valor = f_oneway(*lista)
            if p_valor <= pvalue:
                salida.append(columna)
            
    return salida


def plot_features_cat_regression(data:pd.DataFrame, target_col:str, columns:list=[], pvalue:float=0.05, with_individual_plot:bool=False) -> None:
    '''
    Genera histogramas de la variable objetivo agrupados por los valores de variables categóricas seleccionadas, considerando únicamente aquellas cuya
    relación con la variable objetivo es estadísticamente significativa.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame de entrada que contiene las variables a analizar.
    target_col : str, opcional
        Columna del DataFrame que se utilizará como variable objetivo.
    columns : list de str, opcional
        Lista de nombres de variables categóricas candidatas a representar. Si está vacía, se utilizan todas las variables categóricas.
    pvalue : float, opcional
        Nivel de significación estadística para los tests de relación.
    with_individual_plot : bool, opcional
        Indica si se deben generar gráficos individuales adicionales para cada variable seleccionada.

    Returns
    -------
    list
    Lista de nombres de columnas categóricas que han sido consideradas para los histogramas.
    '''

    data = data.dropna(axis=0)    

    if not isinstance(with_individual_plot, bool):
        return print('Introduce el tipo de variable adecuado para cada parámetro.')

    columns = data.columns.to_list() if columns == [] else columns
    if not(isinstance(columns, list)):
        return print('Introduce el tipo de variable adecuado para cada parámetro.')

    columns_calc = get_features_cat_regression(data, target_col, pvalue)

    columnas_pintar = [col for col in columns if col in columns_calc]

    if with_individual_plot:
        for columna in columnas_pintar:
            valores = data[columna].unique()
            divisiones = math.ceil(len(valores) / 4)
            for div in range(divisiones):
                ncols = min(4, len(valores)-div*4)
                _, axs = plt.subplots(ncols=min(4, len(valores)-div*4), figsize = (5*ncols, 5))
                for i, ax in enumerate(axs):
                    idx = div * 4 + i
                    sns.histplot(data[data[columna] == valores[idx]], x=target_col, ax=ax, fill=True, alpha=0.3)
                    ax.set_xlabel(valores[idx])
                    if i != 0:
                        ax.set_ylabel('')
                    else:
                        ax.set_ylabel(f'{columna} - p-value = 0.005')

    else:
        divisiones = math.ceil(len(columnas_pintar) / 2)
        for div in range(divisiones):
            ncols = min(2, len(columnas_pintar)-div*2)
            _, axs = plt.subplots(ncols=min(2, len(columnas_pintar)-div*2), figsize = (10*ncols, 10))
            for i, ax in enumerate(axs):
                idx = div * 2 + i
                sns.histplot(data, x=target_col, hue=columnas_pintar[idx], ax=ax, fill=True, alpha=0.3)

                ax.set_xlabel(f'{columnas_pintar[idx]} - p-value = 0.005')
                ax.set_ylabel('')
    
    return columnas_pintar