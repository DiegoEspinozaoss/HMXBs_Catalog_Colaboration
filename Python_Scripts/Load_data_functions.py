
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import seaborn as sns
# from mpl_toolkits.basemap import Basemap
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.image as mpimg
# import astropy.units as u
import plotly.express as px
import ipywidgets as widgets
from ipywidgets import interact
import venn
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance as ssd
import warnings
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import parallel_coordinates
import plotly.graph_objects as go
from sklearn.feature_selection import f_regression
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from fancyimpute import SoftImpute, IterativeSVD, KNN
# %%
from pathlib import Path




















def load_catalogs():
    """
    Load and preprocess catalogs: Neumann, Fortin, Kim (transient and persistent), and Malacaria (persistent and transient).
    Returns:
        dict: A dictionary containing the loaded datasets.
    """
    base_path = Path(__file__).resolve().parent.parent / "Datasets"
    excel_name = base_path / "HMXB_catalogs.xlsx"
    excel_name_2 = base_path / "HMXB_cat.xlsx"

    excel_file = pd.ExcelFile(excel_name)

    cat_neuman = excel_file.parse('HMXB_cat_Neumann')
    cat_neuman['Geometric_Mean'] = np.sqrt(cat_neuman['Max_Soft_Flux'] * cat_neuman['Min_Soft_Flux'])
    cat_neuman_update = pd.read_excel(excel_name_2, sheet_name="HMXB_cat")


    cat_fortin = excel_file.parse('v2023-09_Fortin')
    
    kim_transient = excel_file.parse('kim_transient')
    kim_persistent = excel_file.parse('kim_persistent')
    
    malacaria_persistent = excel_file.parse('malacaria_persistent')
    malacaria_transient = excel_file.parse('malacaria_transient')

    cat_kim_transient = kim_transient.iloc[302:367, 0]
    cat_kim_transient = cat_kim_transient.str.split(expand=True)

    cat_kim_persistent = kim_persistent.iloc[173:192, 0]
    cat_kim_persistent = cat_kim_persistent.str.split(expand=True)

    return {
        "cat_neuman": cat_neuman,
        "cat_fortin": cat_fortin,
        "cat_kim_transient": cat_kim_transient,
        "cat_kim_persistent": cat_kim_persistent,
        "cat_malacaria_persistent": malacaria_persistent,
        "cat_malacaria_transient": malacaria_transient,
        "cat_neuman_update": cat_neuman_update,
    }

# %%
catalogs = load_catalogs()






#%%
import optuna
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import optuna
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import optuna

# Generar dataset base
X, y = make_regression(n_samples=30, n_features=7, noise=0.1, random_state=42)
df_original = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])

# Crear máscara de missing aleatoria en la columna 'X2'
missing_mask = np.zeros(df_original.shape[0], dtype=bool)
missing_mask[:6] = True
np.random.shuffle(missing_mask)

# Guardar valores reales antes de introducir NaNs
ground_truth_X2 = df_original['X2'].copy()

# Dataset con NaNs
df_with_nan = df_original.copy()
df_with_nan.loc[missing_mask, 'X2'] = np.nan

# Función de evaluación para Optuna
def objective(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 2, 50)  # rango amplio
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    add_indicator = trial.suggest_categorical("add_indicator", [True, False])
    keep_empty_features = trial.suggest_categorical("keep_empty_features", [True, False])

    # Crear el imputador con todos los hiperparámetros
    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="nan_euclidean",  # único valor permitido por ahora
        add_indicator=add_indicator,
        keep_empty_features=keep_empty_features
    )

    # Imputar sobre una copia
    df_temp = df_with_nan.copy()
    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="nan_euclidean",
        add_indicator=add_indicator,
        keep_empty_features=keep_empty_features
    )
    imputed_array = imputer.fit_transform(df_temp)

    # Cuando add_indicator=True, se añaden columnas extra (indicadores)
    # Pero para evaluar solo nos interesan las columnas originales, que son las primeras df_temp.shape[1]
    df_imputed = pd.DataFrame(imputed_array[:, :df_temp.shape[1]], columns=df_temp.columns)

    y_true = ground_truth_X2[missing_mask]
    y_pred = df_imputed.loc[missing_mask, 'X2']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


# Ejecutar la optimización
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# Mostrar mejores hiperparámetros
print("Best params:", study.best_params)

# ⚠️ Imputar nuevamente con los mejores hiperparámetros y guardar
best_imputer = KNNImputer(
    n_neighbors=study.best_params["n_neighbors"],
    weights=study.best_params["weights"]
)


df_imputed_best = pd.DataFrame(best_imputer.fit_transform(df_with_nan), columns=df_with_nan.columns)

# Paso 1: Identificar las posiciones donde hubo imputación
# Es decir, donde df_with_nan tiene NaN
mask_imputadas = df_with_nan.isna()

# Paso 2: Recorremos todas las columnas y filas donde se imputó, y comparamos
cambios = []

for col in df_original.columns:
    for idx in df_original.index:
        if mask_imputadas.loc[idx, col]:
            valor_original = df_original.loc[idx, col]
            valor_imputado = df_imputed_best.loc[idx, col]
            cambios.append((idx, col, valor_original, valor_imputado))

# Paso 3: Imprimir las duplas
print("Cambios detectados (índice, columna, valor_original, valor_imputado):\n")
for idx, col, original, imputado in cambios:
    print(f"Fila {idx}, Columna '{col}': {original:.4f} → {imputado:.4f}")
