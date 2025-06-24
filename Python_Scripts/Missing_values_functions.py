# %%

from Load_data_functions import load_catalogs

catalogs = load_catalogs()

# %%
catalogs['cat_fortin'].columns
# %%
import random
import random
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from fancyimpute import SoftImpute, IterativeSVD
def model_selection():
    """
    Interactiva: permite seleccionar entre imputar todo el dataset o una simulación experimental
    donde se eliminan artificialmente parámetros para evaluar métodos de imputación.

    Returns:
        df_original, df_imputed, df_comparation (con diferencias entre original e imputado en opción 2)
    """


    df = catalogs['cat_fortin'].copy()
    numeric_columns = ['Period', 'Spin_period', 'RAdeg', 'DEdeg']

    print("¿Qué deseas hacer?")
    print("1. Imputar usando todos los sistemas (sin modificar)")
    print("2. Filtrar sistemas sin valores faltantes en parámetros, eliminar parámetros aleatorios y luego imputar")

    option = input(">>> ").strip()

    if option == '1':
        df_original = df.copy()

    elif option == '2':
        df_clean = df.dropna(subset=numeric_columns).reset_index(drop=True)
        n = len(df_clean)
        half = n // 2
        df_subset = df_clean.iloc[:half].copy()

        df_before_removal = df_subset.copy()
        for index in df_subset.index:
            num_col_to_remove = 1#random.randint(1, len(numeric_columns) // 2)
            cols_to_remove = random.sample(numeric_columns, num_col_to_remove)
            df_subset.loc[index, cols_to_remove] = np.nan

        df_original = df_subset.copy()
    else:
        print("Opción no válida. Se usará la opción 1 por defecto.")
        df_original = df.copy()

    print("\nMétodo para imputar valores numéricos:")
    print("1. Media")
    print("2. Mediana")
    print("3. KNN (scikit-learn)")
    print("4. Imputación constante")
    print("5. SoftImpute (fancyimpute)")
    print("6. Iterative SVD (fancyimpute)")
    print("7. Iterative Imputer (sklearn)")
    print("8. MICE (fancyimpute)")
    numeric_imputation = input(">>> ").strip()

    X_numeric = df_original[numeric_columns].to_numpy()

    if numeric_imputation == '1':
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X_numeric)
    elif numeric_imputation == '2':
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X_numeric)
    elif numeric_imputation == '3':
        imputer = KNNImputer(n_neighbors=3)
        X = imputer.fit_transform(X_numeric)
    elif numeric_imputation == '4':
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X = imputer.fit_transform(X_numeric)
    elif numeric_imputation == '5':
        X = SoftImpute().fit_transform(X_numeric)
    elif numeric_imputation == '6':
        k_max = min(X_numeric.shape) - 1
        if k_max < 1:
            raise ValueError("La matriz es muy pequeña para aplicar IterativeSVD.")
        
        X = IterativeSVD(
            rank=k_max,
            max_iters=1000,  
            convergence_threshold=1e-4,
            verbose=True
        ).fit_transform(X_numeric)

    elif numeric_imputation == '7':
        imputer = IterativeImputer(random_state=42)
        X = imputer.fit_transform(X_numeric)
        
    elif numeric_imputation == '8':
        print("Usando MICE (Multiple Imputation by Chained Equations) con IterativeImputer")
        imputer = IterativeImputer(random_state=100, max_iter=2000, sample_posterior=True)
        X = imputer.fit_transform(X_numeric)

    else:
        print("Método no válido. Se usará media por defecto.")
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X_numeric)

    df_imputed = df_original.copy()
    df_imputed[numeric_columns] = pd.DataFrame(X, columns=numeric_columns)

    print("\nMethod for imputation of the variable 'Class'?")
    print("1. Media")
    print("2. Mediana")
    print("3. Modo (más frecuente)")
    print("4. KNN")
    class_imputation = input(">>> ").strip()

    if class_imputation == '4':
        df_temp = df_imputed.copy()
        target_column = 'Class'
        le = LabelEncoder()

        not_null_mask = df_temp[target_column].notna()
        df_temp.loc[not_null_mask, target_column] = le.fit_transform(df_temp.loc[not_null_mask, target_column])

        df_complete = df_temp[not_null_mask]

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X[not_null_mask.values], df_complete[target_column].astype(int))
        y_pred = knn.predict(X[~not_null_mask.values])

        y_full = df_temp[target_column].copy()
        y_full.loc[~not_null_mask] = y_pred
        y_labels = le.inverse_transform(y_full.astype(int))

        df_imputed['Class'] = y_labels

    else:
        le_used = False
        if df_imputed['Class'].dtype == 'object':
            le = LabelEncoder()
            df_imputed['Class'] = le.fit_transform(df_imputed['Class'].astype(str).replace('nan', np.nan))
            le_used = True

        if class_imputation == '1': 
            class_strategy = 'mean'
        elif class_imputation == '2': 
            class_strategy = 'median'
        elif class_imputation == '3': 
            class_strategy = 'most_frequent'
        else:
            print("Opción no válida. Se usará 'most_frequent' por defecto.")
            class_strategy = 'most_frequent'

        imputer_class = SimpleImputer(strategy=class_strategy)
        y_imputed = imputer_class.fit_transform(df_imputed[['Class']]).flatten()

        if le_used:
            y_labels = le.inverse_transform(y_imputed.astype(int))
        else:
            y_labels = y_imputed

        df_imputed['Class'] = y_labels

    return_columns = ['Main_ID'] + numeric_columns

    if option == '2':
        df_merged_1 = df_before_removal[return_columns].merge(
            df_original[return_columns],
            on='Main_ID',
            suffixes=('_before_removal', '_with_nans')
        )

        df_complete = df_merged_1.merge(
            df_imputed[return_columns],
            on='Main_ID'
        )

        new_names = {col: col + '_imputado' for col in numeric_columns}
        df_complete.rename(columns=new_names, inplace=True)

    else:
        df_complete = None 

    return df_complete


# %%
df_complete = model_selection()
#%%
df_complete