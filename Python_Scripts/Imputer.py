#%%
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from Load_Data import load_catalogs
from IPython.display import display
from Plots import plot_missing_values_distribution, plot_missing_values_by_column, plot_conditional_probability_matrix, highlight_missing_values, corbet_diagram
from Plots import correlation_matrices


catalogs = load_catalogs()
df_catalog = catalogs['cat_fortin']

columns_not_interesting = ['ID', 'Name', 'ref', 'Ref', 'err', 'RA', 'Dec', 'var', 'Err', 'DE', 'Superorbital Period']
numeric_cols_to_use = ['Mx', 'Mo', 'Period',  'Eccentricity', 'RV','Spin_period']
categoric_col_to_use = ['Class']
n_systems=9
divisor = 10
n_neighbours = 5


columns_to_consider = numeric_cols_to_use + categoric_col_to_use


df_original = (
    df_catalog[columns_to_consider]
    .assign(non_null_count=lambda df: df.notna().sum(axis=1))
    .sort_values(by='non_null_count', ascending=False)
    .drop(columns='non_null_count')
    .head(n_systems)
    .reset_index(drop=True)
)

df_with_nans = df_original.copy()
np.random.seed(42)

nan_positions = {}  


for col in numeric_cols_to_use:
    n = len(df_with_nans)
    idx_nan = np.random.choice(n, size=max(1, n // divisor), replace=False)
    nan_positions[col] = idx_nan
    df_with_nans.loc[idx_nan, col] = np.nan



n = len(df_with_nans)
idx_nan_cat = np.random.choice(n, size=max(1, n // divisor), replace=False)
df_with_nans.loc[idx_nan_cat, categoric_col_to_use[0]] = np.nan
nan_positions[categoric_col_to_use[0]] = idx_nan_cat


numerical_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=n_neighbours)),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('imputer', KNNImputer(n_neighbors=n_neighbours))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numeric_cols_to_use),
    ('cat', categorical_pipeline, categoric_col_to_use)
])


df_imputed_array = preprocessor.fit_transform(df_with_nans)


df_imputed = pd.DataFrame(df_imputed_array, columns=numeric_cols_to_use + categoric_col_to_use, index=df_with_nans.index)


scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
df_imputed[numeric_cols_to_use] = scaler.inverse_transform(df_imputed[numeric_cols_to_use])


encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
df_imputed[categoric_col_to_use] = encoder.inverse_transform(df_imputed[categoric_col_to_use])


df_comparacion = pd.DataFrame()

for col in numeric_cols_to_use:
    df_comparacion[f'{col}_original'] = df_original[col]
    df_comparacion[f'{col}_with_nans'] = df_with_nans[col]
    df_comparacion[f'{col}_imputed'] = df_imputed[col]


cat_col = categoric_col_to_use[0]
df_comparacion[f'{cat_col}_original'] = df_original[cat_col]
df_comparacion[f'{cat_col}_with_nans'] = df_with_nans[cat_col]
df_comparacion[f'{cat_col}_imputed'] = df_imputed[cat_col]


df_comparacion['non_null_count'] = df_original[numeric_cols_to_use + categoric_col_to_use].notna().sum(axis=1)


df_comparacion = df_comparacion.sort_values(by='non_null_count', ascending=False).reset_index(drop=True)


df_comparacion_display = df_comparacion.drop(columns='non_null_count')


def highlight_row(row):
    styles = [''] * len(row)
    for base in numeric_cols_to_use:
        col_nan = f'{base}_with_nans'
        col_before = f'{base}_original'
        col_after = f'{base}_imputed'
        if pd.isna(row.get(col_nan, None)):
            try:
                idx_nan = row.index.get_loc(col_nan)
                idx_before = row.index.get_loc(col_before)
                idx_after = row.index.get_loc(col_after)
                styles[idx_nan] = 'background-color: maroon'
                styles[idx_before] = 'background-color: navy; color: white'
                styles[idx_after] = 'background-color: darkgreen; color: white'
            except KeyError:
                continue
    return styles


styled = df_comparacion_display.style.apply(highlight_row, axis=1)


display(styled)

# corbet_diagram(df_comparacion)
correlation_matrices(df_comparacion, numeric_cols_to_use)


plot_missing_values_distribution(df_catalog, columns_not_interesting)
plot_missing_values_by_column(df_catalog, columns_not_interesting)
plot_conditional_probability_matrix(df_catalog, columns_not_interesting)
highlight_missing_values(df_catalog, columns_not_interesting)















#%%
import optuna
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from Load_Data import load_catalogs
import optuna.samplers


np.random.seed(42)

catalogs = load_catalogs()
df_catalog = catalogs['cat_fortin']

numeric_cols_to_use = ['Mx', 'Mo', 'Period', 'RV','Spin_period', "Eccentricity"]
categoric_col_to_use = ['Class']
columns_to_consider = numeric_cols_to_use + categoric_col_to_use

n_systems = 9
n_trials = 300

df_original = (
    df_catalog[[*columns_to_consider, 'Main_ID']]
    .assign(non_null_count=lambda df: df.notna().sum(axis=1))
    .sort_values(by='non_null_count', ascending=False)
    .drop(columns='non_null_count')
    .head(n_systems)
    .reset_index(drop=True)
)



# Si es None, se optimizan todas las columnas (multiobjetivo); si es una str, se optimiza solo esa
opt_parameter = "Mx"  # Ejemplo: opt_parameter = "Period"


def objective(trial):
    np.random.seed(42)
    n_neighbors = trial.suggest_int("n_neighbors", 2, 15)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    scaler_name = trial.suggest_categorical("scaler", ["standard", "minmax", "robust", "none"])

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        scaler = None

    numerical_steps = [('imputer', KNNImputer(n_neighbors=n_neighbors, weights=weights))]
    if scaler is not None:
        numerical_steps.append(('scaler', scaler))
    numerical_pipeline = Pipeline(numerical_steps)
    
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('imputer', KNNImputer(n_neighbors=n_neighbors))  
        ])


    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numeric_cols_to_use),
        ('cat', categorical_pipeline, categoric_col_to_use)
    ])

    errors_por_col = {col: [] for col in numeric_cols_to_use}

    for i in range(n_systems):
        for col in numeric_cols_to_use:
            df_with_nan = df_original.drop(columns='Main_ID').copy()
            true_value = df_with_nan.loc[i, col]
            df_with_nan.loc[i, col] = np.nan

            np.random.seed(42)
            df_imputed_array = preprocessor.fit_transform(df_with_nan)
            df_imputed = pd.DataFrame(df_imputed_array, columns=columns_to_consider, index=df_with_nan.index)

            imputed_value = df_imputed.loc[i, col]
            error_rel = np.abs((imputed_value - true_value) / true_value)
            errors_por_col[col].append(error_rel)

    if opt_parameter is None:
        return [np.mean(errors_por_col[col]) for col in numeric_cols_to_use]
    else:
        return np.mean(errors_por_col[opt_parameter])



sampler = optuna.samplers.TPESampler(seed=42)  # o RandomSampler(seed=42)

if opt_parameter is None:
    study = optuna.create_study(directions=["minimize"] * len(numeric_cols_to_use), sampler=sampler)
else:
    study = optuna.create_study(direction="minimize", sampler=sampler)



study.optimize(objective, n_trials=n_trials)

print("Best trials:", study.best_trials)


if opt_parameter is None:
    best_params = study.best_trials[0].params
else:
    best_params = study.best_params



scaler_map = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(),
    "none": None
}
scaler = scaler_map[best_params["scaler"]]

numerical_steps = [('imputer', KNNImputer(n_neighbors=best_params["n_neighbors"]))]
if scaler is not None:
    numerical_steps.append(('scaler', scaler))
numerical_pipeline = Pipeline(numerical_steps)

categorical_pipeline = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('imputer', KNNImputer(n_neighbors=best_params["n_neighbors"]))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numeric_cols_to_use),
    ('cat', categorical_pipeline, categoric_col_to_use)
])

error_relativos = pd.DataFrame(index=df_original['Main_ID'], columns=numeric_cols_to_use, dtype=float)

for i in range(n_systems):
    for col in numeric_cols_to_use:
        df_with_nan = df_original.drop(columns='Main_ID').copy()
        true_value = df_with_nan.loc[i, col]
        df_with_nan.loc[i, col] = np.nan

        np.random.seed(42)
        df_imputed_array = preprocessor.fit_transform(df_with_nan)
        df_imputed = pd.DataFrame(df_imputed_array, columns=columns_to_consider, index=df_with_nan.index)

        imputed_value = df_imputed.loc[i, col]
        error_rel = np.abs((imputed_value - true_value) / true_value)
        error_relativos.loc[df_original.loc[i, 'Main_ID'], col] = error_rel

promedios = error_relativos.mean(axis=0)
error_relativos.loc['Mean_Relative_Error'] = promedios

error_relativos
#%%
best_params

#%%
df_original