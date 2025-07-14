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

columns_not_interesting = ['ID', 'Name', 'ref', 'Ref', 'err', 'RA', 'Dec', 'var', 'Err', 'DE']
numeric_cols_to_use = ['Mx', 'Mo', 'Period',  'Eccentricity', 'RV','Spin_period']
categoric_col_to_use = ['Class']
n_systems=150
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

corbet_diagram(df_comparacion)
correlation_matrices(df_comparacion, numeric_cols_to_use)


plot_missing_values_distribution(df_catalog, columns_not_interesting)
plot_missing_values_by_column(df_catalog, columns_not_interesting)
plot_conditional_probability_matrix(df_catalog, columns_not_interesting)
highlight_missing_values(df_catalog, columns_not_interesting)































# Questions about data and variables
# 1. What physical or astronomical interpretations can we give to the variables Mx, Mo, Period, Eccentricity, RV, and Spin_period?
# 2. Which of these variables tend to be more reliable or have less observational error?
# 3. How do missing values typically affect astronomical catalogs, and what methods are commonly used in astronomy to handle them?
# 4. In your experience, which categorical variables (like Class) are most relevant for classifying stellar systems?

# Questions about analysis and results
# 5. How important is it to analyze the correlation between variables like Period, Eccentricity, and Spin_period in binary systems?
# 6. Are there physical relationships or theoretical models that explain the observed correlations?
# 7. What additional information could be sought in the catalog to improve statistical and predictive analyses?
# 8. How would you assess data quality after imputing missing values? What metrics or validations do you recommend?

# Questions to better understand the systems
# 9. What criteria or characteristics define the classes in the Class column?
# 10. What role does orbital eccentricity (Eccentricity) play in the evolution of the system?
# 11. Which systems or astronomical variables have the most impact on observable phenomena like pulsations or X-ray emissions?
