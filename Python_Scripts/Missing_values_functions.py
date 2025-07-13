#%%
from Load_data_functions import load_catalogs
from Plots_functions import corbet_diagram
import random
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from fancyimpute import SoftImpute
from IPython.display import display
from Plots_functions import plot_missing_values_distribution, plot_missing_values_by_column, plot_conditional_probability_matrix, highlight_missing_values

def get_scaler(name):
    if name == 'standard':
        return StandardScaler()
    elif name == 'minmax':
        return MinMaxScaler()
    elif name == 'robust':
        return RobustScaler()
    elif name == 'maxabs':
        return MaxAbsScaler()
    elif name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scaler: {name}")

def choose_experiment_mode(df, numeric_columns, top_n=None):
    print("What would you like to do?")
    print("1. Impute using all systems (without modification)")
    print("2. Filter systems without missing values in parameters, remove random parameters, and then impute")

    option = input(">>> ").strip()

    if option == '1':
        df_selected = df.copy()
        if top_n is not None:
            df_selected["non_missing_count"] = df_selected[numeric_columns].notna().sum(axis=1)
            df_selected = df_selected.sort_values("non_missing_count", ascending=False).head(top_n)
            df_selected.drop(columns="non_missing_count", inplace=True)
        return df_selected, None, option

    elif option == '2':
        df["non_missing_count"] = df[numeric_columns].notna().sum(axis=1)
        df_clean = df.sort_values("non_missing_count", ascending=False).head(top_n).drop(columns="non_missing_count").reset_index(drop=True)
        df_subset, df_before_removal = remove_random_values(df_clean, numeric_columns)
        return df_subset.copy(), df_before_removal.copy(), option

    else:
        print("Invalid option. Defaulting to option 1.")
        return df.copy(), None, '1'

def remove_random_values(df_clean, numeric_columns):
    df_subset = df_clean.iloc[:len(df_clean) // 2].copy()
    df_before_removal = df_subset.copy()

    for index in df_subset.index:
        if len(numeric_columns) == 0:
            continue
        max_remove = max(1, len(numeric_columns) // 2)
        num_cols_to_remove = random.randint(1, min(max_remove, len(numeric_columns)))
        cols_to_remove = random.sample(numeric_columns, num_cols_to_remove)
        df_subset.loc[index, cols_to_remove] = np.nan

    return df_subset, df_before_removal

def knn_imputation_optuna(X_numeric, df_original, df_before_removal, numeric_columns, timeout):
    print("Optimizing n_neighbors and scaler for KNNImputer using Optuna...")
    true_values = df_before_removal[numeric_columns].to_numpy()

    def objective(trial):
        scaler_name = trial.suggest_categorical('scaler', ['none', 'standard', 'minmax', 'robust', 'maxabs'])
        n_neighbors = trial.suggest_int("n_neighbors", 1, 15)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("metric", ["nan_euclidean"])

        scaler = get_scaler(scaler_name)
        if scaler is not None:
            X_scaled = scaler.fit_transform(X_numeric)
        else:
            X_scaled = X_numeric.copy()

        knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)
        X_imputed_scaled = knn_imputer.fit_transform(X_scaled)

        if scaler is not None:
            try:
                X_imputed = scaler.inverse_transform(X_imputed_scaled)
            except Exception:
                X_imputed = X_imputed_scaled
        else:
            X_imputed = X_imputed_scaled

        mse_total, count = 0, 0
        for i, col in enumerate(numeric_columns):
            mask = df_original[col].isna().to_numpy()
            if np.sum(mask) == 0:
                continue
            y_true = true_values[mask, i]
            y_pred = X_imputed[mask, i]

            valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if np.sum(valid_mask) == 0:
                continue 
            mse_total += mean_squared_error(y_true[valid_mask], y_pred[valid_mask])
            count += 1

        return np.sqrt(mse_total / count) if count > 0 else float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, timeout=timeout)
    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    scaler = get_scaler(best_params['scaler'])
    if scaler is not None:
        X_scaled = scaler.fit_transform(X_numeric)
    else:
        X_scaled = X_numeric.copy()

    knn_imputer = KNNImputer(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        metric=best_params['metric']
    )
    X_imputed_scaled = knn_imputer.fit_transform(X_scaled)

    if scaler is not None:
        try:
            X_imputed = scaler.inverse_transform(X_imputed_scaled)
        except Exception:
            X_imputed = X_imputed_scaled
    else:
        X_imputed = X_imputed_scaled

    return X_imputed

def select_imputation_method(X_numeric, df_original, numeric_columns, df_before_removal, option, timeout):
    print("\nMethod for imputing numerical values:")
    print("1. Mean")
    print("2. Median")
    print("3. KNN (scikit-learn)")
    print("4. Constant Imputation")
    print("5. SoftImpute (fancyimpute)")
    print("6. Iterative SVD (fancyimpute)")
    print("7. Iterative Imputer (sklearn)")
    print("8. MICE (fancyimpute)")
    imputation_method = input(">>> ").strip()

    if imputation_method == '1':
        return SimpleImputer(strategy='mean').fit_transform(X_numeric)
    elif imputation_method == '2':
        return SimpleImputer(strategy='median').fit_transform(X_numeric)
    elif imputation_method == '3':
        if option != '2':
            print("Optuna optimization requires option 2 (simulation).")
            return None
        return knn_imputation_optuna(X_numeric, df_original, df_before_removal, numeric_columns, timeout)
    elif imputation_method == '4':
        return SimpleImputer(strategy='constant', fill_value=0).fit_transform(X_numeric)
    elif imputation_method == '5':
        return SoftImpute().fit_transform(X_numeric)
    elif imputation_method == '6':
        if option != '2':
            print("Optuna optimization requires option 2 (simulation).")
            return None
        return svd_imputation_optuna(X_numeric, df_original, df_before_removal, numeric_columns, timeout)
    elif imputation_method == '7':
        return IterativeImputer(random_state=42).fit_transform(X_numeric)
    elif imputation_method == '8':
        print("Using MICE (Multiple Imputation by Chained Equations)")
        return IterativeImputer(random_state=100, max_iter=2000, sample_posterior=True).fit_transform(X_numeric)
    else:
        print("Invalid method. Defaulting to mean.")
        return SimpleImputer(strategy='mean').fit_transform(X_numeric)

def impute_class_variable(df_imputed, X):
    print("\nMethod for imputing the variable 'Class'?")
    print("1. Mean")
    print("2. Median")
    print("3. Most Frequent")
    print("4. KNN")
    class_imputation = input(">>> ").strip()
    target_column = 'Class'

    if class_imputation == '4':
        le = LabelEncoder()
        mask_not_null = df_imputed[target_column].notna()
        df_temp = df_imputed.copy()
        df_temp.loc[mask_not_null, target_column] = le.fit_transform(df_temp.loc[mask_not_null, target_column])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_scaled[mask_not_null], df_temp.loc[mask_not_null, target_column].astype(int))
        preds = knn.predict(X_scaled[~mask_not_null])

        y_all = df_temp[target_column].copy()
        y_all.loc[~mask_not_null] = preds
        labels = le.inverse_transform(y_all.astype(int))
        df_imputed[target_column] = labels
    else:
        le_used = False
        if df_imputed[target_column].dtype == 'object':
            le = LabelEncoder()
            df_imputed[target_column] = le.fit_transform(df_imputed[target_column].astype(str).replace('nan', np.nan))
            le_used = True

        strategy = {
            '1': 'mean',
            '2': 'median',
            '3': 'most_frequent'
        }.get(class_imputation, 'most_frequent')

        y_imputed = SimpleImputer(strategy=strategy).fit_transform(df_imputed[[target_column]]).flatten()
        df_imputed[target_column] = (
            le.inverse_transform(y_imputed.astype(int)) if le_used else y_imputed
        )

    return df_imputed

def build_result_dataframe(df_before_removal, df_original, df_imputed, numeric_columns, option):
    return_columns = ['Main_ID'] + numeric_columns + ['Class']
    if option == '2':
        df1 = df_before_removal[return_columns].merge(
            df_original[return_columns], on='Main_ID', suffixes=('_before_removal', '_with_nans')
        )
        df2 = df1.merge(df_imputed[return_columns], on='Main_ID')
        rename_dict = {col: col + '_imputed' for col in numeric_columns}
        df2.rename(columns=rename_dict, inplace=True)
        return df2
    else:
        return None

def model_selection(catalogs, numeric_columns, timeout=600, top_n=50):
    df = catalogs['cat_fortin'].copy()
    df_original, df_before_removal, option = choose_experiment_mode(df, numeric_columns, top_n=top_n)
    X_numeric = df_original[numeric_columns].to_numpy()
    X = select_imputation_method(
        X_numeric, df_original, numeric_columns,
        df_before_removal, option, timeout
    )
    df_imputed = df_original.copy()
    df_imputed[numeric_columns] = pd.DataFrame(X, columns=numeric_columns)
    df_imputed = impute_class_variable(df_imputed, X)
    df_complete = build_result_dataframe(
        df_before_removal, df_original, df_imputed, numeric_columns, option
    )
    return df_complete


def highlight_row(row):
    styles = [''] * len(row)
    for base in numeric_cols_to_use:  
        col_nan = f'{base}_with_nans'
        col_before = f'{base}_before_removal'
        col_after = f'{base}_imputed'
        if pd.isna(row.get(col_nan, None)):
            try:
                idx_nan = row.index.get_loc(col_nan)
                idx_before = row.index.get_loc(col_before)
                idx_after = row.index.get_loc(col_after)
                styles[idx_nan] = 'background-color: maroon'
                styles[idx_before] = 'background-color: navy'
                styles[idx_after] = 'background-color: darkgreen'
            except KeyError:
                continue
    return styles


catalogs = load_catalogs()
df_catalog = catalogs['cat_fortin']

print("Which data would you like to use?")
print("1. Real dataset (cat_fortin)")
print("2. Synthetic dataset (df_fake)")

dataset_choice = input(">>> ").strip()

if dataset_choice == '1':
    df_base = df_catalog.copy()
elif dataset_choice == '2':
    print("How many rows would you like to generate for the synthetic dataset?")
    try:
        n = int(input(">>> ").strip())
        if n <= 0:
            print("The number must be greater than 0. Defaulting to n = 100.")
            n = 100
    except ValueError:
        print("Invalid input. Defaulting to n = 100.")
        n = 100

    np.random.seed(42)
    main_ids = [f"src_{i:04d}" for i in range(n)]
    data = {'Main_ID': main_ids}

    numeric_cols_all = df_catalog.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_catalog.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in numeric_cols_all:
        col_data = pd.to_numeric(df_catalog[col], errors='coerce')
        col_min = col_data.min()
        col_max = col_data.max()
        try:
            data[col] = np.random.uniform(low=col_min, high=col_max, size=n)
        except OverflowError:
            print(f"Column '{col}' ignored due to overflow.")
            continue

    for col in cat_cols:
        unique_vals = df_catalog[col].dropna().unique()
        if len(unique_vals) == 0:
            print(f"Column '{col}' ignored due to no non-null values.")
            continue
        data[col] = np.random.choice(unique_vals, size=n, replace=True)

    df_base = pd.DataFrame(data)
else:
    print("Invalid option. Defaulting to synthetic dataset with n = 100.")
    n = 100
    np.random.seed(42)
    main_ids = [f"src_{i:04d}" for i in range(n)]
    data = {'Main_ID': main_ids}

    numeric_cols_all = df_catalog.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_catalog.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in numeric_cols_all:
        col_data = pd.to_numeric(df_catalog[col], errors='coerce')
        col_min = col_data.min()
        col_max = col_data.max()
        try:
            data[col] = np.random.uniform(low=col_min, high=col_max, size=n)
        except OverflowError:
            print(f"Column '{col}' ignored due to overflow.")
            continue

    for col in cat_cols:
        unique_vals = df_catalog[col].dropna().unique()
        if len(unique_vals) == 0:
            print(f"Column '{col}' ignored due to no non-null values.")
            continue
        data[col] = np.random.choice(unique_vals, size=n, replace=True)

    df_base = pd.DataFrame(data)


numeric_cols_to_use = ['Mx', 'Mo', 'Period', 'Spin_period', 'Eccentricity', 'RV']
numeric_cols_to_use = [col for col in numeric_cols_to_use if col in df_base.columns]

try:
    top_n = int(input("How many of the most complete systems would you like to use? >>> ").strip())
except ValueError:
    top_n = None

df_highlighted = model_selection({'cat_fortin': df_base}, numeric_cols_to_use, timeout=30, top_n=top_n)

styled_df = df_highlighted.style.apply(highlight_row, axis=1)
display(styled_df)

corbet_diagram(df_highlighted)

#%%



columns_not_interesting = ['ID', 'Name', 'ref', 'Ref', 'err', 'RA', 'Dec', 'var', 'Err', 'DE']

plot_missing_values_distribution(df_catalog, columns_not_interesting)
plot_missing_values_by_column(df_catalog, columns_not_interesting)
plot_conditional_probability_matrix(df_catalog, columns_not_interesting)
highlight_missing_values(df_catalog, columns_not_interesting)
