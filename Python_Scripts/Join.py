#%%
from Load_Data import load_catalogs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

catalogs = load_catalogs()
cat_fortin = catalogs['cat_fortin']
cat_neuman = catalogs['cat_neuman']

def normalize_name(name):
    """
    Normalize names: remove spaces and convert to uppercase.
    """
    if isinstance(name, str):
        return ''.join(name.upper().split())
    return ''

def build_final_dataframe(cat_fortin, cat_neuman):
    """
    Construye un DataFrame final combinando catálogos Fortin y Neuman,
    normalizando los nombres para hacer join, filtrando objetos comunes,
    y calculando variables derivadas.
    
    Mantiene los nombres originales de las columnas de IDs ('Main_ID' y 'Name').
    """
    cols_f = ['Main_ID', 'Spectype', 'Mo', 'Period', 'Eccentricity', 'Spin_period', 'Distance', 'Class']
    cols_n = ['Name', 'SpType', 'Mean_Mass', 'Teff', 'N_H',
              'Max_Soft_Flux', 'Min_Soft_Flux', 'Max_Hard_Flux', 'Min_Hard_Flux']

    cat_fortin = cat_fortin.copy()
    cat_neuman = cat_neuman.copy()
    
    cat_fortin['norm_id'] = cat_fortin['Main_ID'].apply(normalize_name)
    cat_neuman['norm_id'] = cat_neuman['Name'].apply(normalize_name)

    common_normalized = set(cat_fortin['norm_id']).intersection(cat_neuman['norm_id'])
    fortin_filtered = cat_fortin[cat_fortin['norm_id'].isin(common_normalized)][cols_f + ['norm_id']]
    neuman_filtered = cat_neuman[cat_neuman['norm_id'].isin(common_normalized)][cols_n + ['norm_id']]
    
    final_dataframe = pd.merge(
        fortin_filtered,
        neuman_filtered,
        on='norm_id',
        how='outer',
        suffixes=('_Fortin', '_Neuman')
    )

    final_dataframe.drop(columns=['norm_id'], inplace=True)

    final_dataframe['Mean_Soft_Flux'] = np.sqrt(
        final_dataframe['Min_Soft_Flux'] * final_dataframe['Max_Soft_Flux']
    )
    final_dataframe['Mean_Hard_Flux'] = np.sqrt(
        final_dataframe['Min_Hard_Flux'] * final_dataframe['Max_Hard_Flux']
    )
    final_dataframe['Hardness'] = (
        final_dataframe['Mean_Hard_Flux'] / final_dataframe['Mean_Soft_Flux']
    )

    final_dataframe.rename(columns={
        "SpType": "SpType_Neuman",   
        "Spectype": "SpType_Fortin", 
        "Mean_Mass": "M_X", 
        "Mo": "M*"
    }, inplace=True, errors='ignore')  

    return final_dataframe


matched_by_two_columns = build_final_dataframe(cat_fortin, cat_neuman)
matched_by_two_columns

def equal_orbital_period(cat_fortin, cat_neuman, percentage=0.05):
    """
    Retorna un DataFrame con los sistemas donde el periodo estelar (Fortin)
    y el periodo orbital (Neuman) coinciden dentro de un porcentaje dado.
    Parámetros:
    - percentage: tolerancia relativa para considerar que los periodos son iguales.
    """
    cat_fortin = cat_fortin.copy()
    cat_neuman = cat_neuman.copy()

    cat_fortin['norm_id'] = cat_fortin['Main_ID'].apply(normalize_name)
    cat_neuman['norm_id'] = cat_neuman['Name'].apply(normalize_name)

    common = set(cat_fortin['norm_id']).intersection(cat_neuman['norm_id'])
    fortin_sel = cat_fortin[cat_fortin['norm_id'].isin(common)][['norm_id', 'Period']]
    neuman_sel = cat_neuman[cat_neuman['norm_id'].isin(common)][['norm_id', 'Porb']]

    df = pd.merge(fortin_sel, neuman_sel, on='norm_id', how='inner')
    df = df.dropna(subset=['Period', 'Porb'])

    relative_diff = np.abs(df['Period'] - df['Porb']) / df['Porb']
    df_equal = df[relative_diff <= percentage]

    return df_equal


missing_porb = cat_neuman['Porb'].isna().sum()
missing_period = cat_fortin['Period'].isna().sum()

step_size = 0.00001



max_percentage=0.1






percentages = np.arange(0, max_percentage + step_size, step_size)

ns = []
systems_by_tolerance = []  

for percentage in percentages:
    equal_systems = equal_orbital_period(cat_fortin, cat_neuman, percentage)
    ns.append(len(equal_systems))
    systems_by_tolerance.append(equal_systems)

delta_n = np.diff(ns)
delta_p = np.diff(percentages)
growth_rate = delta_n / delta_p

growth_drop = np.diff(growth_rate)
inflexion_index = np.argmin(growth_drop) + 1  

inflexion_tolerance = percentages[inflexion_index]
df_period = systems_by_tolerance[inflexion_index]

plt.figure(figsize=(9, 6))
plt.plot(percentages * 100, ns, linewidth=2, label='Number of matching systems')
plt.axvline(inflexion_tolerance * 100, color='red', linestyle='--',
            label=f'Inflexion point ≈ {inflexion_tolerance*100:.2f}%')

plt.xlabel('Tolerance (%)')
plt.ylabel('Number of systems found')
plt.title('Matching between Period and Porb vs. tolerance')
plt.legend()
plt.grid(True)


textstr = (f'Missing values:\n'
           f'Porb (Neuman): {missing_porb}\n'
           f'Period (Fortin): {missing_period}\n\n'
           f'Optimal tolerance:\n{inflexion_tolerance*100:.2f}%\n'
           f'Systems matched:\n{len(df_period)}')

plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()


ids_equal = systems_by_tolerance[inflexion_index]['norm_id'].unique()


df_period = matched_by_two_columns[matched_by_two_columns['Main_ID'].isin(ids_equal)].reset_index(drop=True)


def get_id_name_columns(df):
    return [col for col in df.columns if ('id' in col.lower()) or ('name' in col.lower())]


def find_matching_systems_with_columns_skip_nan(cat_fortin, cat_neuman):
    fortin_id_cols = get_id_name_columns(cat_fortin)
    neuman_id_cols = get_id_name_columns(cat_neuman)

    cat_neuman_norm = cat_neuman.copy()
    for col in neuman_id_cols:
        cat_neuman_norm[col] = cat_neuman_norm[col].astype(str).str.upper().str.replace(' ', '')

    matched_rows = []
    matched_info = []

    for idx, row in cat_fortin.iterrows():
        found_match = False
        for col_f in fortin_id_cols:
            val_raw = row[col_f]
            if pd.isna(val_raw) or str(val_raw).strip() == '':
                continue

            val = str(val_raw).upper().replace(' ', '')
            for col_n in neuman_id_cols:
                if val in cat_neuman_norm[col_n].values:
                    matched_rows.append(row)
                    matched_info.append({
                        'matched_value': val,
                        'fortin_column': col_f,
                        'neuman_column': col_n
                    })
                    found_match = True
                    break
            if found_match:
                break

    matched_by_all_ids = pd.DataFrame(matched_rows).reset_index(drop=True)
    df_columns = pd.DataFrame(matched_info)

    return matched_by_all_ids, df_columns


matched_by_all_ids, df_columns = find_matching_systems_with_columns_skip_nan(cat_fortin, cat_neuman)

print(f"Matching systems found: {len(matched_by_all_ids)}")
print("Information about matched columns:")
print(df_columns)

ids_final = set(matched_by_two_columns['Main_ID'])
ids_matched = set(matched_by_all_ids['Main_ID'])
ids_period = set(df_period['Main_ID'])

only_in_final_vs_matched = ids_final - ids_matched
only_in_matched_vs_final = ids_matched - ids_final

only_in_final_vs_period = ids_final - ids_period
only_in_period_vs_final = ids_period - ids_final

only_in_matched_vs_period = ids_matched - ids_period
only_in_period_vs_matched = ids_period - ids_matched

print(f"Systems in matched_by_two_columns but not in matched_by_all_ids: {len(only_in_final_vs_matched)}")
print(f"Systems in matched_by_all_ids but not in matched_by_two_columns: {len(only_in_matched_vs_final)}")

print(f"Systems in matched_by_two_columns but not in df_period: {len(only_in_final_vs_period)}")
print(f"Systems in df_period but not in matched_by_two_columns: {len(only_in_period_vs_final)}")

print(f"Systems in matched_by_all_ids but not in df_period: {len(only_in_matched_vs_period)}")
print(f"Systems in df_period but not in matched_by_all_ids: {len(only_in_period_vs_matched)}")






def closest_system_pairs(cat_fortin, n_systems=5):
    periods = cat_fortin['Period'].dropna().values
    main_ids = cat_fortin.loc[cat_fortin['Period'].notna(), 'Main_ID'].values
    n = len(periods)

    diff_matrix = np.abs(periods.reshape(-1, 1) - periods.reshape(1, -1))
    min_matrix = np.minimum(np.abs(periods.reshape(-1, 1)), np.abs(periods.reshape(1, -1)))

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_diff_matrix = np.where(min_matrix > 0, diff_matrix / min_matrix, np.nan)

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if not np.isnan(relative_diff_matrix[i, j]):
                pairs.append((main_ids[i], main_ids[j], relative_diff_matrix[i, j]))

    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    closest_pairs = pairs_sorted[:n_systems]

    df_closest = pd.DataFrame(closest_pairs, columns=['System1', 'System2', 'RelativeDifference'])

    return df_closest

n = len(cat_fortin)
num_pairs = n * (n - 1) // 2

df_closest_pairs = closest_system_pairs(cat_fortin, n_systems=num_pairs)

def scatter_relative_differences_all(df_closest):
    plt.figure(figsize=(10, 5))
    relative_diff_percent = df_closest['RelativeDifference'] * 100

    plt.scatter(range(len(df_closest)), relative_diff_percent, color='orange')
    plt.xlabel('Pair index')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Relative Difference (%)')  # etiqueta con porcentaje
    plt.title(f'Relative Differences between top {len(df_closest)} closest system pairs')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    
    from matplotlib.ticker import FuncFormatter
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}%'))

    plt.show()

scatter_relative_differences_all(df_closest_pairs)



#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

column_matches_counts = df_columns.groupby(['fortin_column', 'neuman_column']).size().reset_index(name='count')
print(column_matches_counts)

plt.figure(figsize=(12, 6))
sns.barplot(data=column_matches_counts, x='fortin_column', y='count', hue='neuman_column')
plt.title('Count of Matches by Fortin and Neuman Columns')
plt.ylabel('Number of Matches')
plt.xlabel('Fortin Columns')
plt.xticks(rotation=45)
plt.legend(title='Neuman Columns')
plt.tight_layout()
plt.show()

pivot_table = column_matches_counts.pivot(index='fortin_column', columns='neuman_column', values='count').fillna(0)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='g', cmap='Blues')
plt.title('Heatmap of Matches between Fortin and Neuman Columns')
plt.ylabel('Fortin Columns')
plt.xlabel('Neuman Columns')
plt.tight_layout()
plt.show()



# What is the astrophysical or physical criterion to define the maximum acceptable tolerance when matching the Period and Porb values?

# Is there a standard or reference value in the literature for the tolerance threshold when matching by period?

# Should we complement the period matching with other identifiers (such as normalized names) to ensure the systems are truly the same?

# How do you recommend handling systems with missing values in Period or Porb to avoid losing valuable data?

# What visualizations or metrics would you suggest to evaluate the quality and robustness of the catalog matching?

# How can we improve or make the matching process more efficient for future catalogs or larger databases?
