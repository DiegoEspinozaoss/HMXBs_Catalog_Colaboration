import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as patches


def plot_missing_values_distribution(catalog, exclude_columns):
    """
    Plots the distribution of systems by the number of missing numerical parameters.

    Parameters:
    - catalog (pd.DataFrame): The catalog dataframe to analyze.
    - exclude_columns (list): List of column substrings to exclude from the analysis.
    """
    numeric_columns = [col for col in catalog.select_dtypes(include='number').columns
                       if not any(substr in col for substr in exclude_columns)]

    missing_counts = catalog[numeric_columns].isna().sum(axis=1)
    hist_data = missing_counts.value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(hist_data.index, hist_data.values, color='lightcoral', edgecolor='black')
    plt.xlabel('Number of missing numerical parameters')
    plt.ylabel('Number of systems')
    plt.title('Distribution of systems by number of missing values (numerical variables)')
    plt.xticks(hist_data.index)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print(f"Number of numerical columns analyzed: {len(numeric_columns)}")
    print(f"Columns considered:\n{numeric_columns}")


def plot_missing_values_by_column(catalog, exclude_columns):
    """
    Plots the number of systems with missing values for each numerical parameter.

    Parameters:
    - catalog (pd.DataFrame): The catalog dataframe to analyze.
    - exclude_columns (list): List of column substrings to exclude from the analysis.
    """
    numeric_columns = [col for col in catalog.select_dtypes(include='number').columns
                       if not any(substr in col for substr in exclude_columns)]

    missing_by_column = catalog[numeric_columns].isna().sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(missing_by_column.index, missing_by_column.values, color='skyblue', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}',
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Cantidad de sistemas con valor nulo')
    plt.title('Número de sistemas con valores faltantes por parámetro (variables numéricas)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_conditional_probability_matrix(catalog, exclude_columns):
    """
    Plots a heatmap showing the conditional probability of missing a parameter A
    given that parameter B is also missing.

    Parameters:
    - catalog (pd.DataFrame): The catalog dataframe to analyze.
    - exclude_columns (list): List of column substrings to exclude from the analysis.
    """
    numeric_columns = [col for col in catalog.select_dtypes(include='number').columns
                       if not any(substr in col for substr in exclude_columns)]

    missing_matrix = catalog[numeric_columns].isna().astype(int)

    conditional_prob = pd.DataFrame(index=numeric_columns, columns=numeric_columns, dtype=float)

    for a in numeric_columns:
        for b in numeric_columns:
            b_missing = missing_matrix[b] == 1
            if b_missing.sum() == 0:
                conditional_prob.loc[a, b] = np.nan
            else:
                conditional_prob.loc[a, b] = (missing_matrix[a][b_missing].sum()) / b_missing.sum()

    plt.figure(figsize=(12, 10))
    sns.heatmap(conditional_prob.astype(float), annot=True, fmt=".2f", cmap="Reds", cbar_kws={'label': 'P(A missing | B missing)'})
    plt.title("Conditional Probability Matrix: P(A missing | B missing)")
    plt.xlabel("B (condition)")
    plt.ylabel("A (outcome)")
    plt.tight_layout()
    plt.show()


def highlight_missing_values(catalog, exclude_columns):
    """
    Highlights missing values in a sorted catalog based on the number of missing numerical parameters.

    Parameters:
    - catalog (pd.DataFrame): The catalog dataframe to analyze.
    - exclude_columns (list): List of column substrings to exclude from the analysis.

    Returns:
    - pd.DataFrame.style: A styled dataframe with missing values highlighted.
    """
    def highlight_missing(val):
        return 'background-color: lightcoral' if pd.isna(val) else ''

    numeric_columns = [
        col for col in catalog.select_dtypes(include='number').columns
        if not any(substr in col for substr in exclude_columns)
    ]
    catalog['missing_count'] = catalog[numeric_columns].isna().sum(axis=1)

    catalog_sorted = catalog.sort_values(by='missing_count', ascending=False).reset_index(drop=True)
    return catalog_sorted[numeric_columns].style.applymap(highlight_missing)




def corbet_diagram(df, n_elipses=3):
    spin_col = next((col for col in df.columns if col.startswith('Spin_period') and col.endswith('_imputed')), None)
    period_col = next((col for col in df.columns if col.startswith('Period') and col.endswith('_imputed')), None)
    
    if spin_col is None or period_col is None:
        print("Columns with '_imputed' suffix for 'Spin_Period' or 'Period' not found.")
        return
    
    df_valid = df.dropna(subset=[spin_col, period_col, 'Class_original'])
    top_classes = df_valid['Class_original'].value_counts().nlargest(n_elipses).index.tolist()

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    classes = np.unique(df_valid['Class_original'])

    for i, cls in enumerate(classes):
        class_data = df_valid[df_valid['Class_original'] == cls]
        sns.scatterplot(x=spin_col, y=period_col, data=class_data,
                        marker=markers[i % len(markers)], label=f'Class {cls}', alpha=0.6, ax=ax)

    plt.title('Corbet Diagram with Confidence Ellipses')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(spin_col)
    plt.ylabel(period_col)
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



def correlation_matrices(df_comparacion, numeric_cols_to_use):
    correlation_methods = ['pearson', 'spearman', 'kendall']
    
    original_cols = [f'{col}_original' for col in numeric_cols_to_use]
    imputed_cols = [f'{col}_imputed' for col in numeric_cols_to_use]
    
    df_original = df_comparacion[original_cols]
    df_imputed = df_comparacion[imputed_cols]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex='col', sharey='row')
    
    for i, method in enumerate(correlation_methods):
        corr_orig = df_original.corr(method=method)
        sns.heatmap(corr_orig, ax=axes[0, i], annot=True, cmap='coolwarm', fmt=".2f",
                    cbar=False)
        axes[0, i].set_title(f'{method.capitalize()}')
        
        corr_imputed = df_imputed.corr(method=method)
        sns.heatmap(corr_imputed, ax=axes[1, i], annot=True, cmap='coolwarm', fmt=".2f",
                    cbar=False)
    
    for ax in fig.get_axes():
        ax.label_outer()
    
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Imputed")
    
    fig.suptitle("Matrices of Correlation (Original vs Imputed)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




# def percentage_classes(self, df):
#         class_counts = df['Class'].value_counts()
#         total = class_counts.sum()

#         class_percent = class_counts / total
#         main_classes = class_percent[class_percent >= 0.10]
#         other_classes = class_percent[class_percent < 0.10]

#         final_counts = main_classes.copy()
#         if not other_classes.empty:
#             final_counts['Other'] = other_classes.sum()

#         plt.figure(figsize=(8, 8))
#         plt.pie(final_counts, labels=final_counts.index,
#                 autopct='%1.1f', startangle=90,
#                 colors=plt.cm.tab20.colors[:len(final_counts)],
#                 wedgeprops=dict(edgecolor='k'))

#         plt.title('Distribución de Clases (agrupando <10% en "Other")')
#         plt.axis('equal')  
#         plt.tight_layout()
#         plt.show()




#   def Compare_Neumann_Fortin(self):
#         """
#         It displays the number and names of systems common between Fortin and Neumann catalogs
#         , showing also their single or lonely systems (not common) to each of them.
#         """
#         self.cat_neuman['IDS_list'] = self.cat_neuman['IDS'].str.split(',').apply(
#             lambda lst: [self.normalize_name(x) for x in lst if isinstance(x, str)]
#         )

#         for col in self.cat_neuman.columns:
#             if 'ID' in col:
#                 for index, row in self.cat_neuman.iterrows():
#                     value = row[col]
#                     if isinstance(value, str) and pd.notnull(value):
#                         normalized_value = self.normalize_name(value)
#                         if normalized_value:
#                             self.cat_neuman.at[index, 'IDS_list'].append(normalized_value)

#         neuman_set = set()
#         for _, row in self.cat_neuman.iterrows():
#             neuman_set.add(self.normalize_name(row['Name']))
#             neuman_set.update(row['IDS_list'])

#         fortin_columns = [col for col in self.cat_fortin.columns if 'ID' in col or 'Name' in col]
#         fortin_objects = []

#         for index, row in self.cat_fortin.iterrows():
#             fortin_names = set()
#             rep = None
#             for col in fortin_columns:
#                 value = row[col]
#                 if pd.notnull(value):
#                     norm_val = self.normalize_name(value)
#                     if norm_val:
#                         fortin_names.add(norm_val)
#                         if rep is None:
#                             rep = value
#             if fortin_names:
#                 fortin_objects.append((index, fortin_names, rep))

#         fortin_set = set()
#         for obj in fortin_objects:
#             fortin_set.update(obj[1])

#         common, only_neumann, only_fortin = [], [], []

#         for _, row in self.cat_neuman.iterrows():
#             object_names = set(row['IDS_list'])
#             object_names.add(self.normalize_name(row['Name']))
#             if object_names.intersection(fortin_set):
#                 common.append(row['Name'])
#             else:
#                 only_neumann.append(row['Name'])

#         for _, names_set, rep in fortin_objects:
#             if not names_set.intersection(neuman_set):
#                 if rep is not None and pd.notnull(rep):
#                     only_fortin.append(rep)

#         self.common = [x for x in common if pd.notnull(x)]
#         self.only_neumann = [x for x in only_neumann if pd.notnull(x)]
#         self.only_fortin = [x for x in only_fortin if pd.notnull(x)]

#         print("Common objects:", self.common)
#         print("Number of common objects:", len(self.common))
#         print("\nObjects only in cat_neuman:", self.only_neumann)
#         print("Number of objects only in cat_neuman:", len(self.only_neumann))
#         print("\nObjects only in cat_fortin:", self.only_fortin)
#         print("Number of objects only in cat_fortin:", len(self.only_fortin))


#      def plot_class_distribution(self):
#         """
#         It plots the Distribution of classes in the fortin catalogue.
#         """
#         class_counts = self.cat_fortin['Class'].value_counts()
#         top_classes = class_counts[:4]
#         others_count = class_counts[4:].sum()
#         aggregated_counts = pd.concat([top_classes, pd.Series({'Others': others_count})])

#         plt.figure(figsize=(8, 8))
#         plt.pie(
#             aggregated_counts, 
#             labels=aggregated_counts.index, 
#             autopct='%1.1f', 
#             startangle=90, 
#             colors=plt.cm.tab10.colors[:len(aggregated_counts)]
#         )
#         plt.title('Fortin Distribution of Classes')
#         plt.show()



#     def plot_null_histogram(self, df, title):
#         """
#         It shows the null values for all columns in 
#         the df dataframe
#         """
#         null_counts = df.isnull().sum()
#         non_null_counts = df.notnull().sum()
#         df_nulls = pd.DataFrame({
#             'Nulls': null_counts,
#             'Non-Nulls': non_null_counts
#         })
#         ax = df_nulls.plot(kind='bar', figsize=(10, 6), width=0.8)
#         ax.set_title(f'Null and Non-Null Counts per Column: {title}')
#         ax.set_ylabel('Count')
#         ax.set_xlabel('Columns')
#         plt.xticks(rotation=60, ha='right')
#         plt.tight_layout()
#         plt.show()



#     def compare_null_counts(self, columns_fortin, columns_neuman):
#         """
#         It shows a comparation between the null values for Neuman and Fortin
#         Catalogs.
#         """
#         null_counts_fortin = self.cat_fortin[columns_fortin].isnull().sum()
#         non_null_counts_fortin = self.cat_fortin[columns_fortin].notnull().sum()
#         null_counts_neuman = self.cat_neuman[columns_neuman].isnull().sum()
#         non_null_counts_neuman = self.cat_neuman[columns_neuman].notnull().sum()

#         fig, ax = plt.subplots(figsize=(12, 6))
#         fortin_indexs = np.arange(len(columns_fortin))
#         neumann_indexs = np.arange(len(columns_neuman)) + len(columns_fortin)
#         bar_width = 0.35

#         ax.bar(fortin_indexs - bar_width/2, null_counts_fortin, bar_width, label='Nulls (Fortin)', color='orange')
#         ax.bar(fortin_indexs + bar_width/2, non_null_counts_fortin, bar_width, label='Non-Nulls (Fortin)', color='cyan')
#         ax.bar(neumann_indexs - bar_width/2, null_counts_neuman, bar_width, label='Nulls (Neumann)', color='red', alpha=0.7)
#         ax.bar(neumann_indexs + bar_width/2, non_null_counts_neuman, bar_width, label='Non-Nulls (Neumann)', color='blue', alpha=0.7)

#         all_columns = columns_fortin + columns_neuman
#         ax.set_xticks(np.arange(len(all_columns)))
#         ax.set_xticklabels(all_columns, rotation=60, ha='right')
#         ax.set_ylabel('Count')
#         ax.set_title('Comparison of Null and Non-Null Counts')
#         ax.legend()
#         plt.tight_layout()
#         plt.show()


#     def plot_common_object_heatmap(self, columns_fortin, columns_neuman):
#         """
#         It shows the common objects between Neumann and Fortin, comparing
#         all their ID's columns.
#         """
#         common_count_matrix = np.zeros((len(columns_fortin), len(columns_neuman)))

#         for i, fortin_col in enumerate(columns_fortin):
#             fortin_objects = self.cat_fortin[fortin_col].dropna().unique()
#             for j, neuman_col in enumerate(columns_neuman):
#                 neuman_objects = self.cat_neuman[neuman_col].dropna().unique()
#                 common_objects = set(fortin_objects) & set(neuman_objects)
#                 common_count_matrix[i, j] = len(common_objects)

#         common_count_df = pd.DataFrame(common_count_matrix, index=columns_fortin, columns=columns_neuman)
#         plt.figure(figsize=(12, 8))
#         sns.heatmap(common_count_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Common Objects Count'}, fmt=".0f")
#         plt.title('Heatmap of Common Objects Between Columns of Fortin and Neuman', fontsize=16)
#         plt.xlabel('Neuman Columns', fontsize=14)
#         plt.ylabel('Fortin Columns', fontsize=14)
#         plt.tight_layout()
#         plt.show()



#    def plot_coordinate_groups(self):
#         """
#         Graph the objects in the Neumann catalog according to the 
#         number of times they appear in the different catalogs.
#         """
#         coordinate_groups = {}
#         for i in range(1, 5):
#             elements = self.names_count[self.names_count == i].index.tolist()
#             coordinate_groups[i] = self.cat_neuman[self.cat_neuman["Name"].isin(elements)][["GLON", "GLAT"]]

#         plt.figure(figsize=(10, 6))
#         colours = {1: 'green', 2: 'orange', 3: 'purple', 4: 'red'}

#         for i, coords in coordinate_groups.items():
#             plt.scatter(coords["GLON"], coords["GLAT"], marker='o', color=colours[i], alpha=0.8, label=f'{i} catálogo(s)')

#         plt.title("Coordenadas galácticas (Latitud vs Longitud) en el Catálogo de Neumann")
#         plt.xlabel("GLON [°]")
#         plt.ylabel("GLAT [°]")
#         plt.legend()
#         plt.grid(True)
#         plt.show()
    
    
#     def hammer_proyection_neumann(self):
#         """
#         This method shows the Hammer proyection for the galactic distribution,
#         for all systems in the neumann catalog (not actualized), using the 
#         galactic coordinates.
#         """
#         glon, glat = self.cat_neuman["GLON"].values, self.cat_neuman["GLAT"].values

#         fig, ax = plt.subplots(figsize=(10, 7))

#         m = Basemap(projection='hammer', lon_0=0, ax=ax)

#         img = mpimg.imread('galaxia.jpg')
#         m.imshow(img, origin='upper', alpha=1)

#         x, y = m(glon, glat)
#         m.scatter(x, y, c='cyan', s=10, alpha=0.8, label='HMXBs')

#         for lat in np.arange(-75, 76, 15):
#             x_text, y_text = m(-180, lat)
#             plt.text(x_text, y_text, f'{lat}°', ha='right', va='center', fontsize=9, color='black')

#         plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.1)
#         plt.title("HMXBs on the Galactic Plane (Hammer's Projection) for Neumann Catalog, using GLON/GLAT coordinates")
#         plt.legend(loc='lower right')

#         plt.show()
        
        
#     def plot_galactic_distribution(self):
#         """
#         This method plots the galactic distribution of the neumann catalog (not actualized)
#         using error bars, with the galactic coordinates in the 2D plane. 
#         """
#         df = pd.read_excel(self.name_excel, sheet_name='HMXB_cat_Neumann')
        
#         glon = np.deg2rad(df['GLON'].values)
#         glat = np.deg2rad(df['GLAT'].values)

#         mean_dist = df['Mean_Dist'].values * u.pc
#         low_dist = df['Low_Dist'].values * u.pc
#         high_dist = df['High_Dist'].values * u.pc

#         mean_dist = mean_dist.to(u.kpc).value
#         low_dist = low_dist.to(u.kpc).value
#         high_dist = high_dist.to(u.kpc).value

#         error_sup = (high_dist - mean_dist) / 2
#         error_inf = (mean_dist - low_dist) / 2

#         error_sup = np.abs(error_sup)
#         error_inf = np.abs(error_inf)

#         x = mean_dist * np.cos(glat) * np.cos(glon)
#         y = mean_dist * np.cos(glat) * np.sin(glon)
#         x = -x
#         plane_distance = mean_dist * np.sin(glat)

#         x_err_sup = error_sup * (x / np.sqrt(x**2 + y**2))
#         y_err_sup = error_sup * (y / np.sqrt(x**2 + y**2))
#         x_err_inf = error_inf * (x / np.sqrt(x**2 + y**2))
#         y_err_inf = error_inf * (y / np.sqrt(x**2 + y**2))

#         plt.figure(figsize=(8, 8))

#         img = plt.imread('galaxia_coord.png')
#         plt.imshow(img, extent=[-20, 20, -20, 20], aspect='auto', zorder=0, alpha=0.8)

#         sc = plt.scatter(x, y, c=plane_distance, cmap="bwr", s=30, alpha=1, edgecolor='none', zorder=1)

#         cbar = plt.colorbar(sc)
#         cbar.set_label('Distance to the galactic plane (kpc)')

#         for i in range(len(x)):
#             plt.plot([x[i], x[i] + x_err_sup[i]], 
#                      [y[i], y[i] + y_err_sup[i]], 
#                      color='red', alpha=0.5, linewidth=1.)
#             plt.plot([x[i], x[i] - x_err_inf[i]], 
#                      [y[i], y[i] - y_err_inf[i]], 
#                      color='green', alpha=0.5, linewidth=1.)

#         plt.title('Galactic Distribution for Neumann Catalog, with Confidence intervals at 68%')
#         plt.xlabel('X (kpc)')
#         plt.ylabel('Y (kpc)')
#         plt.axhline(0, color='black', linewidth=0.5)
#         plt.axvline(0, color='black', linewidth=0.5)

#         plt.grid(False)
#         plt.show()
        



        
#     def plot_log_histogram_geometric_mean(self, data, bins_per_decade=8, color='blue'):
#         """
#         Draw a logarithmic histogram of the geometric mean.
#         """
#         data_filtered = data
#         data_clean = data_filtered[~np.isnan(data_filtered)]

#         print(f"Total original data points: {len(data)}")
#         print(f"Total valid data points (filtered and without NaN): {len(data_clean)}")

#         if len(data_clean) > 0:
#             min_exp = np.floor(np.log10(np.min(data_clean)))
#             max_exp = np.ceil(np.log10(np.max(data_clean)))
#             bins = np.logspace(min_exp, max_exp, int((max_exp - min_exp) * bins_per_decade))

#             n, bins, patches = plt.hist(data_clean, bins=bins, color=color, alpha=0.7)
#             plt.xscale('log')
#             plt.xlabel('Geometric Mean Flux (Min and Max Soft Flux from Neumann Catalog)')
#             plt.ylabel('Frequency')
#             plt.title('Histogram of the Geometric Mean of Soft Fluxes')

#             for patch in patches:
#                 patch.set_edgecolor('black')

#             plt.show()
#         else:
#             print("No valid data points to display the histogram.")
    
#     def flux_histogram_plot(self):
#         """
#         Method that generates the histogram for the geometric mean of Soft Flux flows.
#         """
#         self.plot_log_histogram_geometric_mean(self.cat_neuman['Geometric_Mean'], bins_per_decade=6, color='blue')
    

#     def max_distance(self, L):
#         """
#         Calculates the maximum distance at which a source is detectable based on its luminosity.
#         """
#         constant = 7.96e11
#         return np.sqrt(constant * L) * u.cm

#     def plot_luminosity_vs_max_distance(self):
#         """
#         Draw the relationship between luminosity and maximum detection distance,
#         including areas of detectable and non-detectable sources.
#         """
#         luminosities = np.logspace(30, 45, num=100)
#         distances = self.max_distance(luminosities)
#         distances_pc = distances.to(u.pc)
#         max_distance_limit = 1e10 * u.pc

#         plt.figure(figsize=(8, 6))

#         plt.plot(distances_pc.value, luminosities, label=r'$L$ vs $D_{max}$', color='blue')

#         plt.fill_betweenx(luminosities, 0, distances_pc.value, color='green', alpha=0.3, label='Detectable Sources')
#         plt.fill_betweenx(luminosities, distances_pc.value, max_distance_limit.value, color='red', alpha=0.3, label='Non-Detectable Sources')

#         plt.xscale('log')
#         plt.yscale('log')

#         plt.xlabel('Maximum Distance (pc)', fontsize=14)
#         plt.ylabel('Luminosity (erg/s)', fontsize=14)
#         plt.title('Luminosity vs Maximum Detection Distance', fontsize=16)

#         plt.legend()
#         plt.show()
        
#     def plot_flux_distributions(self):
#         """
#         Draw the distribution of maximum and minimum XRT fluxes from the Neumann 2 catalog.
#         """
#         sns.set_style("whitegrid")
        
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#         sns.histplot(self.cat_neuman_2['XRT_max_flux'], bins=30, kde=True, ax=axes[0], color='blue')
#         axes[0].set_title("Distribución de XRT_max_flux")
#         axes[0].set_xlabel("XRT_max_flux")
#         axes[0].set_ylabel("Frecuencia")

#         sns.histplot(self.cat_neuman_2['XRT_min_flux'], bins=30, kde=True, ax=axes[1], color='red')
#         axes[1].set_title("Distribución de XRT_min_flux")
#         axes[1].set_xlabel("XRT_min_flux")
#         axes[1].set_ylabel("Frecuencia")

#         plt.tight_layout()
#         plt.show()
        
#     def plot_log_histogram_luminosity(self, bins_per_decade=8, color='red'):
#         """
#         It draws a logarithm histogram of the calculated luminosity 
#         Draw a logarithmic histogram of the luminosity calculated 
#         from the XRT flux in the Neumann 2 catalog.   
#         """
#         geometric_mean_flux = (((self.cat_neuman_2['XRT_max_flux'] * self.cat_neuman_2['XRT_min_flux'])**0.5)).values * u.erg / (u.cm**2 * u.s)
#         distances = self.cat_neuman_2['Mean_Dist'].values * u.pc

#         geometric_mean_flux = np.where(geometric_mean_flux > 100000 * u.erg / (u.cm**2 * u.s), 
#                                         geometric_mean_flux * 1e-25, geometric_mean_flux)

#         luminosities = 4 * np.pi * geometric_mean_flux * (distances**2)
#         luminosities = luminosities.to(u.erg / u.s)

#         data_filtered = luminosities.value[(luminosities.value >= 1e20) & (luminosities.value <= 1e50)]
#         data_clean = data_filtered[~np.isnan(data_filtered)]

#         print(f"Total original data points: {len(luminosities.value)}")
#         print(f"Total valid data points (filtered and without NaN): {len(data_clean)}")

#         if len(data_clean) > 0:
#             min_exp = np.floor(np.log10(np.min(data_clean)))
#             max_exp = np.ceil(np.log10(np.max(data_clean)))
#             bins = np.logspace(min_exp, max_exp, int((max_exp - min_exp) * bins_per_decade))

#             n, bins, patches = plt.hist(data_clean, bins=bins, color=color, alpha=0.7)
#             plt.xscale('log')
#             plt.xlabel('Luminosity (erg/s)')
#             plt.ylabel('Frequency')
#             plt.title('Luminosity Histogram (0.3-10 keV band)')

#             for patch in patches:
#                 patch.set_edgecolor('black')
#             plt.tight_layout()
#             plt.show()
#         else:
#             print("There aren't any valid data to show.")
            
#     def graficar_distribuciones_cat_neuman(self):
#         """
#         Grafica en una misma figura la distribución (normalizada) de todas las variables numéricas
#         del catálogo Neumann usando sus nombres como etiquetas en la leyenda y también como texto
#         en la curva con su color correspondiente.
#         """
#         df_numerico = self.cat_neuman.select_dtypes(include='number').copy()

#         scaler = MinMaxScaler()
#         df_normalizado = pd.DataFrame(
#             scaler.fit_transform(df_numerico),
#             columns=df_numerico.columns
#         )

#         plt.figure(figsize=(12, 6))
#         palette = sns.color_palette('tab10', n_colors=len(df_normalizado.columns))

#         for i, column in enumerate(df_normalizado.columns):
#             data = df_normalizado[column].dropna()
#             sns.kdeplot(data, fill=True, label=column, color=palette[i], alpha=0.4, linewidth=2)

#             try:
#                 x_vals = np.linspace(data.min(), data.max(), 200)
#                 y_vals = sns.kdeplot(data).get_lines()[-1].get_data()[1]
#                 max_idx = np.argmax(y_vals)
#                 plt.text(x_vals[max_idx], y_vals[max_idx], column, color=palette[i],
#                          fontsize=9, ha='left', va='bottom', alpha=0.8)
#                 plt.gca().lines[-1].remove()
#             except Exception:
#                 continue

#         plt.title("Distribuciones normalizadas de variables numéricas en cat_neuman")
#         plt.xlabel("Valor normalizado")
#         plt.ylabel("Densidad")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
        
        
        
#     def graficar_pairplot_cat_neuman(self, hue=None, kind="scatter", diag_kind="kde"):
#         """
#         Genera un pairplot (grilla de gráficos de dispersión y KDEs) para todas las combinaciones
#         de columnas numéricas en cat_neuman.

#         Parámetros:
#         - hue: str o None, nombre de una columna para usar como clase (si aplica).
#         - kind: "scatter" o "kde", el tipo de gráfico en los cruces.
#         - diag_kind: "kde" o "hist", el tipo de gráfico en la diagonal.
#         """
#         df_numerico = self.cat_neuman.select_dtypes(include='number').copy()

#         df_filtrado = df_numerico.dropna(axis=1, thresh=5)

#         if hue and hue in self.cat_neuman.columns:
#             df_filtrado[hue] = self.cat_neuman[hue]

#         sns.pairplot(
#             data=df_filtrado,
#             hue=hue,
#             kind=kind,
#             diag_kind=diag_kind,
#             plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k'},
#             diag_kws={'shade': True}
#         )
#         plt.suptitle("Gráfico de pares para variables numéricas en cat_neuman", y=1.02)
#         plt.tight_layout()
#         plt.show()       
        
            
#     def graficar_boxplots_escalados(self):
#         """
#         Escala las variables numéricas de cat_neuman con StandardScaler y grafica sus boxplots
#         en una escala asinh (hiperbólica inversa del seno).
#         """
#         df_numerico = self.cat_neuman.select_dtypes(include='number').copy()

#         if df_numerico.empty:
#             print("No hay columnas numéricas en cat_neuman.")
#             return

#         df_numerico = df_numerico.dropna(axis=1, thresh=5)

#         if df_numerico.shape[1] == 0:
#             print("No hay columnas numéricas con suficientes datos no nulos.")
#             return

#         scaler = StandardScaler().fit(df_numerico)
#         dft = pd.DataFrame(
#             data=scaler.transform(df_numerico),
#             index=df_numerico.index,
#             columns=df_numerico.columns
#         )

#         dft_clean = dft.dropna(axis=1, how='all')

#         if dft_clean.empty:
#             print("Todos los datos fueron eliminados por NaNs. Nada que graficar.")
#             return

#         data_para_boxplot = [dft_clean[col].dropna().values for col in dft_clean.columns]

#         fig, ax = plt.subplots(figsize=(10, 1 + len(data_para_boxplot) * 0.5))
#         ax.boxplot(data_para_boxplot, vert=False, labels=dft_clean.columns, flierprops={'marker': '.'})
#         ax.set_title("Boxplots de variables numéricas escaladas (StandardScaler)")
#         ax.set_xlabel("Valor escalado (escala asinh)")
#         ax.set_xscale('asinh', linthresh=0.1)
#         ax.grid(True)
#         plt.tight_layout()
#         plt.show()

#     def graficar_test_shapiro(self, sample_size=500):
#         """
#         Evalúa la normalidad de cada variable numérica escalada (StandardScaler) usando el test de Shapiro-Wilk.
#         Grafica log10 del p-valor de cada variable.
#         """
#         df_numerico = self.cat_neuman.select_dtypes(include='number').copy()
        
#         if df_numerico.empty:
#             print("No hay columnas numéricas para analizar.")
#             return

#         scaler = StandardScaler().fit(df_numerico)
#         dft = pd.DataFrame(scaler.transform(df_numerico), columns=df_numerico.columns)

#         dft_sample = dft.sample(min(sample_size, len(dft)), random_state=42)

#         def log_shapiro(col):
#             data = col.dropna()
#             if len(data) < 3:
#                 return np.nan
#             try:
#                 return np.log10(stats.shapiro(data).pvalue)
#             except Exception:
#                 return np.nan

#         SW = dft_sample.apply(log_shapiro, axis=0).dropna().sort_values()

#         colores = ['green' if 10**val > 0.05 else 'red' for val in SW]

#         plt.figure(figsize=(10, 0.5 * len(SW) + 2))
#         bars = plt.barh(SW.index, SW.values, color=colores)
#         plt.axvline(np.log10(0.05), color='black', linestyle='--', label='p = 0.05')
#         plt.xlabel("log10(p-value) del test de Shapiro-Wilk")
#         plt.title("Normalidad de variables (log10 p-values)")
#         plt.grid(True, axis='x')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
        
            
#     def graficar_matriz_correlacion_jerarquica(self, df):
#         """
#         Calcula y grafica la matriz de correlación de las variables numéricas del dataframe,
#         y ordena la matriz utilizando agrupación jerárquica.
#         """
#         df_numerico = df.select_dtypes(include='number').copy()
        
#         if df_numerico.empty:
#             print("No hay columnas numéricas para analizar.")
#             return

#         cm = df_numerico.corr()

#         if not np.allclose(cm, cm.T):
#             print("La matriz de correlación no es simétrica. Se procederá a corregir.")
#             cm = (cm + cm.T) / 2

#         fig, ax = plt.subplots(figsize=(30, 30))
#         ax.imshow(cm, cmap='coolwarm', interpolation='none')
#         ax.set_yticks(range(df_numerico.shape[1]))
#         ax.set_yticklabels(df_numerico.columns)
#         ax.set_xticks(range(df_numerico.shape[1]))
#         ax.set_xticklabels(df_numerico.columns, rotation=90)
#         ax.set_title("Matriz de Correlación")
#         plt.colorbar(ax.imshow(cm, cmap='coolwarm', interpolation='none'))
#         plt.show()

#         distance_matrix = 1 - cm.abs()

#         if not np.allclose(distance_matrix, distance_matrix.T):
#             print("La matriz de distancia no es simétrica. Se procederá a corregir.")
#             distance_matrix = (distance_matrix + distance_matrix.T) / 2

#         linkage_matrix = sch.linkage(sch.distance.squareform(distance_matrix), method='complete')

#         dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
#         index = dendrogram['leaves']
#         sorted_correlation_matrix = cm.iloc[index, :].iloc[:, index]

#         fig, ax = plt.subplots(figsize=(30, 30))
#         ax.imshow(sorted_correlation_matrix, cmap='coolwarm', interpolation='none')
#         ax.set_yticks(range(df_numerico.shape[1]))
#         ax.set_yticklabels(np.array(df_numerico.columns)[index])
#         ax.set_xticks(range(df_numerico.shape[1]))
#         ax.set_xticklabels(np.array(df_numerico.columns)[index], rotation=90)
#         ax.set_title("Matriz de Correlación Ordenada (Agrupamiento Jerárquico)")
#         plt.colorbar(ax.imshow(sorted_correlation_matrix, cmap='coolwarm', interpolation='none'))
#         plt.show()




#     def graficar_ganancia_mutua(self):
#         exclude_cols = [col for col in self.cat_neuman.columns if 'ID' in col or 'Ref' in col or 'Name' in col or 'Comments' in col or 'Sep.' in col]

#         num_cols = self.cat_neuman.select_dtypes(include=['number']).columns.difference(exclude_cols)
#         cat_cols = self.cat_neuman.select_dtypes(exclude=['number']).columns.difference(exclude_cols)

#         num_imputer = SimpleImputer(strategy='mean')
#         cat_imputer = SimpleImputer(strategy='most_frequent')

#         self.cat_neuman[num_cols] = num_imputer.fit_transform(self.cat_neuman[num_cols])
#         self.cat_neuman[cat_cols] = cat_imputer.fit_transform(self.cat_neuman[cat_cols])

#         label_encoders = {}
#         for col in cat_cols:
#             self.cat_neuman[col] = self.cat_neuman[col].astype(str)
#             le = LabelEncoder()
#             self.cat_neuman[col] = le.fit_transform(self.cat_neuman[col])
#             label_encoders[col] = le

#         X = self.cat_neuman.drop(columns=['SpType'] + exclude_cols)
#         y = self.cat_neuman['SpType']

#         mi = mutual_info_regression(X, y)
#         f_test, _ = f_regression(X, y)

#         mi /= np.max(mi)
#         f_test /= np.max(f_test)

#         mi_f_test = list(zip(X.columns, mi, f_test))
#         mi_f_test.sort(key=lambda x: x[1], reverse=True)

#         plt.figure(figsize=(12, 8))
#         features, mi_values, f_values = zip(*mi_f_test)

#         plt.barh(features, mi_values, color='skyblue', label='Información mutua')
#         plt.barh(features, f_values, color='lightcoral', alpha=0.5, label='F-test')

#         plt.xlabel('Valor', fontsize=14)
#         plt.ylabel('Características', fontsize=14)
#         plt.title('Ganancia mutua y F-test entre las características y el SpType', fontsize=16)

#         plt.yticks(fontsize=10)
#         plt.tight_layout()
#         plt.legend()
#         plt.show()



        

#     def graficar_coordenadas_plotly(self):
#         df_filtered = self.cat_fortin[self.cat_fortin['Class'].isin(['sg', 'Be'])].copy()

#         class_map = {'sg': 0, 'Be': 1}
#         df_filtered['ClassNumeric'] = df_filtered['Class'].map(class_map)

#         num_cols = df_filtered.select_dtypes(include='number').columns
#         num_cols = [col for col in num_cols if not any(sub in col for sub in ['ID', 'Name', 'ref', 'err', 'ClassNumeric'])]
#         num_cols = num_cols[:7]

#         df_numeric = df_filtered[num_cols].copy()
#         df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
#         df_numeric.fillna(df_numeric.median(numeric_only=True), inplace=True)

#         dimensions = []
#         for col in num_cols:
#             col_min = df_numeric[col].min()
#             col_max = df_numeric[col].max()
#             dimensions.append(
#                 dict(
#                     range=[col_min, col_max],
#                     label=col,
#                     values=df_numeric[col]
#                 )
#             )

#         dimensions.append(
#             dict(
#                 tickvals=[0, 1],
#                 ticktext=['sg', 'Be'],
#                 label='Class',
#                 values=df_filtered['ClassNumeric']
#             )
#         )

#         fig = go.Figure(data=go.Parcoords(
#             line=dict(
#                 color=df_filtered['ClassNumeric'],
#                 colorscale=[[0, 'purple'], [1, 'gold']],
#                 showscale=False
#             ),
#             dimensions=dimensions
#         ))

#         fig.update_layout(
#             plot_bgcolor='white',
#             paper_bgcolor='white',
#             title='Gráfico de Coordenadas Paralelas: sg vs Be'
#         )

#         fig.show()




#     def plot_interactive_scatter(self, df=None):
#         """
#         Method for showing an interactive 3 dimensional scatter plot
#         """
#         if df is None:
#             df = self.cat_fortin

#         def plot(df, x_axe, y_axe, Size, log_x, log_y, group_alphabetically):
#             df_temp = df.copy()

#             if group_alphabetically and y_axe == 'SpType_Kim':
#                 df_temp['Spectype_grouped'] = df_temp['SpType_Kim'].apply(
#                     lambda x: 'O' if isinstance(x, str) and x.startswith('O') else ('B' if isinstance(x, str) and x.startswith('B') else x)
#                 )
#                 y_axe = 'Spectype_grouped'

#             if Size == "None":
#                 fig = px.scatter(
#                     df_temp,
#                     x=x_axe,
#                     y=y_axe,
#                     title=f"{x_axe} vs {y_axe}",
#                     labels={x_axe: x_axe, y_axe: y_axe},
#                     log_x=log_x,
#                     log_y=log_y,
#                     hover_name=df_temp.columns[0]
#                 )
#             else:
#                 clean_df = df_temp.dropna(subset=[Size])
#                 fig = px.scatter(
#                     clean_df,
#                     x=x_axe,
#                     y=y_axe,
#                     size=Size,
#                     title=f"{x_axe} vs {y_axe} (Size: {Size})",
#                     labels={x_axe: x_axe, y_axe: y_axe, Size: Size},
#                     log_x=log_x,
#                     log_y=log_y,
#                     hover_name=clean_df.columns[0]
#                 )

#             fig.show()

#         cols = df.columns.tolist()
#         numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
#         Size_options = ["None"] + numeric_cols

#         interact(
#             plot,
#             df=widgets.fixed(df),
#             x_axe=widgets.Dropdown(options=cols, value=cols[0], description="X axis:"),
#             y_axe=widgets.Dropdown(options=cols, value=cols[1], description="Y axis:"),
#             Size=widgets.Dropdown(options=Size_options, value="None", description="Size:"),
#             log_x=widgets.Checkbox(value=False, description="Log X"),
#             log_y=widgets.Checkbox(value=False, description="Log Y"),
#             group_alphabetically=widgets.Checkbox(value=False, description="Agrupar O/B")
#         )
        
#     def Venn_diagram(self, selected_columns, filtered_df):
#         """
#         In this method I show a venn diagram for the selected columns via a 
#         interactive plot.
#         """
#         sets = {col: set(filtered_df.index[filtered_df[col].notna()]) for col in selected_columns}
#         labels = venn.get_labels([sets[col] for col in selected_columns], fill=['number'])
#         fig, ax = venn.venn6(labels, names=selected_columns)
#         for text in ax.texts:
#             text.set_fontsize(8)
#         plt.title(f"Venn Diagram for {len(selected_columns)} variables")
#         plt.show()
#         print("\nIntersections of sets:")
#         for subset_label, indices in labels.items():
#             subset = set.intersection(
#                 *[sets[col] if char == '1' else set() for col, char in zip(selected_columns, subset_label)]
#             )

#     def plot_dendrogram(self, correlation_matrix):
#         """
#         Dendrogram for the positive and also negative correlation coefficients between 
#         the parameters.
#         """
#         positive_matrix = correlation_matrix[correlation_matrix > 0].fillna(0)
#         positive_distance = 1 - positive_matrix
#         positive_links = sch.linkage(ssd.squareform(positive_distance), method='ward')

#         plt.figure(figsize=(8, 4))
#         sch.dendrogram(positive_links, labels=positive_matrix.columns, leaf_rotation=45, leaf_font_size=10)
#         plt.title('Dendrogram for Positive Correlation')
#         plt.show()

#         negative_matrix = correlation_matrix[correlation_matrix < 0].fillna(0)
#         np.fill_diagonal(negative_matrix.values, 1)
#         negative_distance = 1 - np.abs(negative_matrix)
#         negative_links = sch.linkage(ssd.squareform(negative_distance), method='ward')

#         plt.figure(figsize=(8, 4))
#         sch.dendrogram(negative_links, labels=negative_matrix.columns, leaf_rotation=45, leaf_font_size=10)
#         plt.title('Dendrogram for Negative Correlation')
#         plt.show()


#     def plot_log_scale(self, filtered_df):
#         """
#         Corbet Diagram for the filtered_df, 
#         which is the result of choosing some columns
#         """
#         filtered_df = filtered_df.dropna(subset=['Period', 'Spin_period'])

#         if 'Class' not in filtered_df.columns:
#             filtered_df['Class'] = self.cat_fortin['Class']
        
#         classes = filtered_df['Class'].unique()
#         colors = plt.cm.get_cmap('tab20', len(classes))
#         markers = ['o', '^', 'x', '*', 's', 'D', 'p', 'h', 'v', '<', '>', 'X']
#         plt.figure(figsize=(8, 4))
        
#         for i, cls in enumerate(classes):
#             class_data = filtered_df[filtered_df['Class'] == cls]
#             plt.scatter(np.log10(class_data['Period']),
#                         np.log10(class_data['Spin_period']),
#                         alpha=0.7,
#                         color=colors(i),
#                         marker=markers[i % len(markers)],
#                         label=cls)

#         plt.title('Log-Log Scale: Spin Period vs Period')
#         plt.xlabel('Log(Spin Period)')
#         plt.ylabel('Log(Period)')
#         plt.legend(title="Classes")
#         plt.grid(True)
#         plt.show()

#     def correlation_matrix_all(self, selected_classes, selected_columns):
#         """The same as the previous function but for the correlation Matrix and
#         its dendogram."""
#         df = self.cat_fortin
#         if len(selected_columns) == 0:
#             print("Please select at least one numeric variable.")
#             return

#         if len(selected_columns) < 2:
#             print("Select at least two variables to generate the dendrogram.")
#             return

#         if len(selected_classes) == 0:
#             print("Please select at least one class.")
#             return
        
#         numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
#         selected_columns = [col for col in selected_columns if col in numeric_columns]
        
#         if len(selected_columns) > 0:
#             filtered_final_dataframe = df[df['Class'].isin(selected_classes)][selected_columns + ['Main_ID']]
#             filtered_complete = filtered_final_dataframe.dropna(subset=selected_columns)
#             filtered_partial = filtered_final_dataframe[~filtered_final_dataframe.index.isin(filtered_complete.index)]
#             num_complete = len(filtered_complete)
#             kendall_corr = filtered_complete[selected_columns].corr(method='kendall')

#             fig, ax = plt.subplots(figsize=(8, 4))
#             sns.heatmap(kendall_corr, annot=True, cmap="coolwarm", cbar=True, center=0, ax=ax)
#             ax.set_title(f"Correlation Matrix (Kendall) - {', '.join(selected_classes)}\n({num_complete} complete rows)")
#             plt.tight_layout()
#             plt.show()

#             print(f"{num_complete} complete rows:\n", filtered_complete[['Main_ID'] + selected_columns])

#             if 2 <= len(selected_columns) <= 6:
#                 self.Venn_diagram(selected_columns, filtered_final_dataframe)
#             else:
#                 print("Venn diagram only supports between 2 and 6 variables.")
            
#             self.plot_dendrogram(kendall_corr)
#             self.plot_log_scale(filtered_complete)
#         else:
#             print("Please select at least one numeric variable.")

#     def interactive_analysis(self):
#         """In this method we call to the method correlation matrix all."""
#         try:
#             unique_classes = self.cat_fortin['Class'].dropna().unique().tolist()
#             numeric_columns = self.cat_fortin.select_dtypes(include=[np.number]).columns.tolist()

#             interact(
#                 self.correlation_matrix_all,
#                 selected_classes=widgets.SelectMultiple(
#                     options=unique_classes,
#                     value=unique_classes[:2],
#                     description='Classes:',
#                     disabled=False
#                 ),
#                 selected_columns=widgets.SelectMultiple(
#                     options=numeric_columns,
#                     value=numeric_columns[:5],
#                     description='Parameters:',
#                     disabled=False
#                 )
#             )
#         except Exception as e:
#             print("Error in interactive setup:", e)  
            
#     def kendall_correlation_by_class(self, selected_class, param1, param2):
#         """Dendogram for some classes of the fortin catalogue."""
#         df = self.cat_fortin
#         if param1 not in df.columns or param2 not in df.columns:
#             print(f"Error: One or both selected parameters do not exist in cat_fortin.")
#             return
        
#         filtered_df = df[df["Class"] == selected_class]
#         if filtered_df.empty:
#             print(f"No data available for the selected class: {selected_class}")
#             return
        
#         selected_data = filtered_df[[param1, param2]].dropna(subset=[param1, param2])
#         if selected_data.empty:
#             print("There are no records with non-null values in both columns.")
#             return
        
#         kendall_coef, _ = stats.kendalltau(selected_data[param1], selected_data[param2])
#         print(f"Kendall correlation coefficient for class '{selected_class}' between {param1} and {param2}: {kendall_coef:.4f}")

#     def interactive_kendall(self):
#         numeric_columns = self.cat_fortin.select_dtypes(include=[np.number]).columns.tolist()
#         unique_classes = self.cat_fortin["Class"].dropna().unique().tolist()
        
#         _ = interact(
#             self.kendall_correlation_by_class,
#             selected_class=widgets.Select(options=unique_classes, description='Class:'),
#             param1=widgets.Select(options=numeric_columns, description='Parameter 1:'),
#             param2=widgets.Select(options=numeric_columns, description='Parameter 2:')
#         )  
        
#     def correlation_matrix_all(self, selected_classes, selected_columns_nan, selected_columns_non_nan):
#         """Here we obtain a dataframe with selected classes that don't have NaN values in their columns"""
#         df = self.cat_fortin

#         if len(selected_columns_nan) == 0 and len(selected_columns_non_nan) == 0:
#             print("Please select at least one column with null values or known values.")
#             return

#         if len(selected_classes) == 0:
#             print("Please select at least one class.")
#             return

#         numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
#         selected_columns_nan = [col for col in selected_columns_nan if col in numeric_columns]
#         selected_columns_non_nan = [col for col in selected_columns_non_nan if col in numeric_columns]

#         filtered_df = df[df["Class"].isin(selected_classes)]

#         missing_data = filtered_df[filtered_df[selected_columns_nan].isnull().all(axis=1)]
#         complete_data = filtered_df[filtered_df[selected_columns_non_nan].notnull().all(axis=1)]

#         if "Object" in df.columns:
#             intersection_data = pd.merge(
#                 missing_data, 
#                 complete_data, 
#                 how='inner', 
#                 on='Object',
#                 suffixes=('_nan', '_non_nan')
#             )
#         else:
#             print("No se encontró la columna 'Object' en el catálogo.")
#             return

#         if intersection_data.empty:
#             print("No systems meet both conditions (null values in some columns and complete values in others).")
#         else:
#             print("Systems with null values in selected columns and complete values in other columns:")
#             display(intersection_data)

#     def interactive_correlation_matrix(self):
#         df = self.cat_fortin
#         numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
#         unique_classes = df["Class"].dropna().unique().tolist()

#         _ = interact(
#             self.correlation_matrix_all,
#             selected_classes=widgets.SelectMultiple(
#                 options=unique_classes,
#                 value=unique_classes,
#                 description='Classes:',
#                 disabled=False
#             ),
#             selected_columns_nan=widgets.SelectMultiple(
#                 options=numeric_columns,
#                 value=[],
#                 description='NaNs:',
#                 disabled=False
#             ),
#             selected_columns_non_nan=widgets.SelectMultiple(
#                 options=numeric_columns,
#                 value=[],
#                 description='no NaNs:',
#                 disabled=False
#             )
#         )


#     def plot_missing_values_by_system(self, catalog_name, valor):
#         """
#         This method plots histograms based on how many missing (NaN) values each system has in the selected catalog.
#         It creates separate histograms for systems that have NaN in 1 column, 2 columns, 3 columns, etc.

#         Parameters:
#         catalog_name (str): The name of the catalog to select ('cat_fortin', 'cat_neuman', or 'cat_neuman_2')
#         valor (str): Whether to exclude specific columns based on missing value criteria ('si' or 'no')
#         """
#         if catalog_name == 'cat_fortin':
#             catalog = self.cat_fortin
#         elif catalog_name == 'cat_neuman':
#             catalog = self.cat_neuman
#         elif catalog_name == 'cat_neuman_2':
#             catalog = self.cat_neuman_2
#         else:
#             print("Error: Invalid catalog name. Please select 'cat_fortin', 'cat_neuman', or 'cat_neuman_2'.")
#             return

#         if valor not in ['si', 'no']:
#             print("Error: Invalid value for 'valor'. Please use 'si' or 'no'.")
#             return

#         missing_values_per_system = catalog.isna().sum(axis=1)

#         missing_values_per_system_counts = missing_values_per_system.value_counts().sort_index()

#         plt.figure(figsize=(12, 8))
#         missing_values_per_system_counts.plot(kind='bar', color='skyblue')

#         plt.title(f"Number of Systems with Missing Values in {catalog_name}")
#         plt.xlabel('Number of Columns with Missing Values')
#         plt.ylabel('Number of Systems')

#         plt.xticks(rotation=0, fontsize=10)
#         plt.tight_layout()
#         plt.show()



#     def graficar_confusion_matrix(self):
#         """
#         Método para graficar la matriz de confusión y mostrar las probabilidades de predicción
#         para la columna 'Class' en el catálogo 'cat_fortin'.
#         """
#         catalog = self.cat_fortin.copy()

#         columns_to_exclude = ['ref', 'ID', 'Name', 'err', 'var', 'Var', 'Dec', 'RA', 'Err', 'DE']
#         catalog = catalog.loc[:, ~catalog.columns.str.contains('|'.join(columns_to_exclude))]
        
#         numerical_columns = catalog.select_dtypes(include=['float64', 'int64']).columns
#         X = catalog[numerical_columns]
#         y = catalog['Class']
        
#         print(f"Cantidad de valores nulos en 'y': {y.isnull().sum()}")

#         y = y.fillna(y.mode()[0]) 

#         X = X.loc[y.index]

#         print(f"Dimensiones de X después de filtrar: {X.shape}")

#         if X.shape[0] == 0:
#             print("El conjunto de datos X está vacío. No se puede continuar.")
#             return

#         imputer = SimpleImputer(strategy='mean')
#         X_imputed = imputer.fit_transform(X)

#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X_imputed)

#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#         encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  
#         y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1)) 
#         y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))  

#         rf = RandomForestClassifier(random_state=42)
#         rf.fit(X_train, y_train_encoded)

#         y_pred_prob = rf.predict_proba(X_test)

#         y_pred = np.argmax(y_pred_prob, axis=1)

#         cm = confusion_matrix(np.argmax(y_test_encoded, axis=1), y_pred)

#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.categories_[0])
#         disp.plot(cmap='Blues')

#         plt.title('Matriz de Confusión para la predicción de la columna "Class"')
#         plt.xlabel('Predicción')
#         plt.ylabel('Real')

#         print("Probabilidades de predicción para las clases:")
#         for i, class_name in enumerate(encoder.categories_[0]):
#             print(f"\nClase: {class_name}")
#             print(f"Probabilidades para las primeras 5 muestras:")
#             print(y_pred_prob[:5, i]) 

#         plt.show()


