from Load_Data import load_catalogs
import pandas as pd
import numpy as np


def get_id_name_columns(df):
    return [col for col in df.columns if ('id' in col.lower()) or ('name' in col.lower())]

def Joined_Catalogs():
    catalogs = load_catalogs()
    cat_fortin = catalogs['cat_fortin']
    cat_neuman = catalogs['cat_neumann']    
    fortin_id_cols = get_id_name_columns(cat_fortin)
    neuman_id_cols = get_id_name_columns(cat_neuman)

    cat_neuman_norm = cat_neuman.copy()
    for col in neuman_id_cols:
        cat_neuman_norm[col] = cat_neuman_norm[col].astype(str).str.upper().str.replace(' ', '')

    matched_rows_fortin = []
    matched_rows_neuman = []
    matched_info = []

    for idx, row in cat_fortin.iterrows():
        found_match = False
        for col_f in fortin_id_cols:
            val_raw = row[col_f]
            if pd.isna(val_raw) or str(val_raw).strip() == '':
                continue

            val = str(val_raw).upper().replace(' ', '')
            for col_n in neuman_id_cols:
                matched = cat_neuman_norm[cat_neuman_norm[col_n] == val]
                if not matched.empty:
                    matched_rows_fortin.append(row)
                    matched_rows_neuman.append(cat_neuman.loc[matched.index[0]])
                    matched_info.append({
                        'matched_value': val,
                        'fortin_column': col_f,
                        'neuman_column': col_n
                    })
                    found_match = True
                    break
            if found_match:
                break

    df_fortin = pd.DataFrame(matched_rows_fortin).reset_index(drop=True)
    df_neuman = pd.DataFrame(matched_rows_neuman).reset_index(drop=True)
    df_columns = pd.DataFrame(matched_info)

    common_cols = df_fortin.columns.intersection(df_neuman.columns)
    conflict_cols = [col for col in common_cols if not df_fortin[col].equals(df_neuman[col])]

    df_merged = pd.concat(
        [df_fortin, df_neuman],
        axis=1,
        keys=['fortin', 'neuman']
    )

    new_columns = []
    for (source, colname) in df_merged.columns:
        if colname in conflict_cols:
            new_colname = f"{colname}_fortin" if source == "fortin" else f"{colname}_neuman"
        else:
            new_colname = colname
        new_columns.append(new_colname)
    df_merged.columns = new_columns
    final_dataframe = df_merged.copy()

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

    return final_dataframe, df_columns

#EXAMPLE OF USE: matched_by_all_ids, df_columns = Joined_Catalogs()