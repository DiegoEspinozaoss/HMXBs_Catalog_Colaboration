#%%
import pandas as pd
import numpy as np
import re
from pathlib import Path

def create_perfect_df(sheet='kim_persistent', skiprows=174, nrows=18, base_path=None):
    if base_path is None:
        base_path = Path("../Datasets")
    excel_path = base_path / "All_four_catalogs.xlsx"

    df = pd.read_excel(excel_path, sheet_name=sheet, usecols="A", skiprows=10, nrows=76)
    column_lines = df.iloc[:, 0].tolist()

    ranges = []
    column_names = []

    for line in column_lines:
        match_range = re.match(r'\s*(\d+)\s*-\s*(\d+)', line)
        if match_range:
            start, end = match_range.groups()
            ranges.append((int(start) - 1, int(end)))

            match_name = re.match(
                r'\s*\d+\s*-\s*\d+\s+[\w\.]+\s+[\w\-/]*\s+(\S+)', 
                line
            )
            if match_name:
                column_names.append(match_name.group(1))
            else:
                column_names.append("Col")

    data_df = pd.read_excel(excel_path, sheet_name=sheet, skiprows=skiprows, nrows=nrows, header=None)

    def concat_strings(row):
        filtered = [str(x) for x in row if isinstance(x, str) and pd.notna(x)]
        return ''.join(filtered)

    concatenated_list = data_df.apply(concat_strings, axis=1).tolist()

    rows = []
    for row_str in concatenated_list:
        row = [row_str[j:k] for j, k in ranges]
        row = [val.rstrip('\n') for val in row]
        rows.append(row)

    perfect_df = pd.DataFrame(rows, columns=column_names)

    col_sign = 'DE-'
    if col_sign in perfect_df.columns:
        sign_idx = perfect_df.columns.get_loc(col_sign)
        next_idx = sign_idx + 1  

        for i, row in perfect_df.iterrows():
            val_sign = row[col_sign]
            if val_sign:
                numbers = re.findall(r'\d+', val_sign)
                if numbers:
                    num = numbers[0]
                    new_sign = re.sub(r'\d+', '', val_sign).strip()
                    perfect_df.at[i, col_sign] = new_sign

                    val_next = row.iloc[next_idx]
                    if pd.isna(val_next):
                        val_next = ''
                    else:
                        val_next = str(val_next)

                    new_val_next = num + val_next
                    perfect_df.iat[i, next_idx] = new_val_next

    return perfect_df


def calculate_skiprows_nrows(start_row, end_row):
    skiprows = start_row - 1
    nrows = end_row - start_row + 1
    return skiprows, nrows


def load_catalogs():
    """
    Load and preprocess catalogs: Neumann, Fortin, Kim (transient and persistent), and Malacaria (persistent and transient).
    Returns:
        dict: A dictionary containing the loaded datasets.
    """
    base_path = Path(__file__).resolve().parent.parent / "Datasets"
    excel_name = base_path / "All_four_catalogs.xlsx"
    excel_name_2 = base_path / "Neumann_catalog_update.xlsx"

    excel_file = pd.ExcelFile(excel_name)

    cat_neumann = excel_file.parse('HMXB_cat_Neumann')
    cat_neumann['Geometric_Mean'] = np.sqrt(cat_neumann['Max_Soft_Flux'] * cat_neumann['Min_Soft_Flux'])
    cat_neumann_update = pd.read_excel(excel_name_2, sheet_name="HMXB_cat")
    cat_fortin = excel_file.parse('v2023-09_Fortin')
    malacaria_persistent = excel_file.parse('malacaria_persistent')
    malacaria_transient = excel_file.parse('malacaria_transient')

    transient_start = 304
    transient_end = 367
    skiprows_transient, nrows_transient = calculate_skiprows_nrows(transient_start, transient_end)

    persistent_start = 175
    persistent_end = 192
    skiprows_persistent, nrows_persistent = calculate_skiprows_nrows(persistent_start, persistent_end)

    kim_persistent = create_perfect_df(sheet='kim_persistent', skiprows=skiprows_persistent, nrows=nrows_persistent, base_path=base_path)
    kim_transient = create_perfect_df(sheet='kim_transient', skiprows=skiprows_transient, nrows=nrows_transient, base_path=base_path)

    return {
        "cat_neumann": cat_neumann,
        "cat_fortin": cat_fortin,
        "cat_malacaria_persistent": malacaria_persistent,
        "cat_malacaria_transient": malacaria_transient,
        "cat_neumann_update": cat_neumann_update,
        "cat_kim_transient": kim_transient,
        "cat_kim_persistent": kim_persistent
    }