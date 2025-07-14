import pandas as pd
import numpy as np
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