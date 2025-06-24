
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