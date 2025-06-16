# DS Toolkit
import pandas as pd, numpy as np, os, pickle

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Visualisation Tools
import plotly.express as px, plotly.io as pio, plotly.graph_objects as go, altair as alt, matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Plotly defaults
px.defaults.template = "plotly"
# px.defaults.color_continuous_scale = px.colors.sequential.Blackbody
px.defaults.width = 800; px.defaults.height = 500

# Explanations Toolkit
from sklearn import metrics
import shap, holisticai, dalex, explainerdashboard, raiwidgets

def prettyDescribe(data: pd.DataFrame, data_type = np.number, shape = True):
    n = data.shape[0]; d = data.shape[1]
    print(f"{n:,} observations | {d} features") if shape else next
    if data_type == np.number:
        return (data
                .describe(include = [data_type])
                .T
                .assign(na_values = lambda x: x["count"].apply(lambda x: f'{n - x:,.0f} ({100 * (n - x) / n:.2f}%)'))
                .round(2)
                [["min", "mean", "max", "std", "25%", "50%", "75%", "na_values"]]
                )
    elif (data_type == "O") | (data_type == "category"):
        return (data
                .describe(include = [data_type])
                .T
                .assign(na_values = lambda x: x["count"].apply(lambda x: f'{n - x:,.0f} ({100 * (n - x) / n:.2f}%)'),
                        top_freq = lambda x: x["freq"].apply(lambda x: f'{n - x:,.0f} ({100 * (n - x) / n:.2f}%)'))
                .round(2)
                [["unique", "top", "top_freq", "na_values"]]
                )


# Import all files
def to_pkl(data, filename: str):
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol = pickle.HIGHEST_PROTOCOL)


def read_pkl(filepath: str):
    with open(filepath, "rb") as file:
        df = pickle.load(file)
    return df