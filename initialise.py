# DS Toolkit
import pandas as pd, numpy as np, streamlit as st, pickle

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Visualisation Tools
import plotly.express as px, plotly.io as pio, altair as alt, matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Plotly defaults
px.defaults.template = "plotly"
# px.defaults.color_continuous_scale = px.colors.sequential.Blackbody
px.defaults.width = 800; px.defaults.height = 500

# Explanations Toolkit
import holisticai, dalex, explainerdashboard, raiwidgets


# LLM APIs
import os
from dotenv import load_dotenv

from google import genai
from mistralai import Mistral

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from langchain import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain.chains import ConversationChain

# API Keys
load_dotenv()

api_keys = {
    "github": os.getenv('GITHUB_API_KEY'),
    "google": os.getenv('GOOGLE_API_KEY'),
    "mistral": os.getenv('MISTRAL_API_KEY')
    }

clients = {
    "github": ChatCompletionsClient(endpoint = "https://models.github.ai/inference", credential = AzureKeyCredential(api_keys["github"])),
    "google": genai.Client(api_key = api_keys["google"]),
    "mistral": Mistral(api_key = api_keys["mistral"])
}

# Custom functions
def prettyDescribe(data):
    return (data
            .describe()
            .drop("count", axis = 0)
            .round(2)
            .T
            [["min", "mean", "max", "std", "25%", "50%", "75%"]]
            )

# Import data
df = pd.read_excel("Data/publichealth_v10i1e47979_app2.xlsx")