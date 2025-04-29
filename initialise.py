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

# Langchain
from langchain import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
# Conversational Chatbot
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


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

params = {
        "chatgpt": {
            "model": "openai/gpt-4.1",
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 800,
        },
        "deepseek": {
            "model": "deepseek/DeepSeek-V3-0324",
            "temperature": 0.8,
            "top_p": 0.1,
            "max_tokens": 2048,
        },
        "llama": {
            "model": "meta/Llama-4-Scout-17B-16E-Instruct",
            "temperature": 0.8,
            "top_p": 0.1,
            "max_tokens": 2048,
        },
        "mistral-large": {
            "model": "Mistral-large-2411",
            "temperature": 0.8,
            "top_p": 0.1,
            "max_tokens": 2048,
        },
        "microsoft": {
            "model": "microsoft/MAI-DS-R1",
            "temperature": None,
            "top_p": None,
            "max_tokens": 2048,
        }
    }

# Custom functions
def select_model(model_name: str):
    if model_name == "google":
        # https://python.langchain.com/docs/integrations/chat/google_generative_ai/
        return ChatGoogleGenerativeAI(
            model = "gemini-2.0-flash",
            temperature = 0,
            max_tokens = None,
            timeout = None,
            max_retries = 2)
        
    elif model_name == "mistral":
        # https://python.langchain.com/docs/integrations/chat/mistralai/
        return ChatMistralAI(
            model = "mistral-small-latest",
            temperature = 0,
            max_retries = 2)
        
    else:
        # https://python.langchain.com/docs/integrations/chat/azure_ai/
        return AzureAIChatCompletionsModel(
            model_name  = params[model_name]["model"],
            temperature = params[model_name]["temperature"],
            max_tokens  = params[model_name]["max_tokens"],
            max_retries = 2)


# System message
def system_message(user: dict):
    return f'''
    You are part of an interface helping to guide human users through explanations for an artificial intelligence system. 

    This system generates binary predictions on whether someone gets the covid vaccine or not.

    The user works in the field of {user["domain"]}. Rating their skillset on a scale of ["None", "Limited", "Moderate", "Good", "Excellent"]:
    Data analysis: {user["data_analysis"]}. Machine learning: {user["machine_learning"]}. Statistics: {user["statistics"]}. Healthcare: {user["healthcare"]}.

    Provide comprehensive, but concise text-based explanations. Do not provide or recommend any code, unless explicity asked.

    Your explanations should cater to their domain and skillset level, and be relevant to the artificial intelligence system. 

    Only answer questions relating to questions about the artificial intelligence system.

    Always recommend follow-up, clarifying questions the user could ask to help aid their understanding. 
    '''


def prettyDescribe(data):
    return (data
            .describe()
            .drop("count", axis = 0)
            .round(2)
            .T
            [["min", "mean", "max", "std", "25%", "50%", "75%"]]
            )