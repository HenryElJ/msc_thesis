# DS Toolkit
import pandas as pd, numpy as np, re, os, pickle, nltk, textstat, json
from scipy import stats
# nltk.download('gutenberg'); nltk.download('reuters'); nltk.download('shakespeare')
# nltk.download('punkt'); nltk.download('punkt_tab')

# Widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
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
from readability import Readability # pip install py-readability-metrics
import shap, holisticai, dalex, explainerdashboard #, raiwidgets

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

def read_json(filepath: str):
    with open(filepath, "r") as file:
        df = json.load(file)
    return df

# LLMs set-up
# For local running
from dotenv import load_dotenv
# Connect to LLMs
from google import genai
from mistralai import Mistral
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
# Conversational Elements
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Placeholder text
from lorem_text import lorem

load_dotenv()

api_keys = {
    "github": os.getenv('GITHUB_API_KEY'),
    "google": os.getenv('GOOGLE_PROJECT_API_KEY'),
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
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 800,
        },
        "deepseek": {
            "model": "deepseek/DeepSeek-V3-0324",
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 2048,
        },
        "llama": {
            "model": "meta/Llama-4-Scout-17B-16E-Instruct",
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 2048,
        },
        "mistral-large": {
            "model": "Mistral-large-2411",
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 2048,
        },
        "microsoft": {
            "model": "microsoft/MAI-DS-R1",
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 2048,
        }
    }

# Custom functions
def select_model(model_name: str):
    if model_name == "google":
        # https://python.langchain.com/docs/integrations/chat/google_generative_ai/
        return ChatGoogleGenerativeAI(
            model = [
                "gemini-3-flash-preview"
                , "gemini-2.5-flash"
                , "gemini-2.5-flash-lite"
                # Pro models do not work. Not part of the free tier?
                # , "gemini-3-pro-preview"
                # , "gemini-2.5-pro"
                ][2],
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
    
    elif model_name == "mistral-large":
        # https://python.langchain.com/docs/integrations/chat/mistralai/
        return ChatMistralAI(
            model = params[model_name]["model"],
            base_url = "https://models.inference.ai.azure.com", # replace with OS.GETENV[YADDA YADDA]
            mistral_api_key = api_keys["github"],
            temperature = 0,
            max_retries = 2)
    
    else:
        # https://python.langchain.com/docs/integrations/chat/azure_ai/
        return AzureAIChatCompletionsModel(
            # model_name  = params[model_name]["model"],
            model  = params[model_name]["model"],
            temperature = params[model_name]["temperature"],
            max_tokens  = params[model_name]["max_tokens"],
            max_retries = 2)
    

# Readability

def clean_text(text):
    clean = re.sub("\\s([^\\w\\s]+)", "\\1", text)
    clean = re.sub("(['-])\\s", "\\1", clean)
    return clean


def readabililty_scores(text, df = False):
    # https://readabilityformulas.com/
    # https://pypi.org/project/py-readability-metrics/
    
    # if textstat.textstat.lexicon_count(text) >= 100:
    #     r = Readability(text)

    #     scores = {
    #         "text": text,
    #         # "stats": r.statistics(),
    #         "flesch_kincaid": {
    #             "score": r.flesch_kincaid().score, 
    #             "grade_level": r.flesch_kincaid().grade_level,
    #         },
    #         "flesch_reading": {
    #             "score": r.flesch().score,
    #             "ease": r.flesch().ease, 
    #             "grade_level": r.flesch().grade_levels,
    #         },
    #         "dale_chall": {
    #             "score": r.dale_chall().score,
    #             "grade_level": r.dale_chall().grade_levels,
    #         },
    #         "ari": {
    #             "score": r.ari().score,
    #             "grade_level": r.ari().grade_levels,
    #             "ages": r.ari().ages
    #         },
    #         "coleman_liau": {
    #             "score": r.coleman_liau().score, 
    #             "grade_level": r.coleman_liau().grade_level
    #         },
    #         "gunning_fog": {
    #             "score": r.gunning_fog().score, 
    #             "grade_level": r.gunning_fog().grade_level
    #         },
    #         "smog": {
    #             "score": r.smog().score, 
    #             "grade_level": r.smog().grade_level
    #         },
    #         "spache": {
    #             "score": r.spache().score, 
    #             "grade_level": r.spache().grade_level
    #         },
    #         "linsear_write": {
    #             "score": r.linsear_write().score, 
    #             "grade_level": r.linsear_write().grade_level
    #         }
    #     }
    # else:
    scores = {
        "flesch_reading_ease":          round(textstat.flesch_reading_ease(text), 2),
        "flesch_kincaid_grade":         round(textstat.flesch_kincaid_grade(text), 2),
        "smog_index":                   round(textstat.smog_index(text), 2),
        "coleman_liau_index":           round(textstat.coleman_liau_index(text), 2),
        "automated_readability_index":  round(textstat.automated_readability_index(text), 2),
        "dale_chall_readability_score": round(textstat.dale_chall_readability_score(text), 2),
        "linsear_write_formula":        round(textstat.linsear_write_formula(text), 2),
        "gunning_fog":                  round(textstat.gunning_fog(text), 2),
        "spache_readability":           round(textstat.spache_readability(text), 2),
        # "text_standard":                textstat.text_standard(text),
        # "difficult_words":  textstat.difficult_words(text),
        }

    return pd.DataFrame(scores).drop(["text"], axis = 1).drop(["ease", "ages"], axis = 0) if df else scores


import transformers
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def sentence_similarity(human_text, llm_text, simple = False, model = "sentence-transformers/all-MiniLM-L6-v2"):
    if simple:
        # Convert the texts into TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([human_text, llm_text])
        # Calculate the cosine similarity between the vectors
        similarity = cosine_similarity(vectors)
        return round(similarity[0, 1], 2)
    else: 
        # Load the BERT/RoBERTa model
        # model = transformers.BertModel.from_pretrained('bert-base-uncased') / transformers.RobertaModel.from_pretrained('roberta-base')
        similarity_model = SentenceTransformer(model)
        # Tokenize and encode the texts
        encoding1 = similarity_model.encode(human_text, normalize_embeddings = True)
        encoding2 = similarity_model.encode(llm_text, normalize_embeddings = True)
        # Calculate the cosine similarity between the embeddings
        similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
        return round(similarity, 2)
    

def correct_dtypes(data: pd.DataFrame, data_types = dict) -> pd.DataFrame:
    # Select columns which actually appear in dataset
    
    # common_data_types = {
    #     key: data_types[key] for key in data.columns if key in data_types.keys()
    # }

    common_data_types = {
        key: value for key, value in data_types.items() if key in data.columns
    }

    # Ensure unwanted float values don't throw error when converting to int
    int_cols = [x for x, y in common_data_types.items() if y == "Int64"]
    # Check if values are numeric before applying modulo operation
    data[int_cols] = data[int_cols].apply(
        lambda x: round(x) if pd.api.types.is_numeric_dtype(x) and any(x % 1 > 0) else x
    )

    return data.astype(common_data_types)