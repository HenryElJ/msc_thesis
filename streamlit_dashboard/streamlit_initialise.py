# https://github.com/Socvest/st-screen-stats # https://discuss.streamlit.io/t/build-responsive-apps-based-on-different-screen-features/51625
# https://github.com/Socvest/streamlit-browser-engine # https://discuss.streamlit.io/t/get-browser-stats-like-user-agent-broswer-name-chrome-firefox-ie-etc-whether-app-is-running-on-mobile-or-desktop-and-more/66735
# browser_info = browser_detection_engine()
# https://pypi.org/project/streamlit-dimensions/

# Basic modules
import streamlit as st, pickle, base64, os, re
# Streamlit hacks
from streamlit_extras.floating_button import floating_button
from st_screen_stats import ScreenData
import streamlit.components.v1 as components
# from st_pages import get_pages, get_script_run_ctx 
from streamlit_javascript import st_javascript
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
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Placeholder text
from lorem_text import lorem

load_dotenv()

# api_keys = {
#     "github": os.getenv('GITHUB_API_KEY'),
#     "google": os.getenv('GOOGLE_API_KEY'),
#     "mistral": os.getenv('MISTRAL_API_KEY')
#     }

# clients = {
#     "github": ChatCompletionsClient(endpoint = "https://models.github.ai/inference", credential = AzureKeyCredential(api_keys["github"])),
#     "google": genai.Client(api_key = api_keys["google"]),
#     "mistral": Mistral(api_key = api_keys["mistral"])
# }

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
            model_name  = params[model_name]["model"],
            temperature = params[model_name]["temperature"],
            max_tokens  = params[model_name]["max_tokens"],
            max_retries = 2)


def stream_output(stream):
    for chunk, _ in stream:
        yield chunk.content


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


def read_img(file):
    return base64.b64encode(file.read()).decode()


def generate_tab(tab):
    return tab.replace(";\n", ";")


def add_chatbox_col(tab):
    return f'''with {tab}_ai:
        \tchat_container = st.container(height = screen_height - chat_padding, key = "{tab}_container");
    
        \twith chat_container:
            \t\tfor message in st.session_state.messages:
                \t\t\twith st.chat_message(name = message["name"], avatar =  message["avatar"]): 
                    \t\t\t\tst.write(message["content"]);
    
        \tif query := st.chat_input("Ask me a question!", accept_file = True, key = "{tab}_chat_input"):
            \t\tif query["text"] is None:
                \t\t\tpass
            \t\telse:
                \t\t\twith chat_container:
                    \t\t\t\twith st.chat_message(name = "user", avatar = user_avatar):
                        \t\t\t\t\tst.write(query["text"]); st.session_state.messages.append({{"name": "user", "avatar": user_avatar, "content": query["text"]}})
                    \t\t\t\twith st.chat_message(name = "assistant", avatar = ai_avatar):
                        # \t\t\t\t\t\tmessage_input = [{{"type": "text", "text": query["text"]}}]; responses = lorem.paragraphs(10); st.write(responses); st.session_state.responses = responses; st.session_state.messages.append({{"name": model_selection, "avatar": ai_avatar, "content": st.session_state.responses}});
                        \t\t\t\t\t\tmessage_input = [{{"type": "text", "text": query["text"]}}]; responses = st.session_state.app.stream({{"messages": [HumanMessage(message_input)]}}, config, stream_mode = "messages"); st.write(stream_output(responses)); st.session_state.responses = st.session_state.app.get_state(config)[0]["messages"]; st.session_state.messages.append({{"name": model_selection, "avatar": ai_avatar, "content": st.session_state.responses[-1].content}})
        
        \tst.info("Large language models can make mistakes. Please verify information before decisions.", icon = ":material/info:")'''


# Icons of LLMs
llm_name_mapping = {"google"          : "Gemini (Google)", 
                    "chatgpt"         : "ChatGPT (OpenAI)", 
                    "mistral"         : "Mistral - Small",
                    "mistral-large"   : "Mistral - Large", 
                    "deepseek"        : "DeepSeek", 
                    "llama"           : "Llama (Meta)", 
                    "microsoft"       : "Microsoft (DeepSeek)"}

llm_images = []
filepaths = [x for x in os.listdir("images/llm_icons") if re.search(".png", x) is not None]; filepaths.sort()
for filepath in filepaths:
    with open(f"images/llm_icons/{filepath}", "rb") as file:
        llm_name = filepath.replace(".png", "")
        llm_images += [[llm_name, read_img(file)]]

sidebar_setup = '''

'''