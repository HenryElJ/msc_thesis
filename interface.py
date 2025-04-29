# conda activate thesis_3.11 && cd github/msc_thesis && streamlit run interface.py

import streamlit as st, pickle
# from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

# Conversational Chatbot
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from initialise import select_model, system_message, params

# load_dotenv()

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

with open("explanations_output.pickle", "rb") as file:
    explanations_output = pickle.load(file)

st.set_page_config(layout = "wide")

st.sidebar.markdown('''<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" />
                    <span class="material-symbols-rounded" style="font-size: 48px; color: #e3e3e3;">cognition</span>''', 
                    unsafe_allow_html = True)
st.sidebar.write("# Conversational Explanations")
st.sidebar.divider()
domain = st.sidebar.selectbox("What is your area of expertise/domain?", ["Data Science", "Healthcare", "Other"])
if domain == "Other":
    domain = st.sidebar.text_input("Please enter your domain:")
st.sidebar.divider()
st.sidebar.write(f"Please rate your skillset level:")
slider_options = ["None", "Limited", "Moderate", "Good", "Excellent"]
analysis_rating = st.sidebar.select_slider(label = "Data Analysis", options = slider_options, value = "Moderate")
ml_rating = st.sidebar.select_slider(label = "Machine Learning", options = slider_options, value = "Moderate")
stats_rating = st.sidebar.select_slider(label = "Statistics", options = slider_options, value = "Moderate")
healthcare_rating = st.sidebar.select_slider(label = "Healthcare", options = slider_options, value = "Moderate")
st.sidebar.divider()

# This needs to also be updated in the model when it's updated in the interface
system_message = system_message(
    {
        "domain": domain,
        "data_analysis": analysis_rating,
        "machine_learning": ml_rating,
        "statistics": stats_rating,
        "healthcare": healthcare_rating
    }
    )

selectbox_buffer = 0.2
_, selectbox_col = st.columns([1 - selectbox_buffer, selectbox_buffer])

with selectbox_col:
    model_selection = st.selectbox(label = "Select the LLM you want to use:",
                               options = ("google", "chatgpt", "mistral", "mistral-large", "deepseek", "llama", "microsoft"),
                               format_func = lambda x: {"google": "Gemini", 
                                                        "chatgpt": "ChatGPT", 
                                                        "mistral": "Mistral-Small",
                                                        "mistral-large": "Mistral-Large", 
                                                        "deepseek": "DeepSeek", 
                                                        "llama": "Llama", 
                                                        "microsoft": "Microsoft (DeepSeek)"}[x])    

model = select_model(model_selection)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_selection" not in st.session_state:
    st.session_state.model_selection = None

if "responses" not in st.session_state:
    st.session_state.responses = []

config = {"configurable": {"thread_id": "conversation_1"}}

if ("app" not in st.session_state) | (st.session_state.model_selection != model_selection):

    st.session_state.model_selection = model_selection

    prompt_template = ChatPromptTemplate(
        [
        ("system", system_message),
        MessagesPlaceholder(variable_name = "messages")
        ])

    def call_model(state: MessagesState):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow = StateGraph(state_schema = MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    memory = MemorySaver()
    
    st.session_state.app = workflow.compile(checkpointer = memory)
    _ = st.session_state.app.update_state(config, {"messages": st.session_state.responses})


def stream_output(stream):
    for chunk, _ in stream:
        yield chunk.content


chatbox_buffer = 0.5
viz_col, chatbox_col = st.columns([chatbox_buffer, 1 - chatbox_buffer])

with viz_col:
    with st.container(height = 750):
        st.plotly_chart(explanations_output["confusion_matrix"])
        st.plotly_chart(explanations_output["feature_importance"])
        st.plotly_chart(explanations_output["roc_auc"])

with chatbox_col:
    chat_container = st.container(height = 600, border = True)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(name = message["name"], avatar =  message["avatar"]):
                st.write(message["content"])

    if query := st.chat_input("Ask me a question!", accept_file = False, file_type = ["txt", "csv", "xlsx", "jpg", "jpeg", "png", "pdf"]):
        
        # Hacky just for proof of concept
        # if re.search("confusion matrix", query.lower()) is not None:
        #     query += explanations_output["confusion_matrix"].data[0].to_json()
        # elif re.search("feature importance", query.lower()) is not None:
        #     query += explanations_output["feature_importance"].data[0].to_json()
        # elif re.search("roc auc", query.lower()) is not None:
        #     query += explanations_output["roc_auc"].data[0].to_json()


        user_avatar = ":material/cognition:"; ai_avatar = f"Images/{model_selection}.png"
        
        with chat_container:
            with st.chat_message(name = "user", avatar = user_avatar):
                st.session_state.messages.append({"name": "user", "avatar": user_avatar, "content": query})
                st.write(query)

        stream = True
        with chat_container:
            with st.chat_message(name = "assistant", avatar = ai_avatar):
                if stream:
                    responses = st.session_state.app.stream({"messages": [HumanMessage(query)]}, config, stream_mode = "messages")    
                    st.write_stream(stream_output(responses))

                    st.session_state.responses = st.session_state.app.get_state(config)[0]["messages"]
                    st.session_state.messages.append({"name": "assistant", "avatar": ai_avatar, "content": st.session_state.responses[-1].content})

                else:
                    responses = st.session_state.app.invoke({"messages": [HumanMessage(query)]}, config,)["messages"]
                    with chat_container:            
                        st.write(responses[-1].content)

                    st.session_state.responses = st.session_state.app.get_state(config)[0]["messages"]
                    st.session_state.messages.append({"name": "assistant", "avatar": ai_avatar, "content": st.session_state.responses[-1].content})

    st.info("Large language models can make mistakes. Please verify information for your decisions.", icon = ":material/info:")


####
