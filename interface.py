# conda activate thesis_3.11 && cd github/msc_thesis && streamlit run interface.py

import streamlit as st, pickle
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

# Conversational Chatbot
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from initialise import select_model, system_message, params

load_dotenv()

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
                st.markdown(message["content"])

    if query := st.chat_input("Ask me a question!", accept_file = False, file_type = ["txt", "csv", "xlsx", "jpg", "jpeg", "png", "pdf"]):
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