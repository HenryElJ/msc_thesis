# conda activate thesis_3.11 && cd github/msc_thesis && streamlit run interface.py

import streamlit as st
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

from initialise import params

st.set_page_config(layout = "wide")

load_dotenv()

system_message = "Answer in 30 words"

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

selectbox_buffer = 0.2
selectbox_col, _ = st.columns([selectbox_buffer, 1 - selectbox_buffer])

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
chatbox_col, _ = st.columns([chatbox_buffer, 1 - chatbox_buffer])

with chatbox_col:

    chat_container = st.container(height = 600, border = True)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(name = message["name"], avatar =  message["avatar"]):
                st.markdown(message["content"])

    if query := st.chat_input("Ask me a question!"):
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

####