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
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import SystemMessage

from initialise import params

st.set_page_config(layout = "wide")

load_dotenv()

user = {
    "domain": "data science",
    "machine_learning": 6,
    "statistics": 6,
    "healthcare": 1,
}

system_message = f'''
You are part of an interface helping to guide human users through explanations for an artificial intelligence system. 

This system generates binary predictions on whether someone gets the covid vaccine or not.

The user works in the field of {user["domain"]}. Rating their skillset out of 10:

Machine learning: {user["machine_learning"]}. Statistics: {user["statistics"]}. Healthcare: {user["healthcare"]}.

Provide comprehensive, but concise text-based explanations. Do not provide or recommend any code, unless explicity asked.

Your explanations should cater to their domain and skillset level, and be relevant to the artificial intelligence system. 

Only answer questions relating to questions about the artificial intelligence system.

Always recommend follow-up, clarifying questions the user could ask to help aid their understanding. 
'''

system_message = "Answer shortly"

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

col1, _, _ = st.columns([0.2, (1 - 0.2) / 2, (1 - 0.2) / 2])
with col1:
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


# def stream_output(stream):
#     for chunk, _ in stream:
#         yield chunk.content

for message in st.session_state.messages:
    with st.chat_message(name = message["name"], avatar =  message["avatar"]):
        st.write(message["content"])

if query := st.chat_input("Ask me a question!"):
    user_avatar = ":material/cognition:"; ai_avatar = f"Images/{model_selection}.png"
    
    with st.chat_message(name = "user", avatar = user_avatar):
        st.session_state.messages.append({"name": "user", "avatar": user_avatar, "content": query})
        st.write(query)

    with st.chat_message(name = "assistant", avatar = ai_avatar):
        # responses = st.session_state.app.stream({"messages": [HumanMessage(query)]}, config, stream_mode = "messages")
        # response = st.write_stream(stream_output(responses))

        responses = st.session_state.app.invoke({"messages": [HumanMessage(query)]}, config,)["messages"]
        st.session_state.responses = responses
        st.session_state.messages.append({"name": "assistant", "avatar": ai_avatar, "content": responses[-1].content})
        st.write(responses[-1].content)

####