# conda activate thesis_3.11 && cd github/msc_thesis && streamlit run interface_sandbox.py

import streamlit as st, pickle, base64, httpx
from streamlit_extras.floating_button import floating_button
from streamlit_extras.grid import grid
from streamlit_js_eval import streamlit_js_eval
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

# Conversational Chatbot
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from initialise import select_model, system_message, params, api_keys

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_selection" not in st.session_state:
    st.session_state.model_selection = None

if "responses" not in st.session_state:
    st.session_state.responses = []

# Token usage not tracked in streaming for `AzureAIChatCompletionsModel(...)`
# if ("usage_metadata" not in st.session_state):
#     st.session_state.usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

if "user_skillset" not in st.session_state:
    st.session_state.user_skillset = {}

load_dotenv()

with open("explanations_output.pickle", "rb") as file:
    explanations_output = pickle.load(file)

st.set_page_config(layout = "wide")
st.markdown('''<style>.block-container {padding-top: 0rem}</style>''', unsafe_allow_html = True)
st.markdown('''<style>.block-container {padding-bottom: 1rem}</style>''', unsafe_allow_html = True)
screen_height = streamlit_js_eval(js_expressions = "screen.height")

st.sidebar.markdown('''<center><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" />
                    <span class="material-symbols-rounded" style="font-size: 48px; color: #e3e3e3;">cognition</span></center>''', 
                    unsafe_allow_html = True)
st.sidebar.markdown("<center><h1>Conversational Explanations</h1></center>", unsafe_allow_html = True)

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
st.sidebar.markdown("<center><h5>Henry El-Jawhari, 2025</h5></center>", unsafe_allow_html = True)

user_skillset = {
    "domain": domain,
    "data_analysis": analysis_rating,
    "machine_learning": ml_rating,
    "statistics": stats_rating,
    "healthcare": healthcare_rating
    }

# This needs to also be updated in the model when it's updated in the interface
system_message = system_message(user_skillset)

# with st.popover(":sparkles: Ask AI", use_container_width = True):
# @st.dialog("Chat Support", width="large")
# def chat_dialog():
_, token_col, select_llm_col = st.columns([0.5, 0.3, 0.2], vertical_alignment = "center")

with select_llm_col:
    model_selection = st.selectbox(label = "Select the LLM you want to use:",
                                options = ("google", "chatgpt", "mistral", "mistral-large", "deepseek", "llama", "microsoft"),
                                index = 2,
                                format_func = lambda x: {"google": "Gemini", 
                                                            "chatgpt": "ChatGPT", 
                                                            "mistral": "Mistral-Small",
                                                            "mistral-large": "Mistral-Large", 
                                                            "deepseek": "DeepSeek", 
                                                            "llama": "Llama", 
                                                            "microsoft": "Microsoft (DeepSeek)"}[x])    

model = select_model(model_selection)

config = {"configurable": {"thread_id": "conversational_explainer"}}

if ("app" not in st.session_state) | (st.session_state.model_selection != model_selection) | (st.session_state.user_skillset != user_skillset):

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

    st.plotly_chart(explanations_output["roc_auc"])
    st.plotly_chart(explanations_output["confusion_matrix"])
    st.plotly_chart(explanations_output["feature_importance"])


with chatbox_col:
    chat_container = st.container(height = 600, border = True) # screen_height
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(name = message["name"], avatar =  message["avatar"]):
                st.write(message["content"])

    accepted_file_types = ["jpg", "jpeg", "png"] # ["txt", "csv", "xlsx", "pdf"]
    if query := st.chat_input("Ask me a question!", accept_file = True, file_type = accepted_file_types):
        with chat_container:

            user_avatar = ":material/cognition:"; ai_avatar = f"Images/{model_selection}.png"

            with st.chat_message(name = "user", avatar = user_avatar):
                st.write(query["text"], use_container_width = True)
                if query["files"] != []:
                    st.image(query["files"], width = 100) 
                st.session_state.messages.append({"name": "user", "avatar": user_avatar, 
                                                "content": query["text"] + f" [{query['files'][0].name}]" if query["files"] != [] else query["text"]})

        stream = True
        with chat_container:
            
            with st.chat_message(name = "assistant", avatar = ai_avatar):

                message_input = [{"type": "text", "text": query["text"]}]
                if query["files"] != []:
                    image_data = base64.b64encode(query["files"][0].read()).decode("utf-8")
                    message_input += [{"type": "image", "source_type": "base64", "mime_type": "image/jpg", "data": image_data}]

                if stream:
                    responses = st.session_state.app.stream({"messages": [HumanMessage(message_input)]}, config, stream_mode = "messages") 
                    st.write_stream(stream_output(responses))

                    st.session_state.responses = st.session_state.app.get_state(config)[0]["messages"]
                    st.session_state.messages.append({"name": model_selection, "avatar": ai_avatar, "content": st.session_state.responses[-1].content})
                    # st.session_state.usage_metadata = st.session_state.app.get_state(config)[3]["writes"]["model"]["usage_metadata"]

                else:
                    responses = st.session_state.app.invoke({"messages": [HumanMessage(message_input)]}, config,)["messages"]
                    with chat_container:            
                        st.write(responses[-1].content)

                    st.session_state.responses = st.session_state.app.get_state(config)[0]["messages"]
                    st.session_state.messages.append({"name": model_selection, "avatar": ai_avatar, "content": st.session_state.responses[-1].content})
                    # st.session_state.usage_metadata = st.session_state.app.get_state(config)[3]["writes"]["model"]["usage_metadata"]

    warning_col, download_col = st.columns([0.8, 0.2])
    warning_col.info("Large language models can make mistakes. Please verify information before you make decisions.", icon = ":material/info:")
    with download_col:

        export_data = "\n\n".join([x["name"] + ": " + (x["content"] + f"\n\n{'=' * 100}" if x["name"] != "user" else x["content"]) for x in st.session_state.messages])

        st.download_button(
            label = "Export",
            data = export_data,
            file_name = "chat.txt",
            icon = ":material/download:"
        )

# with token_col:
#     text = " | ".join([f"{x}: {y:,}" for x, y in st.session_state.usage_metadata.items()])
#     st.markdown(f'''<center><font size="2">{text}</font></center>''', unsafe_allow_html = True)

# if floating_button(":sparkles: Ask AI"):
#         chat_dialog()