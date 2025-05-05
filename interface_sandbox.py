# conda activate thesis_3.11 && cd github/msc_thesis && streamlit run interface_sandbox.py

# https://github.com/Socvest/st-screen-stats # https://discuss.streamlit.io/t/build-responsive-apps-based-on-different-screen-features/51625
# https://github.com/Socvest/streamlit-browser-engine # https://discuss.streamlit.io/t/get-browser-stats-like-user-agent-broswer-name-chrome-firefox-ie-etc-whether-app-is-running-on-mobile-or-desktop-and-more/66735
# browser_info = browser_detection_engine()
# https://pypi.org/project/streamlit-dimensions/

import streamlit as st, pickle, base64, os, re
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

from lorem_text import lorem
from st_screen_stats import ScreenData
from streamlit_dimensions import st_dimensions
from browser_detection import browser_detection_engine

global model_selection

if "messages" not in st.session_state:
    st.session_state.messages = []

if "responses" not in st.session_state:
    st.session_state.responses = []

if "model_selection" not in st.session_state:
    st.session_state.model_selection = "mistral"

if "button_selection" not in st.session_state:
    st.session_state.button_selection = "mistral"

# Token usage not tracked in streaming for `AzureAIChatCompletionsModel(...)`
# if ("usage_metadata" not in st.session_state):
#     st.session_state.usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

if "user_skillset" not in st.session_state:
    st.session_state.user_skillset = {}

load_dotenv()

with open("explanations_output.pickle", "rb") as file:
    explanations_output = pickle.load(file)

st.set_page_config(layout = "wide")

screen_height = ScreenData().st_screen_data(key="screen_stats_")["innerHeight"]
viz_padding = 70; chat_padding = 220; response_height = screen_height - chat_padding - 105

llm_images = []
filepaths = [x for x in os.listdir("Images") if re.search(".png", x) is not None]; filepaths.sort()
for filepath in filepaths:
    with open(f"Images/{filepath}", "rb") as file:
        llm_name = filepath.replace(".png", "")
        llm_images += [[llm_name, base64.b64encode(file.read()).decode()]]

st.logo(f"Images/{st.session_state.button_selection}.png", size = "small")

st.markdown(f'''<style>
            .block-container {{padding-top: 0.2rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem; overflow: hidden}}
            .st-emotion-cache-1xgtwnd {{padding-top: 0.15; padding-bottom: 0}}
            .stChatMessage.st-emotion-cache-4oy321.ea2tk8x0:last-child {{height: {response_height}px; overflow: scroll !important; flex-direction: column}}
            .st-emotion-cache-1ir3vnm.ea2tk8x1:last-child {{margin: 0px}}
            </style>''', unsafe_allow_html = True)

st.sidebar.markdown('''
                    <div style="display: flex; align-items: center; gap: 10px;">
                    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" />
                    <span class="material-symbols-rounded" style="font-size: 48px; color: #e3e3e3;">cognition</span>
                    <h1 style="font-size: 1.5em; margin: 0; width: fit-content;">Conversational Explanations</h1>
                    </div>''', unsafe_allow_html = True)

st.sidebar.divider()

llm_name_mapping = {"google"          : "Gemini (Google)", 
                    "chatgpt"         : "ChatGPT (OpenAI)", 
                    "mistral"         : "Mistral - Small",
                    "mistral-large"   : "Mistral - Large", 
                    "deepseek"        : "DeepSeek", 
                    "llama"           : "Llama (Meta)", 
                    "microsoft"       : "Microsoft (DeepSeek)"}

button = True
if button:
    def button_selection(name):
        st.session_state.button_selection = name

    with st.sidebar.expander("Select the LLM to use:"):
        llm_1, llm_2, llm_3, _ = st.columns(4)
        llm_4, llm_5, llm_6, llm_7 = st.columns(4)
        st.markdown("""<style> .st-emotion-cache-c1nttv.eacrzsi2:focus {background-color: #000000; border-color: #FF4B4B} </style>""", unsafe_allow_html = True)
        llm_1.button(f"![](data:image/png;base64,{llm_images[0][1]})", on_click = button_selection, help = llm_name_mapping[llm_images[0][0]], args = [llm_images[0][0]])
        llm_2.button(f"![](data:image/png;base64,{llm_images[1][1]})", on_click = button_selection, help = llm_name_mapping[llm_images[1][0]], args = [llm_images[1][0]])
        llm_3.button(f"![](data:image/png;base64,{llm_images[2][1]})", on_click = button_selection, help = llm_name_mapping[llm_images[2][0]], args = [llm_images[2][0]])
        llm_4.button(f"![](data:image/png;base64,{llm_images[3][1]})", on_click = button_selection, help = llm_name_mapping[llm_images[3][0]], args = [llm_images[3][0]])
        llm_5.button(f"![](data:image/png;base64,{llm_images[4][1]})", on_click = button_selection, help = llm_name_mapping[llm_images[4][0]], args = [llm_images[4][0]])
        llm_6.button(f"![](data:image/png;base64,{llm_images[5][1]})", on_click = button_selection, help = llm_name_mapping[llm_images[5][0]], args = [llm_images[5][0]])
        llm_7.button(f"![](data:image/png;base64,{llm_images[6][1]})", on_click = button_selection, help = llm_name_mapping[llm_images[6][0]], args = [llm_images[6][0]])

        st.markdown("<h5>You can change this model anytime.</h5>", unsafe_allow_html = True)

        # with stylable_container("green", css_styles = """button {background-color: #00FF00; color: black; }""",):    
        #     test_button = st.button("Test", type = "primary")

        model_selection = st.session_state.button_selection
else:
    model_selection = st.sidebar.selectbox(
        label = "Select the LLM you want to use:",
        options = ("google", "chatgpt", "mistral", "mistral-large", "deepseek", "llama", "microsoft"),
        index = 2,
        format_func = lambda x: llm_name_mapping[x])

st.sidebar.divider()

domain = st.sidebar.selectbox("What is your area of expertise/domain?", ["Data Science", "Healthcare", "Other"])
if domain == "Other":
    domain = st.sidebar.text_input("Please enter your domain:")

st.sidebar.divider()
st.sidebar.write(f"Please rate your skillset level:")
slider_options      = ["None", "Limited", "Moderate", "Good", "Excellent"]
analysis_rating     = st.sidebar.select_slider(label = "Data Analysis",     options = slider_options, value = "Moderate")
ml_rating           = st.sidebar.select_slider(label = "Machine Learning",  options = slider_options, value = "Moderate")
stats_rating        = st.sidebar.select_slider(label = "Statistics",        options = slider_options, value = "Moderate")
healthcare_rating   = st.sidebar.select_slider(label = "Healthcare",        options = slider_options, value = "Moderate")

st.sidebar.divider()
st.sidebar.markdown("<center><h5>Henry El-Jawhari, 2025</h5></center>", unsafe_allow_html = True)

user_skillset = {
    "domain"            : domain,
    "data_analysis"     : analysis_rating,
    "machine_learning"  : ml_rating,
    "statistics"        : stats_rating,
    "healthcare"        : healthcare_rating
    }

# This needs to also be updated in the model when it's updated in the interface
system_message = system_message(user_skillset); system_message = "answer shortly"

model = select_model(model_selection)

config = {"configurable": {"thread_id": "conversational_explainer"}}

# On initial start-up, model or user skillset changes
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

viz_col, chatbox_col = st.columns([0.4, 0.6])

with viz_col:
    viz_container = st.container(height = screen_height - viz_padding, border = False)
    with viz_container:
        st.plotly_chart(explanations_output["roc_auc"])
        st.plotly_chart(explanations_output["confusion_matrix"])
        st.plotly_chart(explanations_output["feature_importance"])

with chatbox_col:
    chat_container = st.container(height = screen_height - chat_padding)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(name = message["name"], avatar =  message["avatar"]):
                st.write(message["content"])

    accepted_file_types = ["jpg", "jpeg", "png"] # ["txt", "csv", "xlsx", "pdf"]
    if query := st.chat_input("Ask me a question!", accept_file = True, file_type = accepted_file_types):
        with chat_container:

            user_avatar = ":material/cognition:"; ai_avatar = f"Images/{model_selection}.png"

            with st.chat_message(name = "user", avatar = user_avatar):
                st.write(query["text"])
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
                    st.write(stream_output(responses))

                    st.session_state.responses = st.session_state.app.get_state(config)[0]["messages"]
                    st.session_state.messages.append({"name": model_selection, "avatar": ai_avatar, "content": st.session_state.responses[-1].content})
                    # st.session_state.usage_metadata = st.session_state.app.get_state(config)[3]["writes"]["model"]["usage_metadata"]

                    # Lorem ipsum test
                    # responses = lorem.paragraphs(10)
                    # st.write(responses)
                    # st.session_state.responses = responses
                    # st.session_state.messages.append({"name": model_selection, "avatar": ai_avatar, "content": st.session_state.responses})

                else:
                    responses = st.session_state.app.invoke({"messages": [HumanMessage(message_input)]}, config,)["messages"]     
                    st.write(responses[-1].content)

                    st.session_state.responses = st.session_state.app.get_state(config)[0]["messages"]
                    st.session_state.messages.append({"name": model_selection, "avatar": ai_avatar, "content": st.session_state.responses[-1].content})
                    # st.session_state.usage_metadata = st.session_state.app.get_state(config)[3]["writes"]["model"]["usage_metadata"]
                
                # st.markdown('''<style>.stChatMessage.st-emotion-cache-4oy321.ea2tk8x0:last-child {overflow: visible !important}</style>''', unsafe_allow_html = True)
    
    if st.session_state.messages == []:
        st.info("Large language models can make mistakes. Please verify information before decisions.", icon = ":material/info:")
    else:
        warning_col, download_col = st.columns([0.8, 0.2], vertical_alignment = "center")
        warning_col.info("Large language models can make mistakes. Please verify information before decisions.", icon = ":material/info:")
        
        export_data = "\n\n".join([x["name"] + ": " + (x["content"] + f"\n\n{'=' * 100}" if x["name"] != "user" else x["content"]) for x in st.session_state.messages])
        download_col.download_button(
            label = "Export chat",
            data = export_data,
            file_name = "chat.txt",
            icon = ":material/download:")

# with token_col:
#     text = " | ".join([f"{x}: {y:,}" for x, y in st.session_state.usage_metadata.items()])
#     st.markdown(f'''<center><font size="2">{text}</font></center>''', unsafe_allow_html = True)

# if floating_button(":sparkles: Ask AI"):
#         chat_dialog()

# Custom CSS code to make scrollbar transparent
transparent_scrollbar_style = """
<style>
/* For WebKit browsers (Chrome, Safari) */
::-webkit-scrollbar {
    width: 10px; /* Width of the scrollbar */
}

::-webkit-scrollbar-track {
    background: transparent; /* Transparent track */
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.3); /* Semi-transparent thumb */
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.5); /* Slightly darker transparent thumb on hover */
}

/* For Firefox */
* {
    scrollbar-width: thin;
    scrollbar-color: rgba(0, 0, 0, 0.3) transparent; /* Transparent track with semi-transparent thumb */
}
"""
st.markdown(transparent_scrollbar_style, unsafe_allow_html = True)