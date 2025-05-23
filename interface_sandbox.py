# conda activate thesis_3.11 && cd github/msc_thesis && streamlit run interface_sandbox.py

# Custom imports
from streamlit_initialise import *
from tabs import *

# Import all files
with open("explanations_output.pickle", "rb") as file:
    explanations_output = pickle.load(file)

# Initialise session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "responses" not in st.session_state:
    st.session_state.responses = []

if "model_selection" not in st.session_state:
    st.session_state.model_selection = "mistral"

if "button_selection" not in st.session_state:
    st.session_state.button_selection = "mistral"

if "user_skillset" not in st.session_state:
    st.session_state.user_skillset = {}

if "tab" not in st.session_state:
    st.session_state.tab = None

if "ask_ai" not in st.session_state:
    # [introduction, data, model]
    st.session_state.ask_ai = [False, False, False]

# Icons of LLMs
llm_images = []
filepaths = [x for x in os.listdir("llm_icons") if re.search(".png", x) is not None]; filepaths.sort()
for filepath in filepaths:
    with open(f"llm_icons/{filepath}", "rb") as file:
        llm_name = filepath.replace(".png", "")
        llm_images += [[llm_name, base64.b64encode(file.read()).decode()]]

st.logo(f"llm_icons/{st.session_state.button_selection}.png", size = "small")

user_avatar = ":material/cognition:"; ai_avatar = f"llm_icons/{st.session_state.model_selection}.png"

# Page configurations
st.set_page_config(layout = "wide")

screen_height = ScreenData().st_screen_data(key="screen_stats_")["innerHeight"]
dict_padding = 220; viz_padding = 130; chat_padding = 270; response_height = screen_height - chat_padding - 105

# .block-container{{overflow: hidden}}
st.markdown(f'''<style>
            /* Overall screen padding */
            .block-container {{padding-top: 0rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem}}
            /* Sidebar padding*/
            .st-emotion-cache-1xgtwnd {{padding-top: 1rem; padding-bottom: 0rem}}
            /* Last message of chatbox */
            .stChatMessage.st-emotion-cache-4oy321.ea2tk8x0:last-child {{height: {response_height}px; overflow: scroll !important; flex-direction: column}}
            /**/
            .st-emotion-cache-1ir3vnm.ea2tk8x1:last-child {{margin: 0px}}
            /* Tab height */
            .st-al.st-as.st-bh.st-bd.st-fh.st-fi.st-fj.st-fk.st-fl.st-fm.st-fn.st-fo.st-fp {{height: 2rem}}
            /* Tab body padding */
            .st-bd.st-c0.st-dv.st-c8.st-c6.st-c7 {{padding-top: 0rem}}
            /* Floating button */
            .st-emotion-cache-i2li6s.eacrzsi1 {{background-image: linear-gradient(to bottom right, red, yellow); border-color: gold}} /*blue, purple*/
            .st-emotion-cache-i2li6s.eacrzsi1:hover {{background-image: linear-gradient(to bottom right, yellow, red); border-color: gold}}
            </style>''', unsafe_allow_html = True)

# .st-emotion-cache-8atqhb.e1mlolmg0 {{height: 0}}

# Sidebar configurations
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
st.sidebar.markdown("<center><h5>Conversational Explanations ©, 2025</h5></center>", unsafe_allow_html = True)

user_skillset = {
    "domain"            : domain,
    "data_analysis"     : analysis_rating,
    "machine_learning"  : ml_rating,
    "statistics"        : stats_rating,
    "healthcare"        : healthcare_rating
    }

# Model configurations
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

# Tab configuration
introduction_tab, data_tab, model_tab, explanations_tab = st.tabs(["\u2001" * 8 + x for x in [":microbe: Introduction", ":bar_chart: Data", ":robot_face: Model", ":sparkles: Explanations"]])

# INTRODUCTION
with introduction_tab:

    st.session_state.tab = "introduction"
    if floating_button(":sparkles: Ask AI", type = "primary", key = st.session_state.tab + "_tab"):
       st.session_state.ask_ai[0] = not st.session_state.ask_ai[0]

    if st.session_state.ask_ai[0]:
        exec(f'''{st.session_state.tab}_col, {st.session_state.tab}_ai = st.columns([0.5, 0.5]);\nwith {st.session_state.tab}_col:\n\t{generate_tab(introduction_config)}''')
        exec(add_chatbox_col(st.session_state.tab))
    else:
        exec(generate_tab(introduction_config))

# DATA
with data_tab:

    st.session_state.tab = "data"
    if floating_button(":sparkles: Ask AI", type = "primary", key = st.session_state.tab + "_tab"):
        st.session_state.ask_ai[1] = not st.session_state.ask_ai[1]
    
    if st.session_state.ask_ai[1]:
        exec(f'''{st.session_state.tab}_col, {st.session_state.tab}_ai = st.columns([0.5, 0.5]);\nwith {st.session_state.tab}_col:\n\t{generate_tab(data_config)}''')
        exec(add_chatbox_col(st.session_state.tab))
    else:
        exec(generate_tab(data_config))

# MODEL
with model_tab:

    st.session_state.tab = "model"
    if floating_button(":sparkles: Ask AI", type = "primary", key = st.session_state.tab + "_tab"):
        st.session_state.ask_ai[2] = not st.session_state.ask_ai[2]

    if st.session_state.ask_ai[2]:
        exec(f'''{st.session_state.tab}_col, {st.session_state.tab}_ai = st.columns([0.5, 0.5]);\nwith {st.session_state.tab}_col:\n\t{generate_tab(model_config)}''')
        exec(add_chatbox_col(st.session_state.tab))
    else:
        exec(generate_tab(model_config))

# EXPLANATIONS
with explanations_tab:
        
    st.session_state.tab = "explanations"

    viz_col, chatbox_col = st.columns([0.5, 0.5])

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

                user_avatar = ":material/cognition:"; ai_avatar = f"llm_icons/{model_selection}.png"

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
