# conda activate thesis_3.11 && cd github/msc_thesis && streamlit run dynamic_static_columns.py

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import streamlit.components.v1 as components
from lorem_text import lorem

# st.set_page_config(layout = "wide")
# https://github.com/Socvest/st-screen-stats # https://discuss.streamlit.io/t/build-responsive-apps-based-on-different-screen-features/51625
# https://github.com/Socvest/streamlit-browser-engine # https://discuss.streamlit.io/t/get-browser-stats-like-user-agent-broswer-name-chrome-firefox-ie-etc-whether-app-is-running-on-mobile-or-desktop-and-more/66735

# from st_screen_stats import ScreenData

# from browser_detection import browser_detection_engine
# value = browser_detection_engine()
# st.write(value)

# using react component


# # --- Helper function to create a sample plot ---
# def create_sample_plot(x, y, title):
#     return go.Figure(data=go.Scatter(x=x, y=y, mode='lines'), layout=go.Layout(title=title))

# # --- Generate sample data and plots ---
# x = np.linspace(0, 10, 100)
# plots = []
# captions = []
# for i in range(20):
#     y = np.sin(x + i * 0.5) * (i + 1)  # Vary the sine wave
#     title = f"Plot {i+1}: Sine Wave {i+1}"
#     plots.append(create_sample_plot(x, y, title))
#     captions.append(f"This plot shows the sine wave with a frequency shift and amplitude scaling. (Plot {i+1})")

# # --- Static content for the right column ---
# container_content = [
#     "This is content in the container.",
#     "It will stay static while the plots scroll.",
#     "You can add any Streamlit elements here.",
#     "For example, buttons, text input, etc."
# ]

# # --- Create two columns ---
# col1, col2 = st.columns(2)

# # --- Inject CSS ---
# st.markdown(
#     """
#     <style>
#         body {
#             overflow: hidden; /* Prevent overall page scrolling */
#         }
#         .stHorizontalBlock {
#             display: flex;
#             align-items: flex-start;
#             height: 90vh; /* Make the columns take up the full viewport height */
#         }
#         .stColumn {
#             flex: 1;           /* Make both columns equal width */
#             max-width: 50%;    /* Limit the width to 50% to prevent stretching */
#         }
#         .stColumn:first-child {
#             overflow-y: auto;
#             max-height: 90vh; /* Match the height of the parent */

#         }
#         .stColumn:last-child {
#             position: sticky;
#             top: 0;
#             overflow-y: auto;
#             max-height: 90vh;
#         }

#         /* Adjust this selector as needed based on your HTML */
#         .main > div:first-child > div:nth-child(2) {
#             height: 90vh;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # --- Left column: Scrolling Plots with Captions ---
# with col1:
#     st.subheader("Scrolling Plots")
#     for i in range(20):
#         st.plotly_chart(plots[i], use_container_width=True)
#         st.caption(captions[i])

# # --- Right column: Static Container ---
# with col2:
#     st.subheader("Static Container")
#     for item in container_content:
#         st.write(item)
#     st.button("A Button in the Container")
#     st.text_input("Text Input in the Container")


#  ====
st.set_page_config(layout = "wide")
st.title("Echo Bot")
st.sidebar.title("Testing")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container(height = 300)

with chat_container:
# Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    with chat_container:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
    
# Display assistant response in chat message container
    with chat_container:
        with st.chat_message("assistant"):
            response = lorem.paragraphs(3)
            st.write(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})