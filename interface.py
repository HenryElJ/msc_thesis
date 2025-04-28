import streamlit as st
from google import genai
from mistralai import Mistral
from initialise import api_keys

st.set_page_config(layout = "wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def stream_output(stream):
    for chunk in stream:
        yield chunk.text if model == "google" else chunk.data.choices[0].delta.content

model = "google"
client = genai.Client(api_key = api_keys["google"])
chat = client.chats.create(model = "gemini-2.0-flash", history = st.session_state.chat_history)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me a question"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = chat.send_message_stream(prompt) #, config = genai.types.GenerateContentConfig(system_instruction = ""))
            # stream = client.chat.stream(model = "mistral-small", messages = [{"role": "user", "content": prompt}])
            response = st.write_stream(stream_output(stream))
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history = chat.get_history()

####