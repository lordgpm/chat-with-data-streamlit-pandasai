import streamlit as st
import pandas as pd
from pandasai import Agent
from pandasai.llm import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse



st.title('Chat with Data Using PandasAI and OpenAI')


with st.sidebar:
    api_key = st.text_input('Enter your API key here','sk-xxx',type='password')
    reset = st.button('Reset chat history')

if "messages" not in st.session_state or reset:
    st.session_state.messages = []

st.write("Chat with your data. Upload your csv file below")

file = st.file_uploader('Upload your CSV file here')

llm = OpenAI(api_token=api_key, model='gpt-3.5-turbo', temperature=0.5)

for msg in st.session_state.messages:
    if msg['role'] != "system":
        st.chat_message(msg['role']).write(msg['content']) 

if file:
    df = pd.read_csv(file)
    pandas_ai = Agent(df, config={'llm': llm, "conversational": True, 'response_parser': StreamlitResponse}, description='You are a data analysis agent. Your main goal is to help non-technical users to analyze data. You are running in a streamlit app so code to display plots should be for streamlit. If a question does not make sense, ask for clarification. ')

    prompt = st.chat_input("Ask your question about your data.")

            
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
    
        with st.chat_message('assistant'):
            with st.spinner('Thinking'):
                result = pandas_ai.chat(prompt)
                st.session_state.messages.append({"role": "assistant", "content": result})
                st.write(result)