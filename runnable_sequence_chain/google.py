from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

load_dotenv()

## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Chatbot Using Google api")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
)

st.subheader("Enter your prompt")

input_text = st.text_input("Enter: ")

#initialize the parser
parser = StrOutputParser()

chain = prompt | llm | parser

if input_text:
    st.write(chain.invoke({"question":input_text}))