import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
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


generation_config = {"temperature": 0 }
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
'''model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-8b",
  generation_config=generation_config,
)'''

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
)
#generation_config=generation_config)

st.subheader("Enter your prompt")

input_text = st.text_input("Enter: ")

parser = PydanticOutputParser(pydantic_object=input_text)

chain = prompt | llm

out = chain.invoke({"question":input_text})

if input_text:
    st.write(parser.invoke(out))
    #st.write(chain.run({"question": input_text}))


#st.write(parser.invoke({"query": state["query"]}))
