from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import RetryOutputParser

from dotenv import load_dotenv
import os

load_dotenv()  # Add this line to load environment variables from .env file


## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")