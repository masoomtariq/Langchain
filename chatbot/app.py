from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import RetryOutputParser

import os
from import load_dotenv

## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")