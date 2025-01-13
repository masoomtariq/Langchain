from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import RetryOutputParser

from dotenv import load_dotenv
import os
from mistralai import Mistral


load_dotenv()  # Load environment variables from .env file

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def answer_user_query(query):
    """
    Sends the user query to the Mistral model and returns the generated response.
    """
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    return chat_response.choices[0].message.content

# Example usage
user_query = input("Enter your query: ")
response = answer_user_query(user_query)
print(response)

# Removed redundant load_dotenv() call

## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")