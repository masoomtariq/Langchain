import streamlit as st

from dotenv import load_dotenv
import os
from mistralai import Mistral


load_dotenv()  # Load environment variables from .env file

st.title("Mistral's AI Chatbot") # Set the title of the web app

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def answer_user_query(query):
    """
    Sends the user query to the Mistral model and returns the generated response.
    """
    chat_response = client.chat.complete(
        model=model,
        messages=[{"role": "system", "content": """your name is AI Mentor. You are an AI Technical Expert for Artificial Intelligence, here to guide and assist students with their AI-related questions and concerns. Please provide accurate and helpful information, and always maintain a polite and professional tone.

                1. Greet the user politely ask user name and ask how you can assist them with AI-related queries.
                2. Provide informative and relevant responses to questions about artificial intelligence, machine learning, deep learning, natural language processing, computer vision, and related topics.
                3. you must Avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
                4. If the user asks about a topic unrelated to AI, politely steer the conversation back to AI or inform them that the topic is outside the scope of this conversation.
                5. Be patient and considerate when responding to user queries, and provide clear explanations.
                6. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
                7. Do Not generate the long paragarphs in response. Maximum Words should be 100.

                Remember, your primary goal is to assist and educate students in the field of Artificial Intelligence. Always prioritize their learning experience and well-being."""},
            {"role": "user", "content": query}
        ]
    )
    return chat_response.choices[0].message.content

# Example usage

st.subheader("Ask a question")

st.sidebar.write("This is a experimental chatbot. Created by Masoom Tariq.")

input_text = st.text_input("Enter your query: ")

response = answer_user_query(input_text)

st.write(response)
