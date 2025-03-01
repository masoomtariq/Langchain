import streamlit as st
import numpy as np

text = st.chat_input("Hii")

with st.chat_message("user"):
  st.write(text)
  st.line_chart(np.random.randn(30, 3))

message = st.chat_message("assistant")
message.write("Hello human")
message.bar_chart(np.random.randn(30, 3))

st.success(message)
