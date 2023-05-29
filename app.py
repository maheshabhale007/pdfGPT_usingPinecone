import streamlit as st
import time

st.title("pdfGPT")
success_message = st.sidebar.success('Hello')

time.sleep(5)

success_message.empty()