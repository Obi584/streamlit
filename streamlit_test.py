import streamlit as st
import pandas as pd

st.set_page_config(
    page_title = "Neural Network Models",
    page_icon = "🥳"
)

if 'page' not in st.session_state:
    st.session_state.page = 'streamlit_test'

st.title("Homepage")
st.write("This is a streamlit test that includes 3 models made from dense and convoluted neural networks!")

    # Navigation buttons
if st.session_state.page == 'streamlit_test':
    if st.button("Regression Model"):
        st.session_state.page = 'pages/1_regmodel'
    if st.button("Text Classification Model"):
        st.session_state.page = 'pages/2_textmodel'
    if st.button("Image Classification Model"):
        st.session_state.page = 'pages/3_imagemodel'