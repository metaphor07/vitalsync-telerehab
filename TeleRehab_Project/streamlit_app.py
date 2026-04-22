import streamlit as st

st.set_page_config(page_title="Test App", layout="wide")

st.title("Streamlit Test App")
st.write("If you can see this text, Streamlit is working correctly.")

name = st.text_input("Type anything here:")

if name:
    st.success(f"You typed: {name}")