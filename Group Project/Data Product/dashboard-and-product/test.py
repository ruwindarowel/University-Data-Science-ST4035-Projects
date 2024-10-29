import streamlit as st

if "toggle" not in st.session_state:
    st.session_state.toggle = True  # Default to on

if "toggle_key" not in st.session_state:
    st.session_state.toggle_key = 1


def toggle_toggle():
    st.session_state.toggle = not st.session_state[st.session_state.toggle_key]
    st.session_state.toggle_key += 1


st.toggle("Toggle", value=st.session_state.toggle, key=st.session_state["toggle_key"],on_change=toggle_toggle)

st.write(st.session_state[st.session_state.toggle_key])