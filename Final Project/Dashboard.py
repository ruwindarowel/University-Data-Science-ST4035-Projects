import streamlit as st
import pandas as pd

st.title("Hello WorldFFFasdfaF")

#Backend
df = pd.read_csv("../Final Project/churn.csv")

st.bar_chart(df["churn"])