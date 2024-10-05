import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
st.set_page_config(
    layout='wide'
)
st.title("Dashboard")

#Backend
df = pd.read_csv("churn.csv")

#piechart
churn_counts = df['churn'].value_counts()
voicemail_counts = df['voicemailplan'].value_counts()
fig_churn = go.Figure(data=[go.Pie(labels=churn_counts.index, values=churn_counts.values, hole=0.3)])
fig_vociemail = go.Figure(data=[go.Pie(labels=voicemail_counts.index, values=voicemail_counts.values, hole=0.3)])

fig_months_box = px.box(df, x='churn', y='accountlength', points="all", title="Boxplot of Value by Group")

#Charts# Streamlit app layout

with st.sidebar:
    st.title("Controls")
    
col1, col2, col3 = st.columns(3)
# Display the Plotly pie chart in Streamlit
with col1:
    st.plotly_chart(fig_months_box)
    st.header("Pie Chart")
    st.plotly_chart(fig_churn)
    
with col2:
    st.header("Voicemail Plan")
    st.plotly_chart(fig_vociemail)