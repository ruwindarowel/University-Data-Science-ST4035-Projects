import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import date

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

import altair  as alt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pickle

import warnings
warnings.filterwarnings('ignore')

#page configuration
st.set_page_config(
    page_title="Stock Prices Dashboard",
    page_icon="📊",
    layout='wide',
    initial_sidebar_state='expanded'
)

#Backend

#import models
with open("tesla_LSTM.pkl","rb") as f:
    tesla_LSTM = pickle.load(f)
with open("google_LSTM.pkl","rb") as f:
    google_LSTM = pickle.load(f)
with open("nvidia_LSTM.pkl","rb") as f:
    nvidia_LSTM = pickle.load(f)
with open("amazon_LSTM.pkl","rb") as f:
    amazon_LSTM = pickle.load(f)
with open("meta_LSTM.pkl","rb") as f:
    meta_LSTM = pickle.load(f)

with open("amazon_up_down.pkl",'rb') as f:
    amazon_model = pickle.load(f)
    
with open("google_up_down.pkl",'rb') as f:
    google_model = pickle.load(f)
    
with open("meta_up_down.pkl",'rb') as f:
    meta_model = pickle.load(f)
    
with open("nvidia_up_down.pkl",'rb') as f:
    nvidia_model = pickle.load(f)
    
with open("tesla_up_down.pkl",'rb') as f:
    tesla_model = pickle.load(f)
    
#import data
data_tesla = yf.download('TSLA', period='1y', interval='1h')
data_google = yf.download('GOOGL', period='1y', interval='1h')
data_meta = yf.download('META', period='1y', interval='1h')
data_amazon = yf.download('AMZN', period='1y', interval='1h')
data_nvidia = yf.download('NVDA', period='1y', interval='1h')

#setting dict
index_dict = {
    "AMAZON":[data_amazon,amazon_model,amazon_LSTM],
    "GOOGLE":[data_google,google_model,google_LSTM],
    "META":[data_meta,meta_model,meta_LSTM],
    "NVIDIA":[data_nvidia,nvidia_model,nvidia_LSTM],
    "TESLA":[data_tesla,tesla_model,tesla_LSTM],    
}  

#Up down
def calc_up_down(data,model):
    data['Return'] = data['Close'].pct_change()
    data['Direction'] = data['Return'].apply(lambda x: 1 if x > 0 else 0)

    data = data.dropna()

    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Direction']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    next_hour_direction = model.predict([X.iloc[-1]])[0]
    direction_str = 'Up' if next_hour_direction == 1 else 'Down'

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
   
    return accuracy, direction_str

#Predictor Function

def predict_hour(h,data,model,scaler = MinMaxScaler(feature_range=(0, 1))):
    
    close_prices = data['Close'].values
    close_prices = close_prices.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    predictions = []
    look_back=60
    # Start with the last 'look_back' days of data
    # look_back = 60
    last_look_back = scaled_data[-look_back:]
    current_input = last_look_back
    
    for _ in range(h):
        current_input = np.reshape(current_input, (1, look_back, 1))
        next_predicted_price_scaled = model.predict(current_input)
        next_predicted_price = scaler.inverse_transform(next_predicted_price_scaled)
        predictions.append(next_predicted_price[0][0])

        # Update the input sequence to include the latest prediction and remove the oldest one
        next_value_scaled = next_predicted_price_scaled[0]
        current_input = np.append(current_input[0][1:], next_value_scaled)
        current_input = np.reshape(current_input, (1, look_back, 1))
    return predictions
        

#Frontend

st.title("Stock Price Dashboard 💹")

with st.sidebar:
    st.header("Controls")
    company = st.selectbox("Select company",("NVIDIA","META",'GOOGLE','TESLA','AMAZON'))
    value = st.slider("Select a range of values", min_value=1, max_value=24)
    
company_data, up_down_model, forecast_model = index_dict[company][0],index_dict[company][1],index_dict[company][2]
close_price = company_data['Close']
company_data["Datetime"] = company_data.index






col1, col2 = st.columns(2)

with col1:
    st.subheader(f"1 Year historic stock movement of {company}🪙")
    
    c = (
        alt.Chart(company_data).mark_line().encode(x="Datetime",y="Close")
        )
    st.altair_chart(c,use_container_width=True)
    
    accuracy, direction_str = calc_up_down(data=company_data,model=up_down_model)
    st.subheader(f"Price Movement in the next hour, {direction_str}")
    st.subheader(f"Accuracy = {100*accuracy:.3f}%")
    

with col2:
    st.subheader(f"Stock Price Prediction for the next {value} hour(s)💲")
    preds = predict_hour(value,data=company_data,model=forecast_model)
    st.table(
        pd.DataFrame({
            "Hour":[i+1 for i in range(len(preds))],
            "Stock Price":["{:.2f}".format(price) for price in preds]
        })
    )
