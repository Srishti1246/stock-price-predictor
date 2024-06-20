import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import webbrowser

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))


def share_on_social_media(platform, url):
    encoded_url = requests.utils.quote(url)
    if platform == "Twitter":
        twitter_share_url = f"https://twitter.com/intent/tweet?text={encoded_url}"
        webbrowser.open(twitter_share_url)
    elif platform == "LinkedIn":
        linkedIn_share_url = f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_url}"
        webbrowser.open(linkedIn_share_url)

st.sidebar.header("Share on Social Media")
if st.sidebar.button("Share on Twitter"):
    share_on_social_media("Twitter", "https://example.com")  # Replace with your content URL
if st.sidebar.button("Share on LinkedIn"):
    share_on_social_media("LinkedIn", "https://example.com")  # Replace with your content URL
