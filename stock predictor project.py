import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
#from fbprophet import Prophet
#from fbprophet.plot import plot_plotly  # Corrected import
from plotly import graph_objs as go  # Corrected import
#hjyu
START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d") 

st.title("Stock Prediction Website")

stocks = ("AAPL", "GOOG", "MSFT", "GME")

dropdown = st.multiselect('Pick your assets', stocks)

selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)  # Corrected function name
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)  # Corrected attribute name
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)  # Corrected attribute name
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.subheader('Forecast plot')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
