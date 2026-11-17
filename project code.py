import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Predictor")
st.sidebar.header("Select Stock")
stock = st.sidebar.selectbox("Choose Stock", ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"])
data = yf.download(stock, period="1y") 
data.columns = data.columns.get_level_values(0)
st.write(data.columns)
st.subheader("📊 Last 5 Days Data")
st.write(data.tail())
st.subheader("📈 Closing Price Chart")
data['MA50'] = data['Close'].rolling(50).mean() 
st.subheader("50-Day Moving Average")
#st.line_chart(data[['Close','MA50']])
chart_data = pd.concat([data["Close"], data["MA50"]], axis=1)
chart_data.columns = ["Close", "MA50"]
st.line_chart(chart_data)
st.line_chart(data['Close'])
data['Prediction'] = data['Close'].shift(-5)
from sklearn.model_selection import train_test_split
X = np.array(data[['Close']][:-5])
y = np.array(data['Prediction'][:-5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
pred_test = model.predict(X_test)
accuracy = r2_score(y_test, pred_test)
st.subheader("📊 Model Accuracy")
st.write(round(accuracy * 100, 2), "%")
future = np.array(data[['Close']].tail(5))
prediction = model.predict(future)
prediction = [round(p, 2) for p in prediction]
st.subheader("🔮 Next 5 Days Prediction")
days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
pred_df = pd.DataFrame({
    "Day": days,
    "Predicted Price": prediction})
st.table(pred_df)
st.subheader("📉 Prediction Graph")
plt.figure()
plt.plot(days, prediction, marker='o')
plt.xlabel("Days")
plt.ylabel("Predicted Price")
plt.title("5 Days Stock Prediction")
st.pyplot(plt)
st.subheader("💡 Investment Suggestion")
last_price = float(data['Close'].dropna().iloc[-1])
if float(prediction[-1]) > last_price:
    st.success("📈 Suggestion: BUY")
else:
    st.error("📉 Suggestion: SELL")