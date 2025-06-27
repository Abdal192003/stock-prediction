# import numpy as np
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model
# import streamlit as st
# import matplotlib.pyplot as plt

# model = load_model(r'C:\Users\KIIT\Desktop\New folder (2)\my_stock_prediction_model.keras')

# st.header('Stock Market Predictor')

# stock =st.text_input('Enter Stock Symnbol', 'GOOG')
# start = '2012-01-01'
# end = '2022-12-31'

# data = yf.download(stock, start ,end)

# st.subheader('Stock Data')
# st.write(data)

# data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
# data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

# pas_100_days = data_train.tail(100)
# data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
# data_test_scale = scaler.fit_transform(data_test)

# st.subheader('Price vs MA50')
# ma_50_days = data.Close.rolling(50).mean()
# fig1 = plt.figure(figsize=(8,6))
# plt.plot(ma_50_days, 'r')
# plt.plot(data.Close, 'g')
# plt.show()
# st.pyplot(fig1)

# st.subheader('Price vs MA50 vs MA100')
# ma_100_days = data.Close.rolling(100).mean()
# fig2 = plt.figure(figsize=(8,6))
# plt.plot(ma_50_days, 'r')
# plt.plot(ma_100_days, 'b')
# plt.plot(data.Close, 'g')
# plt.show()
# st.pyplot(fig2)

# st.subheader('Price vs MA100 vs MA200')
# ma_200_days = data.Close.rolling(200).mean()
# fig3 = plt.figure(figsize=(8,6))
# plt.plot(ma_100_days, 'r')
# plt.plot(ma_200_days, 'b')
# plt.plot(data.Close, 'g')
# plt.show()
# st.pyplot(fig3)

# x = []
# y = []

# for i in range(100, data_test_scale.shape[0]):
#     x.append(data_test_scale[i-100:i])
#     y.append(data_test_scale[i,0])

# x,y = np.array(x), np.array(y)

# predict = model.predict(x)

# scale = 1/scaler.scale_

# predict = predict * scale
# y = y * scale

# st.subheader('Original Price vs Predicted Price')
# fig4 = plt.figure(figsize=(8,6))
# plt.plot(predict, 'r', label='Original Price')
# plt.plot(y, 'g', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.show()
# st.pyplot(fig4)




import os
import numpy as np
import pandas as pd
import yfinance as yf
import keras
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load model safely with Keras 3.0 compatibility
try:
    # Set Keras backend explicitly
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    
    # Load model with custom objects if needed
    model_path = os.path.join(os.path.dirname(__file__), 'my_stock_prediction_model.keras')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    model = load_model(
        model_path,
        compile=False  # Disable compilation to avoid optimizer loading issues
    )
    
    # Compile model with appropriate settings
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    st.session_state.model_loaded = True
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Please ensure you have the correct versions of TensorFlow and Keras installed.")
    st.error("Try: pip install tensorflow>=2.15.0 keras>=3.0.0")
    st.stop()

st.header('📈 Stock Market Predictor')

# Input
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Fetch Data
data = yf.download(stock, start=start, end=end)

if data.empty:
    st.warning("No data fetched. Check stock symbol or internet connection.")
    st.stop()

# Display Data
st.subheader('Stock Data')
st.write(data.tail())

# Preprocessing
data_train = pd.DataFrame(data['Close'][0: int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test_full = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_full)

# Moving Averages
st.subheader('Price vs MA50')
ma_50 = data['Close'].rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(data['Close'], 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100 = data['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(ma_100, 'b', label='MA100')
plt.plot(data['Close'], 'g', label='Close Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200 = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100, 'r', label='MA100')
plt.plot(ma_200, 'b', label='MA200')
plt.plot(data['Close'], 'g', label='Close Price')
plt.legend()
st.pyplot(fig3)

# Prepare Test Data
x_test = []
y_test = []

for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
try:
    predictions = model.predict(x_test, verbose=0)
    
    # Rescale
    scale_factor = 1 / scaler.scale_[0]
    predictions = predictions.flatten() * scale_factor
    y_test_scaled = y_test * scale_factor
    
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

# Plot Results
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(12, 6))

# Create time indices for x-axis
time_period = range(len(y_test_scaled))

plt.plot(time_period, y_test_scaled, 'g-', label='Actual Price', linewidth=1.5)
plt.plot(time_period, predictions, 'r--', label='Predicted Price', linewidth=1.5)

plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Time Period')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Add some styling
plt.tight_layout()
st.pyplot(fig4)

# Display prediction metrics
try:
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(y_test_scaled, predictions)
    r2 = r2_score(y_test_scaled, predictions)
    
    st.subheader('Model Performance')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.4f}")
    with col2:
        st.metric("R² Score", f"{r2:.4f}")
        
except Exception as e:
    st.warning(f"Could not calculate metrics: {e}")
