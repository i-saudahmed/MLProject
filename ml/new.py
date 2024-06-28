import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('Fresh Mart Sales Records.csv')

# Select necessary columns for analysis
columns_needed = ['Item Type', 'Order Date', 'Units Sold', 'Unit Price']
cleaned_data = data[columns_needed]

# Convert 'Order Date' to datetime
cleaned_data['Order Date'] = pd.to_datetime(cleaned_data['Order Date'])

# Streamlit app
st.title('Item Sales Dashboard')

# Dropdown for selecting item type
item_types = cleaned_data['Item Type'].unique()
selected_item = st.selectbox('Select Item Type', item_types)

# Filter data based on selected item type
filtered_data = cleaned_data[cleaned_data['Item Type'] == selected_item]

# Display results
st.subheader(f'Sales Details for {selected_item}')
total_units_sold = filtered_data['Units Sold'].sum()
unique_unit_prices = filtered_data['Unit Price'].unique()

st.write(f'Total Units Sold: {total_units_sold}')
if len(unique_unit_prices) == 1:
    st.write(f'Unit Price: ${unique_unit_prices[0]:.2f}')
else:
    st.write('Unit Price: Varies')

st.write(filtered_data)

# Time series forecasting
st.subheader('Predict Future Sales')
date_sales_data = filtered_data.groupby('Order Date')['Units Sold'].sum().reset_index()
date_sales_data.set_index('Order Date', inplace=True)

# Split data into training and testing sets
train_size = int(len(date_sales_data) * 0.8)
train_data, test_data = date_sales_data[:train_size], date_sales_data[train_size:]

# Train ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()

# Forecast future sales on test data
forecast_steps = st.slider('Select number of days to forecast', min_value=1, max_value=365, value=30)
forecast = model_fit.forecast(steps=len(test_data))

# Calculate evaluation metrics
mae = mean_absolute_error(test_data, forecast)
mse = mean_squared_error(test_data, forecast)
st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
st.write(f'Mean Squared Error (MSE): {mse:.2f}')

# Forecast future sales for the selected number of days
future_forecast = model_fit.forecast(steps=forecast_steps)

# Create a DataFrame for future predictions
future_dates = pd.date_range(start=date_sales_data.index[-1], periods=forecast_steps+1, freq='D')[1:]
future_forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Units Sold': future_forecast})

# Display future predictions
st.subheader('Future Sales Predictions')
st.write(future_forecast_df)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(date_sales_data, label='Historical Sales',color='red')
plt.plot(test_data.index, forecast, label='Test Forecast',color='black')
plt.plot(future_dates, future_forecast, label='Future Forecast',color='blue')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.title(f'Sales Forecast for {selected_item}')
plt.legend()
plt.grid(True)

st.pyplot(plt)
