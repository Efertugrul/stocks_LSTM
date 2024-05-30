import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_processing import download_data, preprocess_data, df_to_windowed_df, windowed_df_to_date_X_y
from model_training import build_and_train_model, save_model_and_scalers, scale_data
from model_prediction import predict_stock_price_on_date, load_model_and_scalers

# Parameters
ticker = 'AAPL'
first_date_str = '2021-03-25'
last_date_str = '2024-03-23'
window_size = 20
model_filename = 'stock_model.keras'
scaler_X_filename = 'scaler_X.pkl'
scaler_y_filename = 'scaler_y.pkl'

# Load and preprocess data
df = download_data(ticker)
df = preprocess_data(df)

# Generate windowed dataframe
windowed_df = df_to_windowed_df(df, first_date_str, last_date_str, window_size)

# Convert windowed dataframe to date, X, and y
dates, X, y = windowed_df_to_date_X_y(windowed_df)

# Split data into train, validation, and test sets
q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)
dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

# Scale the data
X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y = scale_data(
    X_train, X_val, X_test, y_train, y_val, y_test)

# Build and train model
model, history = build_and_train_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, window_size)

# Evaluate model
train_loss, train_mae = model.evaluate(X_train_scaled, y_train_scaled)
val_loss, val_mae = model.evaluate(X_val_scaled, y_val_scaled)
print(f"Train Loss: {train_loss}, Train MAE: {train_mae}")
print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")

# Save the model and scalers
save_model_and_scalers(model, scaler_X, scaler_y, model_filename, scaler_X_filename, scaler_y_filename)

# Load the model and scalers for prediction
model, scaler_X, scaler_y = load_model_and_scalers(model_filename, scaler_X_filename, scaler_y_filename)

# Predict stock prices on multiple dates to check for variability
target_dates = ['2024-01-01', '2024-02-01', '2024-03-01', '2024-05-30']
predictions = {}
for date in target_dates:
    predicted_price = predict_stock_price_on_date(model, df, date, window_size, scaler_X, scaler_y)
    predictions[date] = predicted_price
    print(f"Predicted stock price for {date}: {predicted_price}")

# Optionally, visualize the predicted price for one of the dates
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Actual Prices')
for date in target_dates:
    plt.axvline(pd.to_datetime(date), color='red', linestyle='--', label=f'Prediction Date: {date}')
    plt.scatter([pd.to_datetime(date)], [predictions[date]], color='red', label=f'Predicted Price: {predictions[date]:.2f}')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()
plt.xlim([pd.to_datetime('2020-01-01'), pd.to_datetime('2025-01-01')])  # Zoom into relevant range
plt.show()
