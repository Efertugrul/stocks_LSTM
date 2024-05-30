from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
from datetime import datetime
from data_processing import download_data, preprocess_data, df_to_windowed_df, windowed_df_to_date_X_y
from model_training import build_and_train_model, save_model_and_scalers, scale_data
from model_prediction import predict_stock_price_on_date, load_model_and_scalers
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Parameters
WINDOW_SIZE = 20
MODEL_FILENAME = 'stock_model.keras'
SCALER_X_FILENAME = 'scaler_X.pkl'
SCALER_Y_FILENAME = 'scaler_y.pkl'

# Store models and scalers in memory
models = {}

def train_model(ticker):
    # Use today's date as the end date for training
    last_date_str = datetime.today().strftime('%Y-%m-%d')

    # Load and preprocess data
    df = download_data(ticker)
    df = preprocess_data(df)

    # Generate windowed dataframe
    windowed_df = df_to_windowed_df(df, '2021-03-25', last_date_str, WINDOW_SIZE)

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
    model, history = build_and_train_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, WINDOW_SIZE)

    # Save the model and scalers
    save_model_and_scalers(model, scaler_X, scaler_y, MODEL_FILENAME, SCALER_X_FILENAME, SCALER_Y_FILENAME)

    # Calculate performance metrics
    train_loss, train_mae = model.evaluate(X_train_scaled, y_train_scaled)
    val_loss, val_mae = model.evaluate(X_val_scaled, y_val_scaled)

    return model, scaler_X, scaler_y, train_mae, val_mae

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    ticker = request.form['ticker']

    # Check if the model for the selected ticker is already trained
    if ticker not in models:
        # Train the model with the selected company
        model, scaler_X, scaler_y, train_mae, val_mae = train_model(ticker)
        models[ticker] = (model, scaler_X, scaler_y, train_mae, val_mae)
    else:
        model, scaler_X, scaler_y, train_mae, val_mae = models[ticker]

    return redirect(url_for('date_selection', ticker=ticker, train_mae=train_mae, val_mae=val_mae))

@app.route('/date_selection')
def date_selection():
    ticker = request.args.get('ticker')
    train_mae = request.args.get('train_mae')
    val_mae = request.args.get('val_mae')
    return render_template('date_selection.html', ticker=ticker, train_mae=train_mae, val_mae=val_mae)

@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form['date']
    ticker = request.form['ticker']

    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD."

    # Load the model and scalers for the selected ticker
    model, scaler_X, scaler_y, train_mae, val_mae = models[ticker]

    # Load and preprocess data
    df = download_data(ticker)
    df = preprocess_data(df)

    # Predict stock price
    predicted_price = predict_stock_price_on_date(model, df, date_str, WINDOW_SIZE, scaler_X, scaler_y)

    # Calculate confidence intervals
    y_preds = []
    for _ in range(100):
        y_pred = predict_stock_price_on_date(model, df, date_str, WINDOW_SIZE, scaler_X, scaler_y)
        y_preds.append(y_pred)
    lower_bound = np.percentile(y_preds, 2.5)
    upper_bound = np.percentile(y_preds, 97.5)
    mean_pred = np.mean(y_preds)
    std_dev = np.std(y_preds)

    # Plot actual vs predicted prices using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=[target_date], y=[predicted_price], mode='markers', name=f'Predicted Price: {predicted_price:.2f}', marker=dict(color='red', size=10)))
    fig.add_trace(go.Scatter(x=[target_date, target_date], y=[lower_bound, upper_bound], mode='lines', name='Confidence Interval', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'Stock Price Prediction for {ticker}', xaxis_title='Date', yaxis_title='Close Price')

    # Convert Plotly figure to HTML
    plot_html = pio.to_html(fig, full_html=False)

    return render_template('predict.html', date=date_str, predicted_price=predicted_price, ticker=ticker,
                           lower_bound=lower_bound, upper_bound=upper_bound, mean_pred=mean_pred, std_dev=std_dev,
                           plot_html=plot_html, train_mae=train_mae, val_mae=val_mae)

@app.route('/loading')
def loading():
    return render_template('loading.html')

if __name__ == '__main__':
    app.run(debug=True)
