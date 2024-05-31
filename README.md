# Stock Price Predictor

This project is a web application that predicts stock prices using Long Short-Term Memory (LSTM) models. The application allows users to select a stock ticker, train a model on historical data, and predict future stock prices. The predictions are displayed along with interactive charts and confidence intervals. Additionally, the application shows model performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Features

- **Dynamic Stock Selection**: Users can enter any stock ticker.
- **Model Training**: The model is trained using LSTM on historical data up to the current date.
- **Interactive Graphs**: Predictions are displayed with interactive Plotly charts.
- **Performance Metrics**: Displays model training and validation metrics.
- **Loading Screen**: Shows a loading screen while the model is training.
- **Confidence Intervals**: Predicts the stock price with confidence intervals.
- **Reselect Dates**: Easily go back to the date selection page for more predictions on the same stock.

## Requirements

- Python 3.6+
- Flask
- Pandas
- Numpy
- TensorFlow
- Scikit-learn
- Plotly
- Yfinance

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Efertugrul/stocks_LSTM.git
   cd stocks_LSTM
