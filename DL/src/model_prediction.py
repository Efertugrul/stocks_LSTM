import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def predict_stock_price_on_date(model, df, target_date_str, window_size, scaler_X, scaler_y):
    target_date = pd.to_datetime(target_date_str)
    df_subset = df.loc[:target_date].tail(window_size)

    if len(df_subset) != window_size:
        raise ValueError(f'Not enough data to create a window of size {window_size} for date {target_date_str}')

    X = df_subset['Close'].to_numpy().reshape(-1, 1)
    print(f"Original X values:\n{X}")

    X_scaled = scaler_X.transform(X).reshape(1, window_size, 1)
    print(f"Scaled X values:\n{X_scaled}")

    y_pred_scaled = model.predict(X_scaled).flatten()[0]
    print(f"Scaled prediction: {y_pred_scaled}")

    y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
    print(f"Inverse scaled prediction: {y_pred}")

    return y_pred

def load_model_and_scalers(model_filename, scaler_X_filename, scaler_y_filename):
    model = load_model(model_filename)
    scaler_X = joblib.load(scaler_X_filename)
    scaler_y = joblib.load(scaler_y_filename)
    return model, scaler_X, scaler_y
