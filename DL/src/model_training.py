from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import joblib
from sklearn.preprocessing import MinMaxScaler

def build_and_train_model(X_train, y_train, X_val, y_val, window_size):
    input_shape = (window_size, 1)
    model = Sequential([
        layers.Input(input_shape),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    return model, history

def save_model_and_scalers(model, scaler_X, scaler_y, model_filename, scaler_X_filename, scaler_y_filename):
    model.save(model_filename)
    joblib.dump(scaler_X, scaler_X_filename)
    joblib.dump(scaler_y, scaler_y_filename)
    print("Model and scalers saved successfully.")

def scale_data(X_train, X_val, X_test, y_train, y_val, y_test):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y
