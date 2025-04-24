# backend/train_lstm_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

DATA_PATH = r"C:\Users\adity\OneDrive\Desktop\Projects\Alpaca trading bot\data\data.csv"
MODEL_SAVE_PATH = "models/lstm_model.h5"
SEQ_LENGTH = 60

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    features = ['Close','High','Low','Open','Volume','SMA_20','EMA_20','RSI_14','MACD','MACD_signal','MACD_hist']
    df = df[features]
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X = []
    y = []
    for i in range(SEQ_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LENGTH:i])
        y.append(scaled_data[i][0])  # Predicting 'Close' price

    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    print("[INFO] Loading and preprocessing data...")
    X, y, _ = load_and_preprocess_data()

    print("[INFO] Building and training LSTM model...")
    model = build_lstm_model((X.shape[1], X.shape[2]))
    early_stop = EarlyStopping(monitor='loss', patience=5)

    model.fit(X, y, epochs=100, batch_size=64, callbacks=[early_stop])

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
