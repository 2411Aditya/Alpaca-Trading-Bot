import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

MODEL_PATH = "models/lstm_model.h5"
DATA_PATH = "data/data.csv"

scaler = MinMaxScaler()

def load_data_for_prediction(seq_length=60):
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load data from {DATA_PATH}: {e}")
        return None

    # Use the updated feature set
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist']
    if not all(feature in df.columns for feature in features):
        print(f"[ERROR] Missing required features in data: {set(features) - set(df.columns)}")
        return None

    df = df[features]
    df.dropna(inplace=True)

    scaled_data = scaler.fit_transform(df)
    X_test = []
    for i in range(seq_length, len(scaled_data)):
        X_test.append(scaled_data[i - seq_length:i])
    
    return np.array(X_test)

def predict_with_lstm():
    try:
        print("[INFO] Loading LSTM model and preparing data for prediction...")
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")
        return None

    X = load_data_for_prediction()
    if X is None:
        return None

    try:
        predictions = model.predict(X)
        
        # Reshape the prediction for inverse scaling
        full_preds = np.concatenate([predictions, np.zeros((len(predictions), len(['High', 'Low', 'Open', 'Volume', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist'])))], axis=1)
        predictions = scaler.inverse_transform(full_preds)[:, 0]

        # Calculate RMSE and accuracy
        true_values = pd.read_csv(DATA_PATH)['Close'].iloc[60:].values  # Align with test data length
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        accuracy = 100 - (rmse / np.mean(true_values)) * 100  # Convert RMSE to accuracy

        print(f"[INFO] RMSE: {rmse}")
        print(f"[INFO] Accuracy: {accuracy:.2f}%")
        print("[INFO] LSTM prediction complete.")

        # Add predictions to the DataFrame
        df = pd.read_csv(DATA_PATH)
        df = df.iloc[60:].reset_index(drop=True)  # Align the predictions with the data starting from row 60
        df['LSTM_Prediction'] = predictions

        # Save the new dataframe with predictions to a new CSV
        df.to_csv("data/stock_data_with_predictions.csv", index=False)
        return predictions
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None

if __name__ == "__main__":
    preds = predict_with_lstm()
    if preds is not None:
        print("[INFO] Sample predictions:", preds[-5:], pd.read_csv("data/stock_data_with_predictions.csv").tail(5))
