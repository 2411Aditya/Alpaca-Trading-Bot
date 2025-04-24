import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from backend.data_fetcher import fetch_data

def add_technical_indicators(df):
    df = df.copy()

    # Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Exponential Moving Average (EMA)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + RS))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'
    start = '2000-01-01'
    end = '2025-03-31'

    # Fetch data
    df = fetch_data(ticker, start, end, save_to_csv=True)

    # Add indicators
    df_with_features = add_technical_indicators(df)

    # Save processed data
    output_path = os.path.join(r'C:\Users\adity\OneDrive\Desktop\Projects\Alpaca trading bot\data\data.csv')
    df_with_features.to_csv(output_path, index=True)
    print(f"Feature-enriched data saved to: {output_path}")

    # Preview
    print(df_with_features.head())
