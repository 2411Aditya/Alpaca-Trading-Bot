import os
import pandas as pd
import yfinance as yf

def fetch_data(ticker_symbol, start_date, end_date, save_to_csv=False, csv_folder='csv'):
   
    # Fetch data from Yahoo Finance
    df = yf.download(ticker_symbol, start=start_date, end=end_date)

    if save_to_csv:
        # Ensure the folder exists; create it if it doesn't
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)

        # Define the path for the CSV file
        csv_file_path = os.path.join(csv_folder, f"{ticker_symbol}_{start_date}_to_{end_date}.csv")

        # Save the DataFrame to a CSV file
        df.to_csv(r'C:\Users\adity\OneDrive\Desktop\Projects\Alpaca trading bot\data\data.csv', index=True)
        print(r"C:\Users\adity\OneDrive\Desktop\Projects\Alpaca trading bot\data\data.csv}")

    return df

# Import the fetch_data function from data_loader.py
from data_fetcher import fetch_data

# Define parameters
ticker = 'AAPL'
start = '2000-01-01'
end = '2025-03-31'

# Fetch data and save to CSV
data_frame = fetch_data(ticker, start, end, save_to_csv=True)

# The data is now saved in the 'csv' folder and also available as a DataFrame
print(data_frame.head(5))