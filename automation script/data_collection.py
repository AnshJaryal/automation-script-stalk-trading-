import yfinance as yf
import pandas as pd

def fetch(ticker,start = "2020-01-01",end = "2024-01-01"):
 data = yf.download(ticker,start = start , end = end)
 if data.empty:
  print("No data found for {ticker}. check the ticker symbol!")
  return None
 file_path = f"data/{ticker}.csv"
 data.to_csv(file_path)
 print(f"Data saved to (file_path)")
 return data

if __name__ == "__main__":
    stock_data = fetch("TSLA")
    print(stock_data.head())