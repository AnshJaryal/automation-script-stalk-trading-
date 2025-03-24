import pandas as pd
import pandas_ta as ta

def compute_techincal_indicators(file_path):
    df = pd.read_csv(file_path)

    if "Close" not in df.columns:
        raise ValueError("CSV file must contain a 'Close' price column!")
    
    # Convert 'Close' column to numeric (in case there are string values)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Drop rows where 'Close' is NaN after conversion
    df.dropna(subset=['Close'], inplace=True)
    
    #relative strength index
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df["SMA_50"] = ta.sma(df['Close'], length=50)
    df["SMA_200"] = ta.sma(df['Close'], length=200)

    # Moving Average Convergence Divergence (MACD)
    macd = ta.macd(df["Close"])
    df['MACD'] = macd['MACD_12_26_9']

    processed_file = file_path.replace(".csv", "_processed.csv")
    df.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")

    return df

if __name__ == '__main__':
    processed_data = compute_techincal_indicators("data/TSLA.csv")
    print(processed_data.tail())  # Corrected print statement
