import alpaca_trade_api as tradeapi
import numpy as np
import time
import torch
import pandas as pd
from trading_env import StockTradingEnv
from dqn_Trader import DQNAgent
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"


# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Define the stock symbol to trade
STOCK = "AAPL"

# Load the trained model
agent = DQNAgent(state_dim=4, action_dim=3)
agent.model.load_state_dict(torch.load("trained_model.pth"))
agent.model.eval()

def get_real_time_data(symbol):
    barset = api.get_bars(symbol, timeframe='1Min', limit=10)  # ✅ Correct format
    close_prices = [bar.c for bar in barset]  # Extract closing prices
    return np.array(close_prices)

def execute_trade(action):
    """Executes buy/sell orders on Alpaca."""
    if action == 0:
        print(" Holding - No trade executed")
    elif action == 1:
        print(" Buying 1 share")
        api.submit_order(symbol=STOCK, qty=1, side='buy', type='market', time_in_force='gtc')
    elif action == 2:
        print(" Selling 1 share")
        api.submit_order(symbol=STOCK, qty=1, side='sell', type='market', time_in_force='gtc')

def live_trading():
    """Runs live trading with real-time stock data."""
    while True:
        state = get_real_time_data(STOCK)
        if state is None:
            print(" No data received. Retrying...")
            time.sleep(60)
            continue

        state = np.reshape(state, [1, 10])  # Reshape for the model
        action = agent.act(state)  # Get action from AI model

        execute_trade(action)  # Execute trade based on the AI decision

        print(f" Action: {action}, State: {state}")
        time.sleep(60)  # Wait 1 minute before the next trade

if __name__ == "__main__":
    live_trading()
