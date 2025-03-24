import gym
import numpy as np
import pandas as pd
from gym import spaces

class StockTradingEnv(gym.Env):
    """A reinforcement learning environment for stock trading"""
    
    def __init__(self, data_path, initial_balance=10000):
        super(StockTradingEnv, self).__init__()

        # Load stock data
        self.df = pd.read_csv(data_path)

        # Ensure required columns exist
        if "Close" not in self.df.columns or "RSI" not in self.df.columns:
            raise ValueError("Dataset must contain 'Close' and 'RSI' columns!")

        # Environment parameters
        self.initial_balance = initial_balance  # Starting money
        self.balance = initial_balance  # Current money
        self.shares_held = 0  # Number of shares owned
        self.current_step = 0  # Current index in stock data
        self.done = False  # Whether episode is finished

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def reset(self):
        """Resets the environment at the beginning of each episode"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """Returns the current state"""
        return np.array([
            self.df.iloc[self.current_step]["Close"],  # Stock price
            self.df.iloc[self.current_step]["RSI"],    # RSI indicator
            self.shares_held,                          # Number of shares owned
            self.balance                               # Current balance
        ], dtype=np.float32)

    def step(self, action):
        """Performs a step in the environment"""
        if self.done:
            return self._get_observation(), 0, self.done, {}

        current_price = self.df.iloc[self.current_step]["Close"]

        # Execute action
        if action == 1:  # Buy
            num_shares = self.balance // current_price
            self.shares_held += num_shares
            self.balance -= num_shares * current_price
        elif action == 2 and self.shares_held > 0:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        # Calculate reward (profit/loss)
        portfolio_value = self.balance + (self.shares_held * current_price)
        reward = portfolio_value - self.initial_balance

        return self._get_observation(), reward, self.done, {}

    def render(self):
        """Displays environment state"""
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}")

if __name__ == "__main__":
    # Example usage
    env = StockTradingEnv("data/TSLA_processed.csv")
    state = env.reset()
    
    for _ in range(10):  # Take 10 random actions
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
