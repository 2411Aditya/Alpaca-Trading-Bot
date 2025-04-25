import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df, seq_length=60, initial_balance=50000):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.seq_length = seq_length
        self.initial_balance = initial_balance
        self.current_step = seq_length
        self.max_steps = len(df)

        numeric_df = df.select_dtypes(include=[np.number])
        if 'LSTM_Prediction' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['LSTM_Prediction'])

        self.feature_count = numeric_df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.seq_length, self.feature_count), dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Hold, 2: Sell
        self.reset()

    def reset(self):
        self.current_step = self.seq_length
        self.cash = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.buy_price = 0
        return self.get_observation()

    def step(self, action):
        done = self.current_step >= self.max_steps - 1
        reward = 0

        current_data = self.df.iloc[self.current_step]
        current_price = current_data["Close"]
        num_shares = 1

        if action == 0:  # Buy
            if self.shares_held == 0 and self.cash >= current_price * num_shares:
                self.shares_held = num_shares
                self.buy_price = current_price
                self.cash -= current_price * num_shares

        elif action == 2:  # Sell
            if self.shares_held > 0:
                profit = (current_price - self.buy_price) * self.shares_held
                reward = profit
                self.total_profit += profit
                self.cash += current_price * self.shares_held
                self.shares_held = 0
                self.buy_price = 0

        # Hold = do nothing

        self.current_step += 1
        observation = self.get_observation()

        return observation, reward, done, {
            "total_profit": self.total_profit,
            "cash": self.cash,
            "shares_held": self.shares_held
        }

    def get_observation(self):
        start = self.current_step - self.seq_length
        end = self.current_step

        numeric_df = self.df.select_dtypes(include=[np.number])
        if 'LSTM_Prediction' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['LSTM_Prediction'])

        data = numeric_df.iloc[start:end].values.astype(np.float32)
        return data
