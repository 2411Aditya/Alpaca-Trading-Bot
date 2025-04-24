import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df, seq_length=60):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.seq_length = seq_length
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
        self.position = None  # None = no stock, else = price at which we bought
        self.total_profit = 0

    def reset(self):
        self.current_step = self.seq_length
        self.position = None
        self.total_profit = 0
        return self.get_observation()

    def step(self, action):
        done = self.current_step >= self.max_steps - 1
        reward = 0

        current_data = self.df.iloc[self.current_step]
        current_price = current_data["Close"]

        if action == 0:  # Buy
            if self.position is None:
                self.position = current_price  # Buy at current price
        elif action == 2:  # Sell
            if self.position is not None:
                profit = current_price - self.position
                reward = profit
                self.total_profit += profit
                self.position = None  # Clear position

        # Hold does nothing (reward = 0)

        self.current_step += 1
        observation = self.get_observation()

        return observation, reward, done, {"total_profit": self.total_profit}

    def get_observation(self):
        start = self.current_step - self.seq_length
        end = self.current_step

        numeric_df = self.df.select_dtypes(include=[np.number])
        if 'LSTM_Prediction' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['LSTM_Prediction'])

        data = numeric_df.iloc[start:end].values.astype(np.float32)
        return data
