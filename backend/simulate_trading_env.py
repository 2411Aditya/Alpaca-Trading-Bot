import gym
from gym import spaces
import numpy as np
import pandas as pd

# Load the stock data with LSTM predictions
DATA_PATH = 'data/stock_data_with_predictions.csv'  # Ensure this is the updated CSV with LSTM predictions
df = pd.read_csv(DATA_PATH)

class StockTradingEnv(gym.Env):
    def __init__(self, df, seq_length=60):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.seq_length = seq_length
        self.current_step = self.seq_length
        self.max_steps = len(self.df)

        # Action space: Buy, Hold, Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: the features used for prediction, plus the LSTM prediction as the last column
        # 
        numeric_df = df.select_dtypes(include=[np.number])
        if 'LSTM_Prediction' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['LSTM_Prediction'])

        self.feature_count = numeric_df.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.seq_length, self.feature_count), dtype=np.float32)

    def reset(self):
        self.current_step = self.seq_length
        return self.get_observation()

    def step(self, action):
        # Ensure the environment doesn't go past the end of the data
        if self.current_step >= self.max_steps - 1:
            done = True
        else:
            done = False

        # Execute the action
        current_data = self.df.iloc[self.current_step]
        if self.current_step + 1 >= len(self.df):
            done = True
            obs = self.get_observation()
            reward = 0  # or your custom penalty/reward
            return obs, reward, done, {}

        next_data = self.df.iloc[self.current_step + 1]


        reward = 0
        if action == 0:  # Buy
            reward = next_data['Close'] - current_data['Close']  # Profit or loss from buying
        elif action == 1:  # Hold
            reward = 0  # No change
        elif action == 2:  # Sell
            reward = current_data['Close'] - next_data['Close']  # Profit or loss from selling

        self.current_step += 1
        observation = self.get_observation()

        return observation, reward, done, {}

    def get_observation(self):
    # Get the window of data
        end = self.current_step
        start = end - self.seq_length
    
    # Filter numeric columns only and drop the LSTM_Prediction if needed
        numeric_df = self.df.select_dtypes(include=[np.number])
    
    # Optional: Drop 'LSTM_Prediction' if it's future-leaking
        if 'LSTM_Prediction' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['LSTM_Prediction'])

        data = numeric_df.iloc[start:end].values.astype(np.float32)
        return data

# Example of usage
if __name__ == "__main__":
    env = StockTradingEnv(df)
    obs = env.reset()
    print(f"Initial observation: {obs[0]}")
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, _ = env.step(action)
        print(f"Step {env.current_step}, Reward: {reward}, Action: {action}")
    