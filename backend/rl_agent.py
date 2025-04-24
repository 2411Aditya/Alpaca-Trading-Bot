# backend/rl_agent.py

import os
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from backend.simulate_trading_env import StockTradingEnv

MODEL_PATH = "models/trading_rl_model"
DATA_PATH = "data/stock_data_with_predictions.csv"  # <- make sure this file exists

def train_rl_agent(total_timesteps=100000):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("[INFO] Initializing trading environment...")
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    print("[INFO] Setting up RL model...")
    model = DQN('MlpPolicy', env, verbose=1)

    print(f"[INFO] Training model for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    print("[INFO] Saving trained model...")
    model.save(MODEL_PATH)

    return model, env

def evaluate_model(model, env, n_eval_episodes=5):
    print("[INFO] Evaluating model...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"[RESULT] Mean Reward: {float(mean_reward):.2f} ± {float(std_reward):.2f}")

def test_model(model, env):
    obs = env.reset()
    total_reward = 0
    print("[INFO] Running model on environment for 100 steps...")
    for step in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        reward_value = reward.item() if hasattr(reward, 'item') else reward
        print(f"Step {step+1} | Action: {action}, Reward: {float(reward_value):.2f}")


        if done:
            print("[INFO] Episode done. Resetting environment.")
            obs = env.reset()
    print(f"[RESULT] Total reward after 100 steps: {float(total_reward):.2f}")

if __name__ == "__main__":
    model, env = train_rl_agent()
    evaluate_model(model, env)
    test_model(model, env)
