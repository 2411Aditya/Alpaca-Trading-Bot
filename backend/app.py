# backend/app.py

from flask import Flask, jsonify, request
from rl_agent import train_rl_agent, test_model, evaluate_model
from simulate_trading_env import StockTradingEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Flask App
app = Flask(__name__)

# Constants
MODEL_PATH = "models/trading_rl_model"
DATA_PATH = "data/stock_data_with_predictions.csv"

# Alpaca credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")  # e.g. "https://paper-api.alpaca.markets"

alpaca_api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# Load the trained RL model and environment
def load_model():
    model = DQN.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    return model, env

# Home route
@app.route("/")
def home():
    return jsonify({"message": "🤖 Welcome to the AI Trading Bot API!"})

# Train route
@app.route('/train', methods=['GET'])
def train():
    model, env = train_rl_agent(total_timesteps=100000)
    return jsonify({"message": "✅ Training complete, model saved!"}), 200

# Test route
@app.route('/test', methods=['GET'])
def test():
    model, env = load_model()
    test_model(model, env)
    return jsonify({"message": "✅ Testing complete!"}), 200

# Predict one action
@app.route('/predict', methods=['GET'])
def predict():
    model, env = load_model()
    obs = env.reset()
    action, _ = model.predict(obs)
    return jsonify({"action": int(action)})

# Live Alpaca trading based on model prediction
@app.route('/start-trading', methods=['POST'])

def start_trading():
    try:
        # Log request data
        print(f"Received request: {request.json}")

        # Load model and make prediction
        model, env = load_model()
        obs = env.reset()
        action, _ = model.predict(obs)

        action_map = {
            0: "buy",
            1: "hold",
            2: "sell"
        }

        selected_action = int(action)
        action_str = action_map.get(selected_action, "hold")

        symbol = request.json.get("symbol", "AAPL")
        qty = int(request.json.get("qty", 10))

        # Log predicted action and order details
        print(f"Predicted action: {action_str}, Symbol: {symbol}, Quantity: {qty}")

        # Place order based on predicted action
        if action_str == "buy":
            response = alpaca_api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="gtc")
        elif action_str == "sell":
            response = alpaca_api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="gtc")

        return jsonify({
            "message": f"🚀 Executed {action_str.upper()} action on {symbol} for {qty} share(s).",
            "action": action_str,
            "symbol": symbol,
            "quantity": qty
        }), 200

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run Flask server
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
