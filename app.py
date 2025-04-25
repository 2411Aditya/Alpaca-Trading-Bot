import time
from flask import Flask, jsonify, request, send_from_directory
import os
from backend.rl_agent import train_rl_agent, test_model, evaluate_model
from backend.simulate_trading_env import StockTradingEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from alpaca_trade_api.rest import REST
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env 
load_dotenv()

# Flask App
app = Flask(__name__, static_folder='frontend/static', template_folder='frontend')

# Constants
MODEL_PATH = "models/trading_rl_model"
DATA_PATH = "data/stock_data_with_predictions.csv"

# Global trading flag
is_trading = False

# Alpaca credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")
alpaca_api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# Serve the main HTML page
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

# Load model and environment
def load_model():
    model = DQN.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    return model, env

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

# Predict route
@app.route('/predict', methods=['GET'])
def predict():
    model, env = load_model()
    obs = env.reset()
    action, _ = model.predict(obs)
    return jsonify({"action": int(action)})

# Start autonomous trading
@app.route('/start-autotrading', methods=['POST'])
def start_autotrading():
    global is_trading
    is_trading = True

    try:
        symbol = request.json.get("symbol", "AAPL")
        qty = int(request.json.get("qty", 10))

        model, env = load_model()
        obs = env.reset()

        while is_trading:
            action, _ = model.predict(obs)

            action_map = {0: "buy", 1: "hold", 2: "sell"}
            action_str = action_map.get(int(action), "hold")

            try:
                if action_str == "buy":
                    alpaca_api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="gtc")

                elif action_str == "sell":
                    try:
                        position = alpaca_api.get_position(symbol)
                        if int(position.qty) >= qty:
                            alpaca_api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="gtc")
                        else:
                            print(f"⚠️ Not enough shares to sell. Current position: {position.qty}")
                    except Exception as e:
                        print(f"⚠️ Error checking position before selling: {e}")

            except Exception as trade_error:
                print(f"⚠️ Trade execution error: {trade_error}")

            obs, reward, done, _ = env.step(action)

            if done:
                print("✅ Episode completed.")
                break

            time.sleep(5)

        return jsonify({"message": "🚀 Autonomous trading started!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Stop trading route
@app.route('/stop-autotrading', methods=['POST'])
def stop_autotrading():
    global is_trading
    is_trading = False
    return jsonify({"message": "🛑 Trading stopped manually."}), 200

# Run Flask
if __name__ == "__main__":
    app.run(debug=True)
