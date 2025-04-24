# backend/main.py

from rl_agent import train_rl_agent, evaluate_model, test_model

def main():
    # Train the agent
    model, env = train_rl_agent(total_timesteps=100000)  # Adjust timesteps as needed
    
    # Evaluate the model
    evaluate_model(model, env)
    
    # Test the model for 100 steps
    test_model(model, env)

if __name__ == "__main__":
    main()
