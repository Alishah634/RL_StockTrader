"""Large action space -100 to 100 ppo strategy:"""
import sys
sys.path.append("env/market_enviroment_2.py")

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3 import A2C

import gymnasium as gym  # Use gymnasium instead of gym
from gymnasium import spaces

from gymnasium import spaces 
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.portfolio_class import Portfolio
from env.market_enviroment_2 import MarketEnvironment


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Preprocess the historical stock market data.

    :param file_path: Path to the CSV file containing the data.
    :return: Preprocessed pandas DataFrame.
    """
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)

    # Strip any leading/trailing whitespace from column names and replace spaces with underscores
    data.columns = data.columns.str.strip('*').str.replace(' ', '_')

    # Ensure the required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Ensure data has: {required_columns}")

    # Sort data by date
    data = data.sort_values('Date').reset_index(drop=True)
    return data

def main():
    # Load and preprocess the data
    data_path = "data/raw/sp500/AAPL.csv"  # Replace with your actual data file
    # data_path = "data/raw/sp500/JPM.csv"  # Replace with your actual data file
    data_path = "data/raw/sp500/AAL.csv"  # Replace with your actual data file
    data_path = "data/raw/sp500/MSFT.csv"  # Replace with your actual data file
    data_path = "data/raw/sp500/DLTR.csv"  # Replace with your actual data file
    data = preprocess_data(data_path)
    # data = data.head(5*365)
    # data = data.head(3485)
     
    print(data)
    # exit(0)
    # Initialize the portfolio and environment
    initial_balance = 1000
    portfolio = Portfolio(name='John', initial_balance=initial_balance)
    env = MarketEnvironment(data=data, portfolio=portfolio, initial_balance=initial_balance)

    # Check the environment
    check_env(env)

    # Wrap the environment for monitoring and vectorization
    env = Monitor(env)
    # env = DummyVecEnv([lambda: env])
    
    env = DummyVecEnv([lambda: MarketEnvironment(data=data, portfolio=portfolio, initial_balance=1000, render_mode='human')])

    # Define the DQN model
    # model = DQN(
    #     policy="MlpPolicy",
    #     env=env,
    #     learning_rate=1e-4,
    #     buffer_size=1000,
    #     learning_starts=1000,
    #     batch_size=32,
    #     tau=1.0,
    #     gamma=0.99,
    #     train_freq=(1, "step"),
    #     gradient_steps=1,
    #     target_update_interval=1000,
    #     exploration_fraction=0.1,
    #     exploration_initial_eps=1.0,
    #     exploration_final_eps=0.05,
    #     verbose=1,
    #     tensorboard_log="./dqn_trading_tensorboard/"
    # )


    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=2048,
        # batch_size=64*2,
        # n_epochs=10,
        gamma=0.9,
        gae_lambda=0.95,
        # clip_range=0.2,
        verbose=1,
        tensorboard_log="./A2C_trading_tensorboard/"
    )



    # Setup evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        # eval_freq=1000,
        eval_freq= 500,
        deterministic=True,
        render=True,
    )

    # Train the model
    # total_timesteps = 100000  # You can adjust this value based on your data and needs
    total_timesteps = 100 #00  # You can adjust this value based on your data and needs
    print("Training started...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=1)
    print("Training completed.")

    # Save the model
    model.save("AC2_largeACTION_dqn_DLTR_changed_trading_model")
    print("Model saved as 'dqn_trading_model'.")

    model.load("AC2_largeACTION_dqn_DLTR_changed_trading_model")
    print("Model loaded as 'dqn_DLTR_changed_trading_model'.")

    # Evaluate the trained model
    obs = env.reset()
    done = False
    print("Evaluating the model...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
 