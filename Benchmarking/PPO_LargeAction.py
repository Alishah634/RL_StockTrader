"""Large action space -100 to 100 ppo strategy:"""

import os 
import sys
import pandas as pd
from typing import List

from stable_baselines3 import PPO
import gymnasium as gym  # Use gymnasium instead of gym
from gymnasium import spaces 
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


sys.path.append("../")
from config.logging_config import ensure_directory
ensure_directory("logs/")
from env.portfolio_class import Portfolio
from env.LargeAction_market_enviroment import MarketEnvironment


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

def PPO_LargeAction(data_path, model_path: str = None, is_train_mode: bool = True ):
    # Pre process the data:
    preprocessed_data = preprocess_data(data_path)

    # Initialize the portfolio and environment
    initial_balance = 1000
    portfolio = Portfolio(name='John', initial_balance=initial_balance)
    env = MarketEnvironment(data=preprocessed_data, portfolio=portfolio, initial_balance=initial_balance)

    # Check the environment
    check_env(env)

    # Wrap the environment for monitoring and vectorization
    env = Monitor(env)
    env = DummyVecEnv([lambda: MarketEnvironment(data=preprocessed_data, portfolio=portfolio, initial_balance=1000, render_mode='human')])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64*2,
        n_epochs=10,
        gamma=0.9,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./largeAction_ppo_trading_tensorboard/"
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

    if model_path == None:
        # Use pre trained model path::
        ensure_directory("trained_models")
        model_path = "trained_models/largeACTION_PPO_DLTR_changed_trading_model"
    else:
        model_path = os.path.join("trained_models/", model_path)

    # Train the model if in training mode:
    if is_train_mode and os.path.exists(model_path):
        print("Training started...")
        total_timesteps = 10000  # You can adjust this value based on your data and needs
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=0)
        print("Training completed.")

        # Save the model
        model.save(f"{model_path}")
        print("Model saved as '{model_path}'.")

    model.load(f"{model_path}")
    print("Model loaded as 'model_path'.")

    # Evaluate the trained model
    obs = env.reset()
    done = False
    print("Evaluating the model...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
    print("Evaluation completed.")

    return


if __name__ == "__main__":
    # Set for evaluation of the pretrained model:
    PPO_LargeAction("../data/raw/sp500/DLTR.csv", is_train_mode=False)

    # # Load and preprocess the data
    # data_path = "data/raw/sp500/AAPL.csv"  
    # # data_path = "data/raw/sp500/JPM.csv"  
    # data_path = "data/raw/sp500/AAL.csv"  
    # data_path = "data/raw/sp500/MSFT.csv"  
    # # data_path = "data/raw/sp500/DLTR.csv"  
    # data = preprocess_data(data_path)
