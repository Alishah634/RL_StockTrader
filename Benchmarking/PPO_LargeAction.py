"""Large action space -100 to 100 ppo strategy:"""

import os 
import sys
import csv
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = os.getenv('PROJECT_ROOT', "/home/shantanu/RL_Proj/RL_StockTrader")
sys.path.append(ROOT)

from config.logging_config import ensure_directory
ensure_directory(os.path.join(ROOT, "Benchmarking/logs/"))
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
    print(data_path)
    print(is_train_mode)
    data_path = os.path.join(ROOT, data_path)
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

    # Setup evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(ROOT, "Benchmarking/logs/PPO_Large/best_model/"),
        log_path=os.path.join(ROOT, "Benchmarking/logs/PPO_Large/results/"),
        eval_freq= 500,
        deterministic=True,
        render=True,
    )

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
        tensorboard_log=os.path.join(ROOT, "Benchmarking/TensorBoards/LargeAction_ppo_trading_tensorboard/")
    )


    if model_path == None:
        # Use pre trained model path::
        ensure_directory(os.path.join(ROOT, "Benchmarking/trained_models"))
        model_path = os.path.join(ROOT, "Benchmarking/trained_models/PPO_Large_DLTR_changed_trading_model.zip")
    else:
        model_path = os.path.join(ROOT, "Benchmarking/trained_models/", model_path)

    model_path = os.path.abspath(model_path)
    print(model_path)
    # Train the model if in training mode:
    if is_train_mode or not os.path.exists(model_path):
        print("Training started...")
        total_timesteps = 10000  # You can adjust this value based on your data and needs
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=1)
        print("\n\nTraining completed.\n\n")

        # Save the model
        model.save(f"{model_path}")
        print(f"Model saved as '{model_path}'.")

    model.load(f"{model_path}")
    print("Model loaded as 'model_path'.")


    # Evaluate the trained model
    obs = env.reset()
    done = False

    # Store timestep vs portfolio value
    timestep = 0
    portfolio_values = list()
    print("Evaluating the model...")
    while not done:
        # action, _states = model.predict(obs, deterministic=True)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
        
        # Append the portfolio value and timestep to the list
        portfolio_value = portfolio.portfolio_value  # Access the portfolio value from the environment
        portfolio_values.append((timestep, portfolio_value))
        timestep += 1

    # Save portfolio values to CSV
    csv_path = os.path.join(ROOT, "Benchmarking/logs/PPO_Large/portfolio_values.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestep", "Portfolio Value"])  # Write the header
        writer.writerows(portfolio_values)  # Write the data

    print(f"Portfolio values saved to {csv_path}")

    print("Evaluation completed.")

    return


if __name__ == "__main__":
    # Set for evaluation of the pretrained model:
    PPO_LargeAction(f"{ROOT}/data/raw/sp500/JPM.csv", is_train_mode=False)

    # "data/raw/sp500/AAPL.csv"  
    # "data/raw/sp500/JPM.csv"  
    # "data/raw/sp500/AAL.csv"  
    # "data/raw/sp500/MSFT.csv"  
    # "data/raw/sp500/DLTR.csv"  
