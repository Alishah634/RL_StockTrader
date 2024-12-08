# Script for training the RL agent
import os
import sys
import numpy as np
from typing import List 
from tqdm  import tqdm 

sys.path.append("../")

from config.logging_config import logging, setup_logging

from Benchmarking.PPO import PPO_SmallAction
from Benchmarking.PPO_LargeAction import PPO_LargeAction
from Benchmarking.A2C import A2C_LargeAction
from Benchmarking.drqn_model import DRQN_main



def train(csv_path: str = None, model_path: str = None, run_all_modes=["PPO", "PPO_LargeAction", "A2C", "DRQN" ]):
    """
    DRQN is our model the others are used for benchmarking 
    """

    print(f"\n\nStarting Training for the Stock: {csv_path} using the following models: {run_all_modes}")
    exit(0)
    
    # Load in the model for the PPO, A2C and Our DRQN model:
    # These need to return something for graphing....

    if "PPO" in run_all_modes:
        print("running ppo small")
        PPO_SmallAction(csv_path, model_path, is_train_mode=True)        
    
    if "PPO_LargeAction" in run_all_modes:
        print("running ppo large")
        PPO_LargeAction(csv_path, model_path, is_train_mode=True)        

    if "A2C" in run_all_modes:
        print("running A2C")
        # A2C_LargeAction(csv_path, model_path, is_train_mode=True)

    if "DRQN" in run_all_modes:
        print("running OUR DRQN")
        # DRQN_main(csv_path, model_path, is_train_mode=True)




    stock_results = {}

    # Industry Stocks:
    csv_paths = [
        "../data/raw/sp500/APPL.csv" 
        # "../data/raw/sp500/HPE.csv" 
        # ,"../data/raw/sp500/PM.csv" 
        # ,"../data/raw/sp500/PSX.csv" 
        # ,"../data/raw/sp500/MDLZ.csv" 
        # ,"../data/raw/sp500/WU.csv" 
        # ,"../data/raw/sp500/MS-PF.csv"  
        # ,"../data/raw/sp500/GS-PJ.csv" 
        # ,"../data/raw/sp500/MET.csv"
    ]

    # Get list of CSV paths
    # csv_paths = [os.path.join("../data/raw/sp500/", path) for path in os.listdir("../data/raw/sp500/") if path.endswith('.csv')]
    # print(csv_paths)

    # Slice to limit the range of paths (if needed)
    # N_paths = [0, 3]
    # csv_paths = csv_paths[N_paths[0]: N_paths[-1]]

    # Regular expression to extract CSV filename
    # pattern = r"[^/]+\.csv$"


    # # Collect results for this CSV
    # avg_reward = np.mean(total_rewards)
    # results["Model_Net_Profit"] = portfolio.net_profit
    # results["Average_Reward"] = avg_reward
    # results["Market_Return"] = env.market_return
    # results["Expected_Evaluation"] = env.expected_evaluation

    # # Extract the CSV filename
    # match = re.search(pattern, csv_path)
    # if match:
    #     csv_name = match.group()
    #     csv_name = os.path.splitext(csv_name)[0]  # Remove .csv extension
    #     print(f"CSV Filename: {csv_name}")
    # else:
    #     print("No CSV filename found.!!!")

        # Store results in stock_results using the CSV name
    #     stock_results[csv_name] = results

    # # Print final results for each stock
    # for stock_name, res in stock_results.items():
    #     print(
    #         f"Results of STOCK {stock_name}: Model_Net_Profit: {res['Model_Net_Profit']} | "
    #         f"Average_Reward: {res['Average_Reward']} | Market_Return: {res['Market_Return']} | "
    #         f"Expected_Evaluation: {res['Expected_Evaluation']}"
    #     )


    pass


if __name__ == '__main__':
    # Runs for all the models
    train("../data/raw/sp500/DLTR.csv", run_all_modes=["PPO"])
    train("../data/raw/sp500/DLTR.csv", run_all_modes=["PPO_LargeAction"])
    train("../data/raw/sp500/DLTR.csv", run_all_modes=["A2C"])
    train("../data/raw/sp500/DLTR.csv", run_all_modes=["DRQN" ])
    train("../data/raw/sp500/DLTR.csv", run_all_modes=["PPO", "PPO_LargeAction", "A2C", "DRQN" ])
