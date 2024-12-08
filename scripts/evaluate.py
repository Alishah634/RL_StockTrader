# Script for evaluating the RL agent
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

def evaluate(csv_path: str = None, model_path: str = None, run_all_modes=["PPO", "PPO_LargeAction", "A2C", "DRQN" ]):
    """
    DRQN is our model the others are used for benchmarking 
    """

    print(f"\n\nStaritng Evaluation for the Stock: {csv_path} using the following models: {run_all_modes}")

    # Load in the model for the PPO, A2C and Our DRQN model:
    # These need to return something for graphing....

    if "PPO" in run_all_modes:
        print("running ppo small")
        PPO_SmallAction(csv_path, model_path, is_train_mode=False)        
    
    if "PPO_LargeAction" in run_all_modes:
        print("running ppo large")
        PPO_LargeAction(csv_path, model_path, is_train_mode=False)        

    if "A2C" in run_all_modes:
        print("running A2C")
        A2C_LargeAction(csv_path, model_path, is_train_mode=False)

    if "DRQN" in run_all_modes:
        print("running OUR DRQN")
        DRQN_main(csv_path, model_path, is_train_mode=False)



if __name__ == '__main__':
    # Runs for all the models
    evaluate("../data/raw/sp500/DLTR.csv", run_all_modes=["PPO"])
    evaluate("../data/raw/sp500/DLTR.csv", run_all_modes=["PPO_LargeAction"])
    evaluate("../data/raw/sp500/DLTR.csv", run_all_modes=["A2C"])
    evaluate("../data/raw/sp500/DLTR.csv", run_all_modes=["DRQN" ])
    evaluate("../data/raw/sp500/DLTR.csv", run_all_modes=["PPO", "PPO_LargeAction", "A2C", "DRQN" ])
