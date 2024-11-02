# RL Stock Trader Project

## Overview

This project aims to build a Reinforcement Learning (RL) framework for trading stocks in a simulated environment.

## Directory Structure

- **config/**: Configuration files for the environment, stock defaults, portfolio metrics, and RL settings.
- **data/**: Organizes raw and processed data files, with a data loader module for preprocessing.
- **env/**: Market environment and action space modules.
- **models/financial/**: Contains classes for Stock, Portfolio, and Portfolio metrics.
- **agents/**: Placeholder for RL agent implementations.
- **logs/**: Log files for tracking performance.
- **scripts/**: Scripts for training, evaluating, and simulating.
- **requirements.txt**: Lists dependencies.
- **main.py**: Entry point for running the project.

# TODOs:

- At the moment, `config/logging_config.py` contains all the possible libraries we would need and some necessary decorators, as the project continues this most likely will change.
- Have not tested the ability to enter your own personal columns to the data processor, setting for this has not been added yet.
- Need to add the ability to load models and evaluate them, should be .pth files most likely
- (Optional) Need to add more metrics for processing, such as market turbulence, sharp ration, EMA 200 Moving Averages.
- Need to add, action states outcomes/func calls to portfolio
- Need to set up Market Env, and the Stock class
- Save model setting, load model setting needs to be added, most likely to DataProcessor Class (in `data/data_loader.py`)
   - Might need to add optimizer for hyper parameters (skorch, bayseian optimization and adam are current known options)

- Need to add reward, biasing Neural Nets. (DRQN Research required.)
- Set up States, Action, Reward, Value, and possibly, transition probs; for MDP
- Policy Iteration, and evaluation
- Setting up multiple agents, firstly, the random agent, RL trained agent, and then agent based on a preloaded model (load model feature)
- How are we evaluating? What is the Accuracy? Implement a method for computing results and then verification of results through _Backtesting_.
- (Optional) Stretch goal add graphical interface can see the trades over time, live analysis.

# Implementation Ideas:
- Could use yfinance to auto download large amounts of financial data for either training, evaluation, simulate, and backtesting functions. 
