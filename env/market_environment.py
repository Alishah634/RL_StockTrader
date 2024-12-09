import gymnasium as gym
# from gym import spaces
from gymnasium import spaces

import os, sys

ROOT = os.getenv('PROJECT_ROOT', "/home/shantanu/RL_Proj/RL_StockTrader")
sys.path.append(ROOT)

import numpy as np
import pandas as pd
from env.portfolio_class import Portfolio
from config.logging_config import setup_file_logger
import logging 


import inspect

def print_call_stack(func):
    """A decorator to print the call stack when the function is called."""
    def wrapper(*args, **kwargs):
        print(f"\nCall stack for {func.__name__}:")
        for frame in inspect.stack()[1:]:  # Skip the current frame
            print(f"  File '{frame.filename}', line {frame.lineno}, in {frame.function}")
        print(f"\nCalling {func.__name__}...\n")
        return func(*args, **kwargs)
    return wrapper



class MarketEnvironment(gym.Env):
    """A custom trading environment for Reinforcement Learning with stock market data."""

    metadata = {'render.modes': ['human']}
    def __init__(self, data: pd.DataFrame, portfolio: Portfolio, initial_balance: float = 1000, render_mode = 'human'):
        super(MarketEnvironment, self).__init__()
        self.render_mode = 'human'  # Store the render mode
        
        # Create a logger specifically for the market environment
        self.market_env_logging = logging.getLogger(__name__)
        setup_file_logger(__name__, f'{ROOT}/logs/market_environment.log', log_level=logging.INFO, will_propogate=True)

        # Initialize attributes
        required_columns = {'Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume'}
        if not required_columns.issubset(data.columns):
            self.market_env_logging.debug(f"{data.columns}")
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

        self.data = data.reset_index(drop=True)

        # print (self.data)
        # exit(0)
        self.portfolio = portfolio
        self.initial_balance = portfolio.initial_balance
        self.current_step = 0
        self.current_price = 0

        # Log or print initial balance
        self.market_env_logging.info(f"Initialized Market Environment with Portfolio Balance: {self.portfolio.balance} and Initial Balance: {self.initial_balance}")

        # Define action space: 0 = hold, 1 = buy, 2 = sell
        # self.action_space = spaces.Discrete(3)

        max_shares = 100  # Define a reasonable maximum for the number of shares
        self.action_space = spaces.MultiDiscrete([3, max_shares + 1])  # Action type and number of shares

        # Define observation space (Open, High, Low, Close, Adj_Close, Volume)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    # def reset(self):
    #     """Reset the environment to an initial state."""
    #     self.current_step = 0
    #     self.portfolio.balance = self.initial_balance
    #     self.portfolio.holdings = 0
    #     self.portfolio.net_profit = 0

    #     # Log reset state
    #     self.market_env_logging.info(f"Environment Reset: Balance: {self.portfolio.balance}, Holdings: {self.portfolio.holdings}, Net Profit: {self.portfolio.net_profit}")

    #     return self._next_observation()
    
    # @print_call_stack
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""

        # Log reset state
        self.market_env_logging.info(
            f"Environment BEFORE RESET: Balance: {self.portfolio.balance}, Holdings: {self.portfolio.holdings}, Net Profit: {self.portfolio.net_profit}"
        )

        # Set the random seed if provided
        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio.balance = self.initial_balance
        self.portfolio.holdings = 0
        self.portfolio.net_profit = 0

        # Log reset state
        self.market_env_logging.info(
            f"Environment Reset: Balance: {self.portfolio.balance}, Holdings: {self.portfolio.holdings}, Net Profit: {self.portfolio.net_profit}"
        )

        # Get the initial observation
        observation = self._next_observation()

        # Return the initial observation and an empty info dictionary
        return observation, {}





    # def _next_observation(self):
    #     """Get the next state/observation."""
    #     obs = self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume']].values
    #     self.current_price = self.data.iloc[self.current_step]['Close']
        return obs

    def _next_observation(self):
        """Get the next state/observation."""
        obs = self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume']].values
        self.current_price = self.data.iloc[self.current_step]['Close']
        return obs.astype(np.float32)  # Ensure dtype matches the observation_space definition



    # def step(self, action, shares=0):
    #     """Execute a trade action and calculate the next state."""
    #     if action == 1:  # Buy
    #         max_affordable_shares = self.portfolio.balance // self.current_price
    #         # shares = min(shares, max_affordable_shares) # TODO This line lets the agent buy as much as possible!!!

    #         # Debugging information to understand the current state of the transaction:
    #         self.market_env_logging.debug(f"Attempting to buy shares: Requested = {shares}, Affordable = {max_affordable_shares}")
    #         self.market_env_logging.debug(f"Current Balance: {self.portfolio.balance}, Current Price: {self.current_price}")

    #         if shares > 0 and shares * self.current_price <= self.portfolio.balance:
    #             self.market_env_logging.info(f"Bought {shares} shares at price {self.current_price}")
    #             self.portfolio.balance -= shares * self.current_price
    #             self.portfolio.holdings += shares
    #             self.portfolio.total_shares_bought += shares
    #         else:
    #             # Proper warning for insufficient balance
    #             self.market_env_logging.warning("Invalid number of shares or insufficient balance for buying.")

    #     elif action == 2:  # Sell
    #         if shares > 0 and shares <= self.portfolio.holdings:
    #             self.market_env_logging.info(f"Attempting to sell shares: Requested = {shares}, Holdings = {self.portfolio.holdings}")
    #             self.portfolio.balance += shares * self.current_price
    #             self.portfolio.holdings -= shares
    #             self.portfolio.total_shares_sold += shares
    #         else:
    #             # Proper warning for insufficient holdings
    #             self.market_env_logging.warning("Invalid number of shares or insufficient holdings for selling.")

    #     # Calculate portfolio value and net profit
    #     self.portfolio.portfolio_value = self.portfolio.balance + (self.portfolio.holdings * self.current_price)
    #     self.portfolio.net_profit = self.portfolio.portfolio_value - self.initial_balance

    #     # Reward: Change in portfolio value
    #     reward = float(self.portfolio.net_profit)

    #     # Move to the next time step
    #     self.current_step += 1
    #     done = self.current_step >= len(self.data) - 1  # Check if at the end of data

    #     # Next observation/state
    #     next_obs = self._next_observation() if not done else None

    #     return next_obs, reward, done, {}
    

    # # @print_call_stack
    # def step(self, action, shares=1):
    #     """Execute a trade action and calculate the next state."""
    #     if action == 1:  # Buy
    #         max_affordable_shares = self.portfolio.balance // self.current_price
    #         # shares = min(shares, max_affordable_shares) # TODO This line lets the agent buy as much as possible!!!

    #         # Debugging information to understand the current state of the transaction:
    #         # self.market_env_logging.debug(f"Attempting to buy shares: Requested = {shares}, Affordable = {max_affordable_shares}")
    #         # self.market_env_logging.debug(f"Current Balance: {self.portfolio.balance}, Current Price: {self.current_price}")

    #         if shares > 0 and shares * self.current_price <= self.portfolio.balance:
    #             self.market_env_logging.info(f"Bought {shares} shares at price {self.current_price}")
    #             self.portfolio.balance -= shares * self.current_price
    #             self.portfolio.holdings += shares
    #             self.portfolio.total_shares_bought += shares
    #         else:
    #             # Proper warning for insufficient balance
    #             self.market_env_logging.warning("Invalid number of shares or insufficient balance for buying.")

    #     elif action == 2:  # Sell
    #         if shares > 0 and shares <= self.portfolio.holdings:
    #             self.market_env_logging.info(f"Attempting to sell shares: Requested = {shares}, Holdings = {self.portfolio.holdings}")
    #             self.portfolio.balance += shares * self.current_price
    #             self.portfolio.holdings -= shares
    #             self.portfolio.total_shares_sold += shares
    #         else:
    #             # Proper warning for insufficient holdings
    #             self.market_env_logging.warning("Invalid number of shares or insufficient holdings for selling.")

    #     # Calculate portfolio value and net profit
    #     self.portfolio.portfolio_value = self.portfolio.balance + (self.portfolio.holdings * self.current_price)
    #     self.portfolio.net_profit = self.portfolio.portfolio_value - self.initial_balance

    #     # Reward: Change in portfolio value
    #     reward = float(self.portfolio.net_profit)

    #     # Move to the next time step
    #     self.current_step += 1
    #     done = self.current_step >= len(self.data) - 1  # Check if at the end of data
    #     truncated = False  # Truncation logic, if applicable

    #     # Get the next observation
    #     next_obs = self._next_observation() if not done else None

    #     # Validate observation
    #     if next_obs is not None:
    #         assert next_obs.shape == self.observation_space.shape, \
    #             f"Observation shape mismatch: expected {self.observation_space.shape}, got {next_obs.shape}"
    #         assert next_obs.dtype == self.observation_space.dtype, \
    #             f"Observation dtype mismatch: expected {self.observation_space.dtype}, got {next_obs.dtype}"
    #     # self.market_env_logging.info(f"Step: {self.current_step}"+ \
    #         # f", Current Price: {self.current_price}"+ \
    #         # f", Balance: {self.portfolio.balance}"+ \
    #         # f", Holdings: {self.portfolio.holdings} shares"+ \
    #         # f", Portfolio Value: {self.portfolio.portfolio_value}"+ \
    #         # f", Net Profit: {self.portfolio.net_profit}")

    #     return next_obs, reward, done, truncated, {}
    
    def step(self, action):
        """
        Execute a trade action with the specified number of shares.
        :param action: A tuple where
            - action[0]: Action Type (0 = hold, 1 = buy, 2 = sell)
            - action[1]: Number of Shares to buy or sell
        """
        action_type, shares = action  # Unpack the action
        cant_buy = False
        cant_sell = False
        is_holding = False
        if action_type == 1:  # Buy
            max_affordable_shares = self.portfolio.balance // self.current_price
            shares = min(shares, max_affordable_shares)  # Limit to what can be afforded
            if shares > 0:
                self.market_env_logging.info(f"Bought {shares} shares at price {self.current_price}")
                self.portfolio.balance -= shares * self.current_price
                self.portfolio.holdings += shares
                self.portfolio.total_shares_bought += shares
            else:
                self.market_env_logging.warning("Insufficient balance to buy shares.")
                cant_buy = True
        elif action_type == 2:  # Sell
            shares = min(shares, self.portfolio.holdings)  # Limit to holdings
            if shares > 0:
                self.market_env_logging.info(f"Sold {shares} shares at price {self.current_price}")
                self.portfolio.balance += shares * self.current_price
                self.portfolio.holdings -= shares
                self.portfolio.total_shares_sold += shares
            else:
                self.market_env_logging.warning("Insufficient holdings to sell shares.")
                cant_sell = True
        else:
                self.market_env_logging.warning("HOLDING!!!")
                is_holding = True

        # Calculate portfolio value and net profit
        self.portfolio.portfolio_value = self.portfolio.balance + (self.portfolio.holdings * self.current_price)
        self.portfolio.net_profit = self.portfolio.portfolio_value - self.initial_balance
        # print(self.portfolio.portfolio_value, self.portfolio.net_profit)

        # Brokerage FEE:
        brokerage_fee = 0.5
        if cant_buy or cant_sell or is_holding:
            reward = 0
        else:
            # Reward: Change in portfolio value
            reward = float(self.portfolio.portfolio_value - brokerage_fee)
            # reward = float(self.portfolio.net_profit)

        # Move to the next time step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1  # Check if at the end of data
        truncated = False  # Truncation logic, if applicable

        # Get the next observation
        next_obs = self._next_observation() if not done else None

        return next_obs, reward, done, truncated, {}

    def render(self, render_mode='human', close=False):
        """Render the environment's current state."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Current Price: {self.current_price}, "
                f"Balance: {self.portfolio.balance}, Holdings: {self.portfolio.holdings} shares, "
                f"Portfolio Value: {self.portfolio.portfolio_value}, Net Profit: {self.portfolio.net_profit}")
        else:
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        # self.market_env_logging.info(f"Step: {self.current_step}"+ \
        #     f", Current Price: {self.current_price}"+ \
        #     f", Balance: {self.portfolio.balance}"+ \
        #     f", Holdings: {self.portfolio.holdings} shares"+ \
        #     f", Portfolio Value: {self.portfolio.portfolio_value}"+ \
        #     f", Net Profit: {self.portfolio.net_profit}")
