# Market environment setup
from config.logging_config import setup_logging, setup_file_logger
from config.rl_config import RL_SETTINGS
from env.action_space import ActionSpace

import gym
from gym import spaces
import numpy as np
import pandas as pd
from env.portfolio_class import Portfolio

class MarketEnvironment(gym.Env):
    """A custom trading environment for Reinforcement Learning with stock market data."""
    """Allows the agent to buy and sell stocks based on the current price, and rewards the agent based on the portfolio value."""

    metadata = {'render.modes': ['human']}
    
    def __init__(self, data: pd.DataFrame, portfolio: Portfolio, initial_balance: float = 10000):
        super(MarketEnvironment, self).__init__()
        
        # Create a logger for this class specifically(will NOT propogate to root logger):
        self.market_env_logging = setup_file_logger(__name__, 'logs/market_environment.log', will_propogate=False)
        
        # Validate that required columns are in the dataset
        required_columns = {'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        # Initialize attributes
        self.data = data.reset_index(drop=True)
        self.portfolio = portfolio
        self.initial_balance = initial_balance
        self.current_step = 0
        self.current_price = 0

        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Define observation space (Open, High, Low, Close, Adj_Close, Volume)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
    
    def reset(self):
        """Reset the environment to an initial state."""
        self.current_step = 0
        self.portfolio.balance = self.initial_balance
        self.portfolio.holdings = 0
        self.portfolio.net_profit = 0
        return self._next_observation()
    
    def _next_observation(self):
        """Get the next state/observation."""
        obs = self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']].values
        self.current_price = self.data.iloc[self.current_step]['Close']
        return obs
    
    def step(self, action, shares=0):
        """Execute a trade action and calculate the next state."""
        if action == 1:  # Buy
            if shares > 0 and shares * self.current_price <= self.portfolio.balance:
                self.portfolio.balance -= shares * self.current_price
                self.portfolio.holdings += shares
                self.portfolio.total_shares_bought += shares
            else:
                self.market_env_logging.warning("Invalid number of shares or insufficient balance for buying.")

        elif action == 2:  # Sell
            if shares > 0 and shares <= self.portfolio.holdings:
                self.portfolio.balance += shares * self.current_price
                self.portfolio.holdings -= shares
                self.portfolio.total_shares_sold += shares

                # Remove stocks from the portfolio if all shares are sold
                if self.portfolio.holdings == 0:
                    self.portfolio.stocks.clear()  # This simulates clearing out stocks after full sale

            else:
                self.market_env_logging.warning("Invalid number of shares or insufficient holdings for selling.")

        # Calculate portfolio value
        self.portfolio.portfolio_value = self.portfolio.balance + (self.portfolio.holdings * self.current_price)
        self.portfolio.net_profit = self.portfolio.portfolio_value - self.initial_balance

        # Reward: Change in portfolio value
        reward = float(self.portfolio.net_profit)

        # Move to the next time step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1  # Check if at the end of data

        # Next observation/state
        next_obs = self._next_observation() if not done else None

        return next_obs, reward, done, {}



    def render(self, mode='human', close=False):
        """Render the environment's current state."""
        self.market_env_logging.info(f"Step: {self.current_step}")
        self.market_env_logging.info(f"Current Price: {self.current_price}")
        self.market_env_logging.info(f"Balance: {self.portfolio.balance}")
        self.market_env_logging.info(f"Holdings: {self.portfolio.holdings} shares")
        self.market_env_logging.info(f"Portfolio Value: {self.portfolio.portfolio_value}")
        self.market_env_logging.info(f"Net Profit: {self.portfolio.net_profit}")
