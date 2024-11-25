import gym
from gym import spaces
import numpy as np
import pandas as pd
from env.portfolio_class import Portfolio
from config.logging_config import setup_file_logger
import logging  # Standard library logging

class MarketEnvironment(gym.Env):
    """A custom trading environment for Reinforcement Learning with stock market data."""

    metadata = {'render.modes': ['human']}
    def __init__(self, data: pd.DataFrame, portfolio: Portfolio, initial_balance: float = 1000):
        super(MarketEnvironment, self).__init__()

        # Create a logger specifically for the market environment
        self.market_env_logging = logging.getLogger(__name__)
        setup_file_logger(__name__, 'logs/market_environment.log', will_propogate=True)

        # Initialize attributes
        required_columns = {'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

        self.data = data.reset_index(drop=True)
        self.portfolio = portfolio
        self.initial_balance = portfolio.initial_balance
        self.current_step = 0
        self.current_price = 0

        # Log or print initial balance
        self.market_env_logging.info(f"Initialized Market Environment with Portfolio Balance: {self.portfolio.balance} and Initial Balance: {self.initial_balance}")

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

        # Log reset state
        self.market_env_logging.info(f"Environment Reset: Balance: {self.portfolio.balance}, Holdings: {self.portfolio.holdings}, Net Profit: {self.portfolio.net_profit}")

        return self._next_observation()

    def _next_observation(self):
        """Get the next state/observation."""
        obs = self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']].values
        self.current_price = self.data.iloc[self.current_step]['Close']
        return obs

    def step(self, action, shares=0):
        """Execute a trade action and calculate the next state."""
        if action == 1:  # Buy
            max_affordable_shares = self.portfolio.balance // self.current_price
            # shares = min(shares, max_affordable_shares) # TODO This line lets the agent buy as much as possible!!!

            # Debugging information to understand the current state of the transaction:
            self.market_env_logging.debug(f"Attempting to buy shares: Requested = {shares}, Affordable = {max_affordable_shares}")
            self.market_env_logging.debug(f"Current Balance: {self.portfolio.balance}, Current Price: {self.current_price}")

            if shares > 0 and shares * self.current_price <= self.portfolio.balance:
                self.market_env_logging.info(f"Bought {shares} shares at price {self.current_price}")
                self.portfolio.balance -= shares * self.current_price
                self.portfolio.holdings += shares
                self.portfolio.total_shares_bought += shares
            else:
                # Proper warning for insufficient balance
                self.market_env_logging.warning("Invalid number of shares or insufficient balance for buying.")

        elif action == 2:  # Sell
            if shares > 0 and shares <= self.portfolio.holdings:
                self.market_env_logging.info(f"Attempting to sell shares: Requested = {shares}, Holdings = {self.portfolio.holdings}")
                self.portfolio.balance += shares * self.current_price
                self.portfolio.holdings -= shares
                self.portfolio.total_shares_sold += shares
            else:
                # Proper warning for insufficient holdings
                self.market_env_logging.warning("Invalid number of shares or insufficient holdings for selling.")

        # Calculate portfolio value and net profit
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
        self.market_env_logging.info(f"Step: {self.current_step}"+ \
            f", Current Price: {self.current_price}"+ \
            f", Balance: {self.portfolio.balance}"+ \
            f", Holdings: {self.portfolio.holdings} shares"+ \
            f", Portfolio Value: {self.portfolio.portfolio_value}"+ \
            f", Net Profit: {self.portfolio.net_profit}")
