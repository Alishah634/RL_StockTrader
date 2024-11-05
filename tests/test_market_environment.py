import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../RL_Stock_Trader')))

import pandas as pd
import pytest
from env.market_environment import MarketEnvironment
from env.portfolio_class import Portfolio
from data.stock_class import Stocks  # Assuming Stocks is imported from data/stock_class.py

@pytest.fixture
def setup_environment():
    # Create a small synthetic dataset for testing
    data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Open': [100 + i for i in range(10)],
        'High': [105 + i for i in range(10)],
        'Low': [95 + i for i in range(10)],
        'Close': [102 + i for i in range(10)],
        'Adj_Close': [102 + i for i in range(10)],
        'Volume': [1000 + (i * 10) for i in range(10)]
    })

    # Initialize the portfolio
    portfolio = Portfolio(name='TestPortfolio', initial_balance=1000.0)

    # Initialize the environment
    env = MarketEnvironment(data=data, portfolio=portfolio)

    return env

@pytest.fixture
def portfolio():
    return Portfolio(name='TestPortfolio', initial_balance=1000.0)

@pytest.fixture
def stock():
    return Stocks(buy_price=100.0, sell_price=110.0, volume=5, adjusted_close=105.0)

def test_environment_steps(setup_environment):
    env = setup_environment
    state = env.reset()
    done = False
    actions = ['Hold', 'Buy', 'Sell']
    
    while not done:
        action = env.action_space.sample()  # Random action for testing
        next_state, reward, done, _ = env.step(action)
        
        # Assertions to check environment behavior
        if not done:
            assert next_state is not None, "Next state should not be None"
        assert isinstance(reward, (int, float)), "Reward should be a numeric type"
        
        # Print action and state for manual verification (optional)
        print(f"Action taken: {actions[action]}")
        print(f"Reward: {reward}")
        print(f"Next State: {next_state}\n")
    
    assert done, "Environment should reach done state after last step"


def test_initialization(setup_environment):
    env = setup_environment
    assert env is not None, "Environment should be initialized"

def test_add_stock_updates_tracking(portfolio, stock):
    initial_balance = portfolio.balance
    portfolio.add_stock(stock)
    assert portfolio.total_shares_bought == stock.volume
    assert portfolio.balance == initial_balance - (stock.buy_price * stock.volume)

def test_sell_stock_updates_tracking(portfolio, stock):
    portfolio.add_stock(stock)
    initial_balance = portfolio.balance
    sell_volume = stock.volume // 2
    portfolio.sell_stock(stock, sell_volume)
    assert portfolio.total_shares_sold == sell_volume
    assert portfolio.holdings == stock.volume  
    assert portfolio.balance == initial_balance + (stock.sell_price * sell_volume)

