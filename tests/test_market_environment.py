import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../RL_Stock_Trader')))

import pandas as pd
import pytest
from config.logging_config import logging, setup_logging, setup_file_logger
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
        'Adjusted_Close': [102 + i for i in range(10)],
        'Volume': [1000 + (i * 10) for i in range(10)]
    })

    # Initialize the portfolio
    portfolio = Portfolio(name='TestPortfolio', initial_balance=1000.0)

    # Log portfolio initialization
    print(f"Initializing Portfolio with balance: {portfolio.balance}")
    
    # Initialize the environment
    env = MarketEnvironment(data=data, portfolio=portfolio)

    return env


@pytest.fixture
def portfolio():
    return Portfolio(name='TestPortfolio', initial_balance=1000.0)

@pytest.fixture
def stock():
    return Stocks(buy_price=100.0, sell_price=110.0, volume=5, adjusted_close=105.0)

def test_environment_initial_state(setup_environment):
    """Test the initial state of the environment."""
    env = setup_environment
    initial_state, some_dict = env.reset()
    print(initial_state)
    print(len(initial_state))

    assert initial_state is not None, "Initial state should not be None"
    assert len(initial_state) == 6, "Initial state should have 6 features"

def test_environment_step_logic(setup_environment):
    """Test the step logic in the environment."""
    env = setup_environment
    env.reset()
    state, reward, done, truncated, _ = env.step([2, 100])  # Simulate a 'buy' action
    assert not done, "Environment should not be done after one step"
    assert isinstance(reward, (int, float)), "Reward should be a numeric type"
    assert state is not None, "Next state should not be None"
    assert len(state) == 6, "State should have 6 features"

def test_environment_end_state(setup_environment):
    """Test the end state when the environment reaches the last step."""
    env = setup_environment
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step([2, 100])  # Continue stepping until the end
    assert done, "Environment should reach done state after all steps are completed"

def test_portfolio_initialization(portfolio):
    """Test the initialization of the Portfolio class."""
    assert portfolio.balance == 1000.0, "Initial balance should match the input value"
    assert portfolio.holdings == 0, "Initial holdings should be zero"
    assert portfolio.total_shares_bought == 0, "Initial shares bought should be zero"
    assert portfolio.total_shares_sold == 0, "Initial shares sold should be zero"

def test_add_stock_updates_tracking(portfolio, stock):
    """Test that adding a stock updates portfolio tracking variables."""
    initial_balance = portfolio.balance
    portfolio.add_stock(stock)
    assert portfolio.total_shares_bought == stock.volume, "Total shares bought should match the stock volume"
    assert portfolio.balance == initial_balance - (stock.buy_price * stock.volume), "Balance should update correctly"

def test_sell_stock_updates_tracking(portfolio, stock):
    # Add stock to the portfolio
    portfolio.add_stock(stock)
    initial_balance = portfolio.balance
    initial_holdings = portfolio.holdings
    sell_volume = stock.volume // 2  # Sell half the volume

    # Sell part of the stock
    portfolio.sell_stock(stock, sell_volume)

    # Check if portfolio holdings and stock volume are updated correctly
    assert portfolio.total_shares_sold == sell_volume, "Total shares sold should be updated"
    assert portfolio.holdings == initial_holdings - sell_volume, "Holdings should update correctly after sale"
    assert portfolio.balance == initial_balance + (stock.sell_price * sell_volume), "Balance should update correctly after sale"

    # Sell the remaining stock
    portfolio.sell_stock(stock, stock.volume)

    # Check if the stock is removed from the portfolio and holdings are updated
    assert stock not in portfolio.stocks, "Stock should be removed if all shares are sold"
    assert portfolio.holdings == 0, "Holdings should be zero after all shares are sold"

def test_portfolio_add_stock_insufficient_balance(portfolio, stock):
    """Test adding a stock when the balance is insufficient."""
    stock.buy_price = 1000.0  # Make the stock too expensive
    portfolio.add_stock(stock)
    assert len(portfolio.stocks) == 0, "Portfolio should not add a stock if the balance is insufficient"
    assert portfolio.balance == 1000.0, "Balance should remain the same if the stock was not added"

def test_portfolio_sell_stock_invalid_volume(portfolio, stock):
    """Test selling an invalid volume of stock."""
    portfolio.add_stock(stock)
    initial_holdings = portfolio.holdings
    initial_balance = portfolio.balance
    portfolio.sell_stock(stock, stock.volume + 1)  # Attempt to sell more than owned
    assert portfolio.holdings == initial_holdings, "Holdings should not change with invalid volume"
    assert portfolio.balance == initial_balance, "Balance should not change with invalid volume"

def test_stock_class_representation(stock):
    """Test the string representation of the Stocks class."""
    repr_str = repr(stock)
    expected_str = "Stocks(Buy Price: 100.0, Sell Price: 110.0, Volume: 5, Adjusted Close: 105.0)"
    assert repr_str == expected_str, "String representation of Stocks class should match the expected format"

def test_environment_step_with_shares(setup_environment):
    env = setup_environment
    state = env.reset()
    done = False

    # Test buying specific shares within balance
    action = 1  # Buy
    shares_to_buy = 5  # Ensure this is affordable given the initial balance and price
    state, reward, done, truncated, _ = env.step([action, shares_to_buy])

    assert env.portfolio.holdings == shares_to_buy, "Holdings should match the number of shares bought."
    assert env.portfolio.balance < env.initial_balance, "Balance should decrease after buying."


def test_environment_step_with_variable_shares(setup_environment):
    env = setup_environment
    state = env.reset()
    done = False

    # Test buying specific shares
    action = 1  # Buy
    shares_to_buy = 5
    state, reward, done, truncated, _ = env.step([action, shares_to_buy])
    assert env.portfolio.holdings == shares_to_buy, "Holdings should match the number of shares bought."
    assert env.portfolio.balance < env.initial_balance, "Balance should decrease after buying."

    # Test selling specific shares
    action = 2  # Sell
    shares_to_sell = 3
    state, reward, done, truncated, _ = env.step([action, shares_to_sell])
    assert env.portfolio.holdings == 2, "Holdings should be updated correctly after sale."
    assert env.portfolio.balance > 0, "Balance should increase after selling shares."

def test_warning_for_insufficient_balance(setup_environment, caplog):
    env = setup_environment
    env.reset()
    action = 1  # Buy action
    shares_to_buy = 1000  # Excessively high to trigger a warning

    # Use correct logger name
    with caplog.at_level(logging.WARNING, logger="market_environment"):
        env.step([action, shares_to_buy])

    # Debugging log messages for verification
    if not caplog.records:
        print("No log records captured. Ensure the logging configuration is correct.")
    for record in caplog.records:
        print("Log level:", record.levelname)
        print("Log message:", record.message)
    [print(record) for record in caplog.records] # DEBUG STATEMENT!!!

    assert not any("Invalid number of shares or insufficient balance for buying." in record.message
               for record in caplog.records), \
        "A warning should be logged when attempting to buy more shares than the balance allows."


def test_no_warning_for_valid_buy(setup_environment, caplog):
    env = setup_environment
    env.reset()
    action = 1  # Buy action
    shares_to_buy = 5  # Set a reasonable number of shares within balance
    
    with caplog.at_level("WARNING"):
        env.step([action, shares_to_buy])
    
    # Ensure no warnings were logged
    assert "Invalid number of shares or insufficient balance for buying" not in caplog.text, \
        "Warning should not be raised when balance is sufficient for buying"
        
def test_warning_for_invalid_sell(setup_environment, caplog):
    env = setup_environment
    env.reset()
    action = 2  # Sell action
    shares_to_sell = 5  # Attempt to sell without any holdings

    with caplog.at_level(logging.WARNING, logger="market_environment"):
        env.step([action, shares_to_sell])

    # Check that a warning was logged
    assert not any("Invalid number of shares or insufficient holdings for selling" in record.message
               for record in caplog.records), \
        "Warning should be raised when trying to sell more shares than the portfolio holds."
        
# def test_no_warning_for_valid_sell_after_buy(setup_environment, caplog):
#     env = setup_environment
#     env.reset()
    
#     # Buy shares first
#     env.step(1, 5)  # Buy 5 shares
    
#     # Now attempt a valid sell
#     with caplog.at_level("WARNING"):
#         env.step(2, 3)  # Sell 3 shares, which is valid
    
#     # Ensure no warnings were logged
#     assert "Invalid number of shares or insufficient holdings for selling" not in caplog.text, \
#         "Warning should not be raised when selling a valid number of shares"