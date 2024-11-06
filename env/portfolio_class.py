# Portfolio class to manage assets and balances

# Setting up File Configs, loading in necessary files:
import logging
from config.logging_config import *  # Import logging setup function
from data.stock_class import Stocks  # Import the Stocks class

# Import Stocks Class:
from data.stock_class import Stocks

class Portfolio:
    def __init__(self, name: str, initial_balance: float = 1000.0, csv_file: str = ''):
        # Create a logger for this class specifically:
        self.portfolio_logger = logging.getLogger(__name__)  
        # Set propogate to false to separate portfolio logger to be separate from project logger:
        self.portfolio_logger = setup_file_logger(__name__, 'logs/portfolio.log', will_propogate=True)

        # Validate inputs using a helper function
        self._validate_input(isinstance(name, str), "The name must be a string.")
        self._validate_input(name.strip() != '', "The name must not be empty.")
        self._validate_input(isinstance(initial_balance, (float, int)), "The initial balance must be a float or int.")
        self._validate_input(initial_balance > 0, "The initial balance must be greater than zero.")
        self._validate_input(isinstance(csv_file, str), "The csv_file must be a string.")
        
        # Could do this instead: but does not adhere to production code standard: (name.strip() and name.replace(' ', '_').rstrip('_').lstrip('_'))
        self.name = name.replace(' ', '_').rstrip('_').lstrip('_') if name.strip() else name
        self.balance = initial_balance
        self.stocks: List[Stocks] = []

        # Tracking Attributes:
        # Shares held, bought, sold:
        self.holdings = 0 
        self.total_shares_bought = 0
        self.total_shares_sold = 0
          
        # self.load_stocks_from_csv(csv_file)
        self.portfolio_logger.info(f"Portfolio initialized with Owner: {self.name}, balance {self.balance}, and current stocks: {self.stocks}")
        
    def load_stocks_from_csv(self, csv_file: str):
        """ Load stocks from a specified CSV file. """
        self.portfolio_logger.info(f"Loading stocks from {csv_file}...")  # Info level log
        try:
            with open(csv_file, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    buy_price = float(row['buy_price'])
                    sell_price = float(row['sell_price'])
                    volume = int(row['volume'])
                    adjusted_close = float(row['adjusted_close'])
                    stock = Stocks(buy_price, sell_price, volume, adjusted_close)
                    self.stocks.append(stock)
                    self.portfolio_logger.debug(f"Loaded stock: {stock}")  # Debug level log
            self.portfolio_logger.info("Stocks loaded successfully.")  # Info level log
        except FileNotFoundError:
            self.portfolio_logger.error(f"The file {csv_file} does not exist.")  # Error level log
        except KeyError as e:
            self.portfolio_logger.error(f"Missing expected column in CSV: {e}")  # Error level log
        except ValueError as e:
            self.portfolio_logger.error(f"Data conversion error: {e}")  # Error level log

    # def add_stock(self, stock: Stocks):
    #     """ Add a stock to the portfolio if there's enough balance. """
    #     total_cost = stock.buy_price * stock.volume
    #     if total_cost <= self.balance:
    #         self.stocks.append(stock)
    #         self.balance -= total_cost
    #         self.portfolio_logger.info(f"Added {stock} to the portfolio.")  # Info level log
    #     else:
    #         self.portfolio_logger.warning("Insufficient balance to add this stock.")  # Warning level log

    def add_stock(self, stock: Stocks):
        """Add a stock to the portfolio if there's enough balance."""
        total_cost = stock.buy_price * stock.volume
        if total_cost <= self.balance:
            self.stocks.append(stock)
            self.balance -= total_cost
            self.holdings += stock.volume
            self.total_shares_bought += stock.volume  # Update total shares bought
            # Logging for successful stock addition
            self.portfolio_logger.info(f"Added stock to the portfolio")
        else:
            # Logging for insufficient balance
            self.portfolio_logger.warning(f"Insufficient balance to add this stock")
    
    # def sell_stock(self, stock: Stocks, volume: int):
    #     """Sell a stock from the portfolio."""
    #     if stock in self.stocks and volume <= self.holdings:
    #         self.balance += stock.sell_price * volume
    #         self.holdings -= volume
    #         self.total_shares_sold += volume  # Update total shares sold
            
    #         # Update the volume of the stock being sold
    #         stock.volume -= volume
    #         if stock.volume <= 0:
    #             self.stocks.remove(stock)
            
    #         self.portfolio_logger.info(f"Sold {volume} shares of {stock} from the portfolio.")
    #     else:
    #         self.portfolio_logger.warning(f"Stock not found or invalid volume.")

    def sell_stock(self, stock: Stocks, volume: int):
        """Sell a stock from the portfolio."""
        if stock in self.stocks and volume <= stock.volume:
            self.balance += stock.sell_price * volume
            stock.volume -= volume
            self.total_shares_sold += volume  # Update total shares sold

            # Remove the stock if all shares are sold
            if stock.volume <= 0:
                self.stocks.remove(stock)

            # Recalculate holdings to reflect the total shares across all stocks
            self.holdings = sum(s.volume for s in self.stocks)

            self.portfolio_logger.info(f"Sold {volume} shares of {stock} from the portfolio.")
        else:
            self.portfolio_logger.warning(f"Stock not found or invalid volume.")



    def __repr__(self):
        return f"Portfolio(Owner: {self.name}, Balance: {self.balance}, Stocks: {self.stocks})"
    
    def _validate_input(self, condition: bool, error_message: str):
        """Helper method to validate inputs, log errors, and raise exceptions."""
        if not condition:
            self.portfolio_logger.error(error_message)
            raise ValueError(error_message)