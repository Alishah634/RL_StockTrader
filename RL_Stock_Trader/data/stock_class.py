# Class representing individual stock data
from config.logging_config import logging, setup_logging, setup_file_logger, cprint

class Stocks:
    def __init__(self, buy_price: float, sell_price: float, volume: int, adjusted_close: float):
        # self.logger = logging.getLogger(__name__)  # Create a logger for this class
        # self.logger = setup_file_logger(__name__, 'logs/portfolio.log', will_propogate=True)

        self.buy_price = buy_price
        self.sell_price = sell_price
        self.volume = volume
        self.adjusted_close = adjusted_close

    def __repr__(self):
        return f"Stocks(Buy Price: {self.buy_price}, Sell Price: {self.sell_price}, Volume: {self.volume}, Adjusted Close: {self.adjusted_close})"
    
    