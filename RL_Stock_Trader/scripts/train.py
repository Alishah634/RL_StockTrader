# Script for training the RL agent

from config.logging_config import logging, setup_logging, cprint, List
from data.data_loader import DataPreprocessor
from env.portfolio_class import Portfolio
from config.rl_config import RL_SETTINGS
# from env.market_environment import MarketEnvironment
# from agents.base_agent import BaseAgent  # Assuming a base agent class exists for RL


def train(episodes: int = RL_SETTINGS["episodes"], learning_rate: float = RL_SETTINGS["learning_rate"], csv_path: (None | str) = None, required_columns: (None | List[str]) = None) -> None:
    logging.debug("Training started...")
    
    # Compact assertions
    assert isinstance(episodes, int) and episodes > 0, "Episodes must be an integer greater than zero."
    assert isinstance(learning_rate, (float, int)) and 0 < learning_rate <= 1, "Learning rate must be a float or int between 0 and 1."
    assert (csv_path is not None) or (isinstance(csv_path, str) and (csv_path.strip() != '')), f"CSV path was type {type(csv_path)} and must be a non-empty string or None."
    assert required_columns is None or (isinstance(required_columns, list) and all(isinstance(col, str) for col in required_columns)), "Required columns must be a list of strings or None."
    logging.debug(f"All input parameters are valid. Proceeding with training...")

    # Placeholder for training code
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    try:
        # Load and preprocess data from the specified CSV path
        processed_data = preprocessor.load_csv(csv_path, required_columns)
        preprocessor.log_csv_head()
        preprocessor.log_dataset_metrics()
        logging.debug("CSV data successfully loaded and preprocessed.")
        # logging.info("CSV data successfully loaded and preprocessed")
        # cprint("Data successfully preprocessed.", "green")
    except Exception as e:
        logging.error(f"Failed to load or preprocess CSV data: {e}")
        return

    # Initialize portfolio:
    portfolio = Portfolio('John')
    
    
    
    '''
    Mothodology/Steps: 
    Set up:
        - Pre-process the stocks in data/
        - Then load them into the Stocks class,, add them to a dictionary for easy look up
        - Then load the data from the dictionary into the Market env, this would simulate the markt data, whose first index is at time step t = 0

        - (Optional): Could calculate more metrics, which can be added to the Stock Class, for better informational data. Would need to update the market Env.

        - Since the portfolio represents the state at a time t = 0, we need to intialize the protfolio with no stocks, and an initial balance in (Dollars) > 0.
     

    
    Most likely then pipe them to the RL models ???
    
    '''
    

if __name__ == '__main__':
    train(episodes= RL_SETTINGS["episodes"], learning_rate= RL_SETTINGS["learning_rate"]) 