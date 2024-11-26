# Module for loading and preprocessing data
import os
import pandas as pd

from config.logging_config import logging, setup_logging, setup_file_logger

class DataPreprocessor:
    def __init__(self, default_raw_csv_folder: str = "data/raw/", default_csv_name: str ='yahoo_data.csv'):
        self.stock_name = default_csv_name[:-4] 
        self.default_raw_csv_folder = default_raw_csv_folder
        self.default_csv_path = os.path.join(self.default_raw_csv_folder, default_csv_name)
        self.default_required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume']
        # Private attributes for dataset:
        self._dataset = None  

    @property
    def dataset(self) -> pd.DataFrame:
        """Getter for the dataset."""
        return self._dataset

    @dataset.setter
    def dataset(self, data: pd.DataFrame) -> None:
        """Setter for the dataset with validation."""
        if not isinstance(data, pd.DataFrame):
            logging.error("Attempted to set dataset with an invalid type. Must be a pandas DataFrame.")
            raise TypeError("Dataset must be a pandas DataFrame.")
        
        if data.empty:
            logging.warning("The dataset being set is empty.")
        else:
            logging.info("Dataset has been set successfully.")
        self._dataset = data
        
    def load_csv(self, csv_path: (None | str) = None, required_columns: (None | list) = None) -> pd.DataFrame:
        """
        Load and preprocess CSV data from the specified path or the default path.
    
        Args:
            csv_path (Optional[str]): Path to the CSV file. Defaults to the pre-set path. If None, the default path is used.
            required_columns (Optional[List[str]]): List of required columns for processing. If None, defaults to a common set of columns.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame with columns specified by required_columns if present.
        """
        path = csv_path if csv_path else self.default_csv_path
        required_columns = required_columns if required_columns else self.default_required_columns
        
        try:
            if not os.path.isfile(path):
                logging.debug("Function assumes the csv file is in data/raw!!!")
                logging.debug("If you want to change this then, pass in folder path in main!!!")
                raise FileNotFoundError(f"CSV file '{path}' does not exist.")

            # Load the CSV using pandas
            data = pd.read_csv(path)
            logging.info(f"Loaded CSV from {path}, Now processing...")

            # Strip any leading/trailing whitespace from column names and replace spaces with underscores
            data.columns = data.columns.str.strip('*').str.replace(' ', '_')
            logging.debug(f"Modified column names: {[col for col in data.columns]}")

            # Check if required columns are in the data, and only select columns that exist
            available_columns = [col for col in required_columns if col in data.columns]
            if not available_columns:
                logging.warning(f"None of the specified required columns found in the CSV.")
                raise ValueError(f"The CSV file does not contain any of the required columns: {', '.join(required_columns)}")

            # Select only the available required columns, ignoring any extra columns
            data = data[available_columns]
            logging.info(f"Selected columns for processing: {available_columns}")

            # Convert 'Date' column to datetime format if it exists
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Convert other columns to numeric, if present
            for col in available_columns:
                if col != 'Date':  # Skip the 'Date' column for numerical conversion
                    data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')


            # Log the number of rows before dropping NA
            initial_row_count = data.shape[0]
            # Drop rows with missing or invalid data
            data = data.dropna()
            final_row_count = data.shape[0]
            rows_dropped = initial_row_count - final_row_count
            logging.debug(f" # of rows before dropping NA: {initial_row_count} and after: {initial_row_count - rows_dropped} i.e rows dropped: {rows_dropped}")
            

            # Use the property setter to assign the dataset
            self.dataset = data
            
            logging.info("CSV data successfully loaded and preprocessed.")
            return data

        except Exception as e:
            logging.error(f"Error during CSV loading or preprocessing: {e}")
            # Re-raise the exception so it can be caught in the caller if needed (propogate the exception through the call stack)
            raise  
    def log_csv_head(self):
        """Logs the first few rows of the dataset."""
        if not self.dataset.empty:
            logging.debug("\nFirst few rows of the dataset:\n" + str(self.dataset.head()) +"\n")
        else:
            logging.warning("No dataset available. Please load a CSV first.")
        return

    def log_dataset_metrics(self):
        """Logs basic metrics and statistics of the dataset."""
        if not self.dataset.empty:
            logging.debug(f"\nDataset Metrics:\nNumber of rows and colums: {(self.dataset.shape[0], self.dataset.shape[1])}\nData Types:\n" + str(self.dataset.dtypes)+"\n")
            logging.debug("\nSummary Statistics:\n" + str(self.dataset.describe()))
        else:
            logging.warning("No dataset available. Please load a CSV first.")
        return