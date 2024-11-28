# Logging and utility setup

# General Imports
import os
import sys
import time
import math
import threading
import logging  # For logging and decorators
import colorlog  # For coloring the logs
import functools  # For logging and decorators
from tqdm import tqdm  # Progress bar
from typing import List, Tuple  # Format typing
import argparse   # For Parsing arguments
from termcolor import cprint  # Formatting prints
import pickle  # For saving intermediate values to avoid recomputing
sys.path.append('../')
# Useful DS, ML, RL, CV Libraries, and others
import cv2
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Ensure that a directory exists at the specified path
def ensure_directory(path: str) -> None:
    if not isinstance(path, str):
        raise ValueError("The path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
        logging.debug(f"Directory created at: {path}", "green")
    else:
        logging.debug(f"Directory already exists at: {path}", "yellow")
    return

def time_decorator(func):
    """Decorator to time the execution of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.debug(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper

def interactive_countdown(seconds: int =60, message : str ="You want to clear all the log files? ? (Type 'yes' to confirm or 'q' to quit, then press ENTER)"):
    """Decorator to pause and allow interactive input during a countdown before executing a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_input = []

            def get_input():
                """Thread function to get user input."""
                response = input(f"\n{message} ").strip().lower()
                user_input.append(response)
            
            # Start the input thread
            input_thread = threading.Thread(target=get_input)
            input_thread.start()

            # Display countdown
            for remaining in range(seconds, 0, -1):
                if input_thread.is_alive():
                    if remaining == seconds:
                        print(f"\r{remaining} seconds till operation is cancelled...\n", end='', flush=True)
                        # print()
                    elif remaining%(0.25*seconds) == 0:
                        print(f"\r{remaining} seconds till operation is cancelled...\n", end='', flush=True)
                    time.sleep(1)
                else:
                    break  # Exit countdown if input is provided
            
            input_thread.join(timeout=1)  # Ensure the thread completes or times out
            
            # Check user input
            if user_input and user_input[0] == 'yes':
                cprint(f"\nConfirmation received. Proceeding...", "white")
                return func(*args, **kwargs)
            elif user_input and user_input[0] == 'q':
                cprint(f"\nOperation cancelled by the user.", "white")
                sys.exit(0)  # Exit the program
            else:
                cprint(f"\nTime's up or no valid input received. Proceeding automatically...", "white")
                return func(*args, **kwargs)

        return wrapper
    return decorator

def log_decorator(enabled=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if enabled:
                logging.info(f"Calling function {func.__name__} with arguments {args} and keyword arguments {kwargs}")
            result = func(*args, **kwargs)
            if enabled:
                logging.info(f"{func.__name__} returned {result}")
            return result
        return wrapper
    return decorator

# Ensure the log directory exists
# ensure_directory("logs/")

log_decorator(enabled=True)
def setup_logging(log_level=logging.INFO):
    """Sets up the logging configuration with colorlog."""
    # Define the log format with the function name included
    log_format = (
        '%(log_color)s%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s'
    )
    color_formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'bold_cyan',
            'INFO': 'light_white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # Set up the logging handler for the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    console_handler.setLevel(log_level)  # Set the log level for the console

    # Set up the logging handler for the file
    file_handler = logging.FileHandler('logs/project.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
    ))
    file_handler.setLevel(log_level)  # Set the log level for the file

    # Get the root logger and configure it
    logger = logging.getLogger()
    logger.handlers.clear()  # Ensure no duplicate handlers
    logger.setLevel(log_level)  # Set the root logger level
    logger.handlers = []  # Clear existing handlers to avoid duplicate logs
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    cprint("Colored logging setup complete.", "green")

def setup_file_logger(name: str, log_file: str, log_level=logging.DEBUG, will_propogate: bool = False):
    """Sets up a dedicated file logger for a specific class or module."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create a file handler if it doesn't exist
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger if it hasn't been added
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in logger.handlers):
        logger.addHandler(file_handler)

    # Prevent logging to propagate to root or not:
    logger.propagate = will_propogate   
    return logger


class ClearLogsAction(argparse.Action):
    @interactive_countdown(60)  # Countdown for 60 seconds
    def __call__(self, parser, namespace, values, option_string=None):
        log_directory = "logs/"

        # Ask for user confirmation
        user_response = input("Are you sure you want to clear all logs? Type 'yes' to confirm: ").strip().lower()
        if user_response != 'yes':
            print("Log clearing aborted.")
            setattr(namespace, self.dest, False)
            return

        # Clear the logs if confirmation was given
        if os.path.exists(log_directory):
            for filename in os.listdir(log_directory):
                file_path = os.path.join(log_directory, filename)
                try:
                    if os.path.isfile(file_path):
                        with open(file_path, 'w') as log_file:
                            log_file.truncate(0)  # Clear the contents of the log file
                        print(f"Cleared log file: {file_path}")
                except Exception as e:
                    print(f"Failed to clear {file_path}: {e}")
        else:
            print(f"Log directory '{log_directory}' does not exist.")

        setattr(namespace, self.dest, True)