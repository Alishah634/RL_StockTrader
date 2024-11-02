# Entry point for the RL Stock Trader project

# Load up the initial configuration file (Debugging and useful imports)
# General Imports
import os, sys
from config.logging_config import setup_logging, logging, ClearLogsAction, cprint
# File imports:
from scripts.train import train
from scripts.evaluate import evaluate
from scripts.simulate import simulate
from argparse import ArgumentParser

def parse_arguments():
    """Parse command-line arguments for configuring the RL Stock Trader."""
    parser = ArgumentParser(
        description="Run the RL Stock Trader project with configurable settings.",
        epilog="Example usage: python main.py --task train --episodes 500 --log-level DEBUG"
    )

    # Task selection: Train, Evaluate, Simulate
    parser.add_argument(
        '--task', '-t', type=str, required=True, choices=['train', 'evaluate', 'simulate'],
        help="Select the task to run: 'train' for training, 'evaluate' for evaluation, or 'simulate' for running a simulation."
    )

    # Add options for training
    parser.add_argument(
        '--episodes', '-e', type=int, default=100,
        help="Number of episodes for training or evaluation (default: 100)."
    )
    parser.add_argument(
        '--learning-rate', '-lr', type=float, default=0.001,
        help="Learning rate for the RL model (default: 0.001)."
    )

    # Environment settings
    parser.add_argument(
        '--environment', '-env', type=str, default='market_environment',
        help="Specify the environment configuration to use (default: 'market_environment')."
    )

    # Logging level
    parser.add_argument(
        '--log-level', '-l', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level for the session (default: 'INFO')."
    )

    # Dry run option
    parser.add_argument(
        '--dry-run', action='store_true',
        help="If specified, the script will set up the environment and configurations but not execute any tasks."
    )
    
    # Add the clear-logs argument
    parser.add_argument(
        '--clear-logs', action=ClearLogsAction, nargs=0,
        help="Clear all logs in the 'logs/' directory before starting the task."
    )
    
    parser.add_argument(
        '--load-csv', type=str, default='', 
        help="Specify the CSV file for evaluation or training. The file name should not contain spaces."
    )

    setup_logging()
    
    # Parse arguments
    try:
        # Parse arguments
        args = parser.parse_args()
    except SystemExit as e:
        logging.error("Invalid task choice. Please select 'train', 'evaluate', or 'simulate'.")
        # Re-raise the exception to allow the program to exit as usual, ensuring that the error is logged but still terminates the script.
        raise e  
    
    return args

def main():
    # Parse arguments
    args = parse_arguments()

    # Initialize logging with the specified log level
    setup_logging(getattr(logging, args.log_level))

    # Ensure you're in the correct directory
    cprint("Make sure you are running from the RL_STOCK_TRADER directory!!!\n", "red")

    # Display chosen configuration
    cprint(f"===================================", "white")
    cprint(f"Chosen Configuration:", "white")
    cprint(f"Selected task: {args.task}", "light_magenta")
    cprint(f"Environment: {args.environment}", "light_magenta")
    cprint(f"Episodes: {args.episodes}", "light_magenta")
    cprint(f"Learning rate: {args.learning_rate}", "light_magenta")
    cprint(f"Log level: {args.log_level}", "light_magenta")
    cprint(f"===================================\n", "white")
    
    # Execute the task based on the argument provided
    if args == None:
        cprint(f"Did not run, either logs were cleared or a failure occured!!!", "red")
    
    if args.dry_run:
        cprint("Dry run selected. Configuration complete, but no task will be executed.", "yellow")
        return
    
    if args.load_csv:
        # Validate that the CSV file name does not contain spaces
        if ' ' in args.load_csv:
            logging.error(f"File name has spaces in it, please remove spaces then load the csv. {args.load_csv}")
            args.load_csv = args.load_csv.replace(' ', '_')
            logging.debug(f"Example: Valid file name: correct_file_name.csv")
            return
        # # Additional check to ensure the file exists and is accessible
        elif not os.path.isfile(args.load_csv):
            logging.error(f"Error: The specified CSV file '{args.load_csv}' does not exist.")
            logging.debug(f"Ensure file name ends with \'.csv\'. For example: Valid file name: correct_file_name.csv")
            return
        
    if args.task == 'train':
        train(episodes=args.episodes, learning_rate=args.learning_rate, csv_path=args.load_csv)
        # train(episodes=args.episodes, learning_rate=args.learning_rate, csv_path=args.load_csv)
    elif args.task == 'evaluate':
        evaluate(episodes=args.episodes)
    elif args.task == 'simulate':
        simulate()
    else:
        # Possibly redundant check as correct task choices are validated in parse_arguments()!!!
        logging.error("Invalid task choice. Please select 'train', 'evaluate', or 'simulate'.")

if __name__ == "__main__":
    main()
