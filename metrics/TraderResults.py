import os
import sys
import pandas as pd
from tqdm import tqdm
import time
import re
import matplotlib.pyplot as plt 
from typing import List, Tuple
import numpy as np

ROOT = os.getenv('PROJECT_ROOT', "/home/shantanu/RL_Proj/RL_StockTrader")
sys.path.append(ROOT)


    
class TradeParser:
    def __init__(self, log_file_path, path_to_portfolio_csv, model_name):
        self.log_file_path = log_file_path
        self.model_name = model_name

        #   Store the parsed data here:
        results = self.ParseLogFile(log_file_path)

        # self.GraphNetProfit(results)

        self.GraphPortfolioVStime(path_to_portfolio_csv)

        self.GraphNetProfit(results=results)


        self.trades = self.parse_trades()


    def parse_trades(self) -> List[Tuple[str, str, float, float]]:
        """
        Parses buy and sell trades from the log file.

        Returns:
            List of tuples containing (action, timestamp, shares, price).
        """
        trades = []
        # Regular expression to capture Buy/Sell actions with details
        trade_regex = r"INFO:env\.LargeAction_market_enviroment:(Bought|Sold) (\d+\.?\d*) shares at price (\d+\.\d+)"
        timestamp_regex = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})"

        with open(self.log_file_path, "r") as file:
            for line in file:
                # Extract timestamp
                timestamp_match = re.search(timestamp_regex, line)
                timestamp = timestamp_match.group(1) if timestamp_match else None

                # Extract trade details
                trade_match = re.search(trade_regex, line)
                if trade_match:
                    action, shares, price = trade_match.groups()
                    trades.append((action, timestamp, float(shares), float(price)))

        return trades

    def plot_trade_timeline(self):
        """
        Plots a timeline of trades (buys and sells) over time.
        """
        # Extract data
        actions = [trade[0] for trade in self.trades]
        timestamps = [trade[1] for trade in self.trades]
        shares = [trade[2] for trade in self.trades]

        # Assign y-values based on action
        y_values = [1 if action == "Bought" else -1 for action in actions]

        # Plot the timeline
        plt.figure(figsize=(12, 6))
        plt.scatter(timestamps, y_values, c=shares, cmap='coolwarm', s=shares, alpha=0.7, edgecolors="k")
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.title(f"Trade Timeline (Buys and Sells) for model {self.model_name}")
        plt.xlabel("Timestamp")
        plt.ylabel("Action (1=Buy, -1=Sell)")
        plt.xticks(rotation=45)
        plt.colorbar(label="Number of Shares")
        plt.tight_layout()
        plt.savefig("Trade_Timeline.png")
        plt.show()

    def plot_action_histogram(self):
        """
        Plots a histogram showing the distribution of actions.
        """
        
        actions = [trade[0] for trade in self.trades]

        # Count actions
        buy_count = actions.count("Bought")
        # for i in range(int(0.25*buy_count)):
        #     actions.append("Sold")

        sell_count = actions.count("Sold")

        # Create histogram
        plt.figure(figsize=(8, 6))
        plt.bar(["Bought", "Sold"], [buy_count, sell_count], color=["blue", "red"], alpha=0.7)
        plt.title(f"Action Distribution (Buy vs. Sell) for model {self.model_name}")
        plt.ylabel("Count")
        plt.xlabel("Action")
        plt.tight_layout()
        plt.savefig("Action_Histogram.png")
        plt.show()

    def print_trades(self):
        """Prints the parsed trades in a readable format."""
        for trade in self.trades:
            action, timestamp, shares, price = trade
            print(f"{timestamp}: {action} {shares} shares at ${price:.2f}")

    def GraphPortfolioVStime(self, path_to_portfolio_csv, downsample_factor=10, smoothing_window=5):
        data = pd.read_csv(f"{path_to_portfolio_csv}")

        # Downsample the data
        data = data.iloc[::downsample_factor, :].reset_index(drop=True)

        # Apply moving average for smoothing
        data["Smoothed Portfolio Value"] = data["Portfolio Value"].rolling(window=smoothing_window).mean()

        # Plot the graph
        plt.figure(figsize=(14, 7))
        # plt.plot(data["Timestep"], data["Portfolio Value"], marker='o', linestyle='-', alpha=0.5, label="Raw Portfolio Value")
        plt.plot(data["Timestep"], data["Smoothed Portfolio Value"], color='red', linestyle='-', linewidth=2, label="Smoothed Portfolio Value")
        plt.title(f"Final Portfolio Value Per Episode (N) for model {self.model_name}", fontsize=16)
        plt.xlabel("Episode (N)", fontsize=14)
        plt.ylabel("Portfolio Value", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        # Plot the graph
        plt.figure(figsize=(14, 7))
        plt.plot(data["Timestep"], data["Portfolio Value"], marker='o', linestyle='-', alpha=0.5, label="Raw Portfolio Value")
        # plt.plot(data["Timestep"], data["Smoothed Portfolio Value"], color='red', linestyle='-', linewidth=2, label="Smoothed Portfolio Value")
        plt.title(f"Final Portfolio Value Per Episode (N) for model {self.model_name}", fontsize=16)
        plt.xlabel("Episode (N)", fontsize=14)
        plt.ylabel("Portfolio Value", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
    
    

    def GraphNetProfitWithNoise(self, results: List[List[float]], noise_mean: float = 0, noise_std: float = 10):
        # Extract net profits and episode numbers
        net_profits = [float(entry[-1]) for entry in results]  # Convert to float
        episodes = [i + 1 for i in range(len(results))]  # 1-based index for episodes

        # Add Gaussian noise to net profits with increased randomness
        noise_mean = 0  # Adjust the mean for a baseline shift if needed
        noise_std = 100  # Increase the standard deviation for more variability
        additional_random_noise = np.random.uniform(-20, 20, size=len(net_profits))  # Uniform random noise

        # Combine Gaussian and uniform noise for higher randomness
        noise = np.random.normal(loc=noise_mean, scale=noise_std, size=len(net_profits)) + additional_random_noise
        noisy_net_profits = [profit + n for profit, n in zip(net_profits, noise)]  # Addition works with floats

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.title(f"Net Profit vs Episode (With Noise) for model {self.model_name}")
        plt.xlabel("Episode Number (N)")
        plt.ylabel("Net Profit")
        # plt.plot(episodes, net_profits, marker='o', linestyle='-', label="Original Net Profit")
        plt.plot(episodes, noisy_net_profits, marker='o', linestyle='-', label="Noisy Net Profit", alpha=1)

        # Set proper y-axis limits and tick intervals
        min_profit = min(noisy_net_profits)
        max_profit = max(noisy_net_profits)
        y_tick_interval = (max_profit - min_profit) / 10  # 10 intervals
        plt.ylim(min_profit - y_tick_interval, max_profit + y_tick_interval)  # Add padding
        plt.yticks(ticks=[min_profit + i * y_tick_interval for i in range(12)])

        # Add grid and legend
        plt.grid(True)
        plt.legend()

        # Save and display the plot
        plt.savefig("TEST_FIGURE_NOISE.png")
        plt.show()

    def GraphNetProfit(self, results: List[List[float]]):
    
        # Extract net profits and episode numbers
        net_profits = [float(entry[-1]) for entry in results]
        episodes = [i + 1 for i in range(len(results))]  # 1-based index for episodes

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.title(f"Net Profit vs Episode for model {self.model_name}")
        plt.xlabel("Episode Number (N)")
        plt.ylabel("Net Profit")
        plt.plot(episodes, net_profits, marker='o', linestyle='-', label="Net Profit")

        # Set proper y-axis limits and tick intervals
        min_profit = min(net_profits)
        max_profit = max(net_profits)
        y_tick_interval = (max_profit - min_profit) / 10  # 10 intervals
        plt.ylim(min_profit - y_tick_interval, max_profit + y_tick_interval)  # Add padding
        plt.yticks(ticks=[min_profit + i * y_tick_interval for i in range(12)])

        # Add grid and legend
        plt.grid(True)
        plt.legend()

        # Save and display the plot
        plt.savefig("TEST_FIGURE_FIXED.png")
        plt.show()
    

    # Parses each individaul log file and creates their respective data frames
    def ParseLogFile(self, log_file_path: str = None):
        assert log_file_path != None , "No log file was provided!!!"
        print(log_file_path)
        count_num_befores = list()
        
        # Regular expression to match "BEFORE RESET" and the desired information
        # regex = r"INFO:env\.[^:]+:Environment BEFORE RESET: Balance: ([\d\.]+), Holdings: (\d+), Net Profit: ([\d\.]+)"
                
                
        # Regular expression to match "BEFORE RESET" and extract Balance, Holdings, Net Profit
        regex = r"Environment BEFORE RESET: Balance: ([\d\.]+), Holdings: ([\d\.]+), Net Profit: ([\d\.]+)"

        results = list()
        count = 0
        # Open the log file and process each line
        with open(log_file_path, "r") as log_file:
            for i, line in tqdm(enumerate(log_file.readlines()), desc="Processing log lines"):
                # Check if the line contains "BEFORE RESET" and matches the regex
                match = re.search(regex, line)
                if match:
                    # Print the matched line and extracted values
                    print(f"Line {i}: {line.strip()}")
                    balance, holdings, net_profit = match.groups()
                    print(f"Extracted - Balance: {balance}, Holdings: {holdings}, Net Profit: {net_profit}")
                    count += 1
                    results.append([balance, holdings, net_profit])


        print(f"\n\nNumber of Extracted BEFORE RESETS: {count}")
        return results
    


    def AllPortfolioGraph(self, ppo_large_data, ppo_small_data, a2c_data, drqn_data):
        """
        Plots portfolio values for multiple models on the same plot with different colors and labels.

        Args:
            ppo_large_data (str): Path to CSV file for PPO (Large Action Space) data.
            ppo_small_data (str): Path to CSV file for PPO (Small Action Space) data.
            a2c_data (str): Path to CSV file for A2C data.
            drqn_data (str): Path to CSV file for DRQN data.
        """
        # Load data
        ppo_large = pd.read_csv(ppo_large_data)
        ppo_small = pd.read_csv(ppo_small_data)
        a2c = pd.read_csv(a2c_data)
        drqn = pd.read_csv(drqn_data)

        # Plot the data
        plt.figure(figsize=(14, 8))
        plt.plot(ppo_large["Timestep"], ppo_large["Portfolio Value"], label="PPO Large", color="blue", linestyle="-", alpha=0.8)
        plt.plot(ppo_small["Timestep"], ppo_small["Portfolio Value"], label="PPO Small", color="green", linestyle="--", alpha=0.8)
        plt.plot(a2c["Timestep"], a2c["Portfolio Value"], label="A2C", color="orange", linestyle="-.", alpha=0.8)
        plt.plot(drqn["Timestep"], drqn["Portfolio Value"], label="DRQN", color="red", linestyle=":", alpha=0.8)
        # plt.xlim(0,1000)
        # Add title and labels
        plt.title("Final Portfolio Values Per Episode for Different Models", fontsize=16)
        plt.xlabel("Episode (N)", fontsize=14)
        plt.ylabel("Portfolio Value", fontsize=14)

        # Add grid, legend, and layout adjustments
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Show the plot
        plt.show()
    

if __name__ == "__main__":
    """LARGE PPO MODEL:"""
    print("Results for Large Action PPO:")
    parser_ppo_large = TradeParser(
        log_file_path=os.path.join(ROOT, "ppo_large_action_evaluation.log"),
        path_to_portfolio_csv=os.path.join(ROOT, "Benchmarking/logs/PPO_Large/portfolio_values.csv"),
        model_name="PPO Large Action"
    )
    parser_ppo_large.print_trades()
    parser_ppo_large.plot_trade_timeline()
    parser_ppo_large.plot_action_histogram()

    """SMALL PPO MODEL:"""
    print("Results for Small Action PPO:")
    parser_ppo_small = TradeParser(
        log_file_path=os.path.join(ROOT, "ppo_small_action_evaluation.log"),
        path_to_portfolio_csv=os.path.join(ROOT, "Benchmarking/logs/PPO_Small/portfolio_values.csv"),
        model_name="PPO Small Action"
    )
    parser_ppo_small.print_trades()
    parser_ppo_small.plot_trade_timeline()
    parser_ppo_small.plot_action_histogram()

    """LARGE A2C MODEL:"""
    print("Results for Large Action A2C:")
    parser_a2c = TradeParser(
        log_file_path=os.path.join(ROOT, "a2c_evaluation.log"),
        path_to_portfolio_csv=os.path.join(ROOT, "Benchmarking/logs/A2C/portfolio_values.csv"),
        model_name="A2C"
    )
    parser_a2c.print_trades()
    parser_a2c.plot_trade_timeline()
    parser_a2c.plot_action_histogram()

    """LARGE DRQN MODEL:"""
    print("Results for DRQN Model:")
    parser_drqn = TradeParser(
        log_file_path=os.path.join(ROOT, "drqn_evaluation.log"),
        path_to_portfolio_csv=os.path.join(ROOT, "Benchmarking/logs/DRQN/portfolio_values.csv"),
        model_name="DRQN"
    )
    parser_drqn.print_trades()
    parser_drqn.plot_trade_timeline()
    parser_drqn.plot_action_histogram()

    # Combined portfolio values plot
    parser_ppo_large.AllPortfolioGraph(
        ppo_large_data=os.path.join(ROOT, "Benchmarking/logs/PPO_Large/portfolio_values.csv"),
        ppo_small_data=os.path.join(ROOT, "Benchmarking/logs/PPO_Small/portfolio_values.csv"),
        a2c_data=os.path.join(ROOT, "Benchmarking/logs/A2C/portfolio_values.csv"),
        drqn_data=os.path.join(ROOT, "Benchmarking/logs/DRQN/portfolio_values.csv")
    )

    pass
