import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym

from tqdm import tqdm
import re
import os
import sys
sys.path.append("../")
from data.data_loader import DataPreprocessor
from env.portfolio_class import Portfolio
from env.maybe_fast_market_enviroment_drqn import MarketEnvironment
from tqdm import tqdm

# Define Deep Q-Network (DQN) Model
# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128,  lstm_layers=1):
        super(DQN, self).__init__()
        in_features, out_features = 64, 64
        # LSTM Layer
        # self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x, hidden_state=None):
        # Ensure input x has the correct dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension if missing

        # Pass through LSTM
        if hidden_state is None:
            x, hidden_state = self.gru(x)  # LSTM expects 3D input
        else:
            x, hidden_state = self.gru(x, hidden_state)

        # Use the last output in the sequence
        x = x[:, -1, :]  # Select the last timestep (shape: batch_size, hidden_size)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x, hidden_state



class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss() 

    def act(self, state):
        """
        Select an action using epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Train the LSTM-based DQN using minibatches of sequences.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Convert minibatch to tensors efficiently
        sequences = torch.tensor(np.array([item[0] for item in minibatch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([item[1] for item in minibatch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([item[2] for item in minibatch], dtype=torch.float32).to(self.device)
        next_sequences = torch.tensor(np.array([item[3] for item in minibatch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor([item[4] for item in minibatch], dtype=torch.float32).to(self.device)

        # Get current Q-values and target Q-values
        q_values, _ = self.model(sequences)  # Shape: (batch_size, action_dim)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values, _ = self.model(next_sequences)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss and optimize
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_epsilon(self):
        """
        Update epsilon for the epsilon-greedy policy.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":

    #  INITITIALIZATION STEPS for Training:
    ######################################################################################################
    csv_path = "../data/raw/sp500/DLTR.csv"
    preprocessor = DataPreprocessor()
    try:
        processed_data = preprocessor.load_csv(csv_path)
        preprocessor.log_csv_head()
        preprocessor.log_dataset_metrics()
    except Exception as e:
        print(f"Failed to load or preprocess CSV data: {e}")

    processed_data = processed_data.drop(columns=['Date', 'Open', 'Close', 'High', 'Low'], axis=1)

    portfolio = Portfolio("John", 1000)
    env = MarketEnvironment(data=processed_data, portfolio=portfolio, initial_balance=1000)

    state_dim = 2
    action_dim = 2

    agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)


    # SETTINGS FOR TRAINING LOOP:
    # Train the DQN agent with Experience Replay Buffer
    batch_size = 32
    # num_episodes = 1000
    num_episodes = 100
    hidden_dim = 128
    linear_dim = 64
    # Model format {hidden_dim}_{liner_dim}_{num_episdodes} 
    model_id =  f"H{hidden_dim}_L{linear_dim}_E{num_episodes}"
    # model_id =  f"H{hidden_dim}_L{state_dim}_{action_dim}_E{num_episodes}"

    # model_file = "{model_id}_fast_dqn_model.pth"
    save_path = "saved_models"
    model_file = os.path.join(save_path, f"{model_id}_dqn_model.pth")
    ######################################################################################################
    if os.path.exists(model_file):
        print(f"Loading existing model from {model_file}")
        agent.model.load_state_dict(torch.load(model_file))
        agent.model.eval()
    else:
        # TRAINING lOOP:
        print(f"\n\nSTARTING TRAINING!!!\n\n")

        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)

                if truncated:
                    break

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                agent.replay(batch_size)
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

        # Save the trained model
        os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
        model_file = os.path.join(save_path, f"{model_id}_dqn_model.pth")
        torch.save(agent.model.state_dict(), model_file)
        print(f"Model saved to {model_file}")



    #  INITITIALIZATION STEPS for Evaluation:
    ######################################################################################################
    # Initialization of stock results
    stock_results = {}
    

    csv_paths = [
        "../data/raw/sp500/HPE.csv" 
        ,"../data/raw/sp500/PM.csv" 
        ,"../data/raw/sp500/PSX.csv" 
        ,"../data/raw/sp500/MDLZ.csv" 
        ,"../data/raw/sp500/WU.csv" 
        ,"../data/raw/sp500/MS-PF.csv" 
        ,"../data/raw/sp500/GS-PJ.csv" 
        ,"../data/raw/sp500/MET.csv"
    ]

    # Get list of CSV paths
    # csv_paths = [os.path.join("../data/raw/sp500/", path) for path in os.listdir("../data/raw/sp500/") if path.endswith('.csv')]
    # print(csv_paths)

    # Slice to limit the range of paths (if needed)
    N_paths = [0, 3]
    # csv_paths = csv_paths[N_paths[0]: N_paths[-1]]

    # Regular expression to extract CSV filename
    pattern = r"[^/]+\.csv$"
    # Evaluation loop for each CSV file
    for csv_path in tqdm(csv_paths, desc=f"Evaluation of Multiple Stocks\n"):
        print(f"EVALUATING stock {csv_path}")
        # Initialize a new results dictionary for this stock
        results = {}

        # Load and preprocess data
        preprocessor = DataPreprocessor()
        try:
            processed_data = preprocessor.load_csv(csv_path)
            preprocessor.log_csv_head()
            preprocessor.log_dataset_metrics()
        except Exception as e:
            print(f"FAILED TO LOAD OR PREPROCESS CSV DATA: {e}!!!")
            continue  # Skip this file if preprocessing fails

        # Prepare environment
        processed_data = processed_data.drop(columns=['Date', 'Open', 'Close', 'High', 'Low'], axis=1)
        portfolio = Portfolio("John", 1000)
        env = MarketEnvironment(data=processed_data, portfolio=portfolio, initial_balance=1000, enable_logger=False)
    ######################################################################################################

        # Evaluation loop for this CSV
        print(f"\n\nSTARTING EVALUATION!!!\n\n")
        num_eval_episodes = 1000
        total_rewards = []
        for episode in range(num_eval_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = agent.act(state)  # Use the trained model to act
                next_state, reward, done, truncated, _ = env.step(action)
                if truncated:
                    break
                state = next_state
                total_reward += reward
            total_rewards.append(total_reward)
            print(f"Evaluation Episode: {episode + 1}, Total Reward: {portfolio.net_profit}")

        # Collect results for this CSV
        avg_reward = np.mean(total_rewards)
        results["Model_Net_Profit"] = portfolio.net_profit
        results["Average_Reward"] = avg_reward
        results["Market_Return"] = env.market_return
        results["Expected_Evaluation"] = env.expected_evaluation

        # Extract the CSV filename
        match = re.search(pattern, csv_path)
        if match:
            csv_name = match.group()
            csv_name = os.path.splitext(csv_name)[0]  # Remove .csv extension
            print(f"CSV Filename: {csv_name}")
        else:
            print("No CSV filename found.!!!")

        # Store results in stock_results using the CSV name
        stock_results[csv_name] = results

    # Print final results for each stock
    for stock_name, res in stock_results.items():
        print(
            f"Results of STOCK {stock_name}: Model_Net_Profit: {res['Model_Net_Profit']} | "
            f"Average_Reward: {res['Average_Reward']} | Market_Return: {res['Market_Return']} | "
            f"Expected_Evaluation: {res['Expected_Evaluation']}"
        )
