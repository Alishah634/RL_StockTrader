import os
import sys
import csv
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import tqdm

ROOT = os.getenv('PROJECT_ROOT', "/home/shantanu/RL_Proj/RL_StockTrader")
sys.path.append(ROOT)

from config.logging_config import ensure_directory
from data.data_loader import DataPreprocessor
from env.portfolio_class import Portfolio
from env.fast_drqn_market_enviroment import MarketEnvironment


import pickle

ST8 = "/home/shantanu/RL_Proj/RL_StockTrader/drqn_god_tier.pkl"

def save_program_state(
    agent,
    env,
    episode,
    training_rewards_per_episode,
    training_portfolio_values,
    training_final_portfolio_values,
    training_profits_losses,
    evaluation_rewards_per_episode,
    evaluation_portfolio_values,
    evaluation_final_portfolio_values,
    evaluation_profits_losses,
):
    # Create a dictionary containing all relevant program states
    program_state = {
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "env_state": env,
        "episode": episode,
        "metrics": {
            "training_rewards_per_episode": training_rewards_per_episode,
            "training_portfolio_values": training_portfolio_values,
            "training_final_portfolio_values": training_final_portfolio_values,
            "training_profits_losses": training_profits_losses,
            "evaluation_rewards_per_episode": evaluation_rewards_per_episode,
            "evaluation_portfolio_values": evaluation_portfolio_values,
            "evaluation_final_portfolio_values": evaluation_final_portfolio_values,
            "evaluation_profits_losses": evaluation_profits_losses,
        },
    }

    # Save to pickle file
    with open(ST8, "wb") as f:
        pickle.dump(program_state, f)
    print(f"Program state saved to {ST8}")


def load_program_state(agent, state_path=ST8):
    if os.path.exists(ST8):
        print(f"Loading program state from {state_path}")
        with open(state_path, "rb") as f:
            program_state = pickle.load(f)
        
        # Restore model and optimizer state
        agent.model.load_state_dict(program_state["model_state_dict"])
        agent.optimizer.load_state_dict(program_state["optimizer_state_dict"])
        agent.epsilon = program_state["epsilon"]

        # Restore environment state if applicable
        env = program_state["env_state"]

        # Restore metrics
        metrics = program_state["metrics"]
        training_rewards_per_episode = metrics["training_rewards_per_episode"]
        training_portfolio_values = metrics["training_portfolio_values"]
        training_final_portfolio_values = metrics["training_final_portfolio_values"]
        training_profits_losses = metrics["training_profits_losses"]
        evaluation_rewards_per_episode = metrics["evaluation_rewards_per_episode"]
        evaluation_portfolio_values = metrics["evaluation_portfolio_values"]
        evaluation_final_portfolio_values = metrics["evaluation_final_portfolio_values"]
        evaluation_profits_losses = metrics["evaluation_profits_losses"]

        episode_start = program_state["episode"] + 1

        return (
            agent,
            env,
            episode_start,
            training_rewards_per_episode,
            training_portfolio_values,
            training_final_portfolio_values,
            training_profits_losses,
            evaluation_rewards_per_episode,
            evaluation_portfolio_values,
            evaluation_final_portfolio_values,
            evaluation_profits_losses,
        )
    else:
        print(f"No program state found at {state_path}")
        return None



class DRQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128,  lstm_layers=1):
        super(DRQN, self).__init__()
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



class DRQNAgent:
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
        self.model = DRQN(state_dim, action_dim).to(self.device)
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

def DRQN_main(data_path, model_path: str = None, is_train_mode: bool = True ):
    data_path = os.path.join(ROOT, data_path)
    save_path = os.path.join(ROOT, "Benchmarking/trained_models")

    #  INITITIALIZATION STEPS for Training:
    ######################################################################################################
    preprocessor = DataPreprocessor()
    try:
        processed_data = preprocessor.load_csv(data_path)
        preprocessor.log_csv_head()
        preprocessor.log_dataset_metrics()
    except Exception as e:
        print(f"Failed to load or preprocess CSV data: {e}")

    portfolio = Portfolio("John", 1000)
    env = MarketEnvironment(data=processed_data, portfolio=portfolio, initial_balance=1000)

    state_dim = 6
    action_dim = 2

    agent = DRQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)

    # SETTINGS FOR TRAINING LOOP:
    # Train the DQN agent with Experience Replay Buffer
    batch_size = 32
    num_episodes = 100
    hidden_dim = 128
    linear_dim = 64

    # Check if the model path exists:
    model_id =  f"H{hidden_dim}_L{linear_dim}_E{num_episodes}"
    ensure_directory(save_path)
    model_path = model_path or os.path.join(save_path, f"drqn_model_{model_id}.pth")
    model_path = os.path.abspath(model_path)


    training_rewards_per_episode = []  # Tracks total rewards earned by the agent in each training episode.
    training_portfolio_values = []    # Tracks portfolio value at each time step during training.
    training_final_portfolio_values = []  # Tracks the final portfolio value at the end of each training episode.
    training_profits_losses = []      # Tracks profit or loss (final value - initial balance) for each training episode.

    # Tracking metrics for evaluation
    evaluation_rewards_per_episode = []  # Tracks total rewards earned by the agent in each evaluation episode.
    evaluation_portfolio_values = []     # Tracks portfolio value at each time step during evaluation.
    evaluation_final_portfolio_values = []  # Tracks the final portfolio value at the end of each evaluation episode.
    evaluation_profits_losses = []       # Tracks profit or loss (final value - initial balance) for each evaluation episode.


    

    if os.path.exists(model_path) or is_train_mode:
        print(f"\n\nLoading existing model from {model_path}\n\n")
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()
        
    else:
        # TRAINING lOOP:
        print(f"\n\nSTARTING TRAINING!!!\n\n")
        exit(0)
        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            total_reward = 0
            episode_portfolio_values = []  # Tracks portfolio value within an episode
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)

                if truncated or done:
                    break

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                agent.replay(batch_size)

            training_rewards_per_episode.append(total_reward)  # Total reward for this training episode
            training_final_portfolio_values.append(env.portfolio.portfolio_value)  # Final portfolio value for this episode
            profit_loss = env.portfolio.portfolio_value - env.initial_balance
            training_profits_losses.append(profit_loss)  # Profit or loss for this episode
            training_portfolio_values.append(episode_portfolio_values)  # Append step-by-step portfolio values

    
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

        # Save the trained model
        torch.save(agent.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        save_program_state(
            agent,
            env,
            episode,
            training_rewards_per_episode,
            training_portfolio_values,
            training_final_portfolio_values,
            training_profits_losses,
            evaluation_rewards_per_episode,
            evaluation_portfolio_values,
            evaluation_final_portfolio_values,
            evaluation_profits_losses,
        )



    #  INITITIALIZATION STEPS for Evaluation:
    ######################################################################################################
    # Evaluation loop for this CSV
    print(f"\n\nSTARTING EVALUATION!!!\n\n")

    num_eval_episodes = 100
    # Store timestep vs portfolio value
    timestep = 0
    portfolio_values = list()

    for episode in tqdm(range(num_eval_episodes), desc="DRQN Evaluation..."):
        state = env.reset()
        total_reward = 0
        episode_portfolio_values = []  # Tracks portfolio value for this episode
        done = False
        while not done:
            action = agent.act(state)  # Use the trained model to act
            next_state, reward, done, truncated, _ = env.step(action)

             # Track portfolio value at each time step (used for "Portfolio Value Over Time")
            episode_portfolio_values.append(env.portfolio.portfolio_value)

            if truncated:
                break
            state = next_state
            total_reward += reward
        

        evaluation_rewards_per_episode.append(total_reward)  # Total reward for this evaluation episode
        evaluation_final_portfolio_values.append(env.portfolio.portfolio_value)  # Final portfolio value
        profit_loss = env.portfolio.portfolio_value - env.initial_balance
        evaluation_profits_losses.append(profit_loss)  # Profit or loss for this episode
        evaluation_portfolio_values.append(episode_portfolio_values)  # Append step-by-step portfolio values


        print(f"Evaluation Episode: {episode + 1}, Total Reward: {portfolio.net_profit}")
        # Append the portfolio value and timestep to the list
        portfolio_value = portfolio.portfolio_value  # Access the portfolio value from the environment
        portfolio_values.append((timestep, portfolio_value))
        timestep += 1

    print("Evaluation completed.")

    # Save portfolio values to CSV
    csv_path = os.path.join(ROOT, "Benchmarking/logs/DRQN/portfolio_values.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestep", "Portfolio Value"])  # Write the header
        writer.writerows(portfolio_values)  # Write the data

    print(f"Portfolio values saved to {csv_path}")

    print("Evaluation completed.")

if __name__ == "__main__":
    # DRQN_main(f"{ROOT}/data/raw/sp500/DLTR.csv", is_train_mode=False)
    DRQN_main(f"{ROOT}/data/raw/sp500/JPM.csv", is_train_mode=False)

    # "data/raw/sp500/AAPL.csv"  
    # "data/raw/sp500/JPM.csv"  
    # "data/raw/sp500/AAL.csv"  
    # "data/raw/sp500/MSFT.csv"  
    # "data/raw/sp500/DLTR.csv"  