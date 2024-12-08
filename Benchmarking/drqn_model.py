import os
import sys
import re
import gym
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import tqdm

sys.path.append("../")
from config.logging_config import ensure_directory
from data.data_loader import DataPreprocessor
from env.portfolio_class import Portfolio
from env.fast_drqn_market_enviroment import MarketEnvironment

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

    state_dim = 2
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
    save_path = "saved_drq_models"

    if model_path == None:
        # Use pre trained model path:
        ensure_directory("saved_drq_models")
        model_path = os.path.join(save_path, f"{model_id}_drqn_model.pth")
    else:
        model_path = os.path.join("saved_drq_models/", model_path)  

    print(model_path)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        agent.model.load_state_dict(torch.load(model_path))
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
        torch.save(agent.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    #  INITITIALIZATION STEPS for Evaluation:
    ######################################################################################################
    # Evaluation loop for this CSV
    print(f"\n\nSTARTING EVALUATION!!!\n\n")
    num_eval_episodes = 100
    total_rewards = []
    for episode in tqdm(range(num_eval_episodes), desc="DRQN Evaluation..."):
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

if __name__ == "__main__":
    DRQN_main("../data/raw/sp500/DLTR.csv", is_train_mode=False)

  