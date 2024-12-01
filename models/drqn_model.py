import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
# from agents.drqn import DRQN

import sys
sys.path.append('../')
from data.data_loader import DataPreprocessor  
import os
from tqdm import tqdm

# class DRQN(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=64):
#         """
#         Deep Recurrent Q-Network to predict next `Open` and `Close` prices.

#         Args:
#             input_dim (int): Number of input features.
#             output_dim (int): Number of outputs to predict (e.g., `Open` and `Close`).
#             hidden_dim (int): Number of units in the LSTM hidden layer.
#         """
#         super(DRQN, self).__init__()
#         self.hidden_dim = hidden_dim

#         # LSTM layer to capture time dependencies
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

#         # Fully connected layer to map hidden state to the output (Open and Close)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, hidden_state):
#         """
#         Forward pass of the DRQN.

#         Args:
#             x (Tensor): Input features, of shape (batch_size, seq_len, input_dim).
#             hidden_state (Tuple[Tensor, Tensor]): LSTM hidden state and cell state.

#         Returns:
#             q_values (Tensor): Predicted values for the next `Open` and `Close` prices.
#             hidden_state (Tuple[Tensor, Tensor]): Updated hidden and cell states.
#         """
#         # LSTM to extract temporal patterns
#         out, hidden_state = self.lstm(x, hidden_state)

#         # Fully connected to map to output values (e.g., Open, Close)
#         q_values = self.fc(out[:, -1, :])  # Taking only the last time step output

#         return q_values, hidden_state

#     def init_hidden(self, batch_size):
#         """
#         Initialize hidden and cell state of the LSTM.

#         Args:
#             batch_size (int): Batch size for training.

#         Returns:
#             Tuple[Tensor, Tensor]: Initialized hidden and cell states.
#         """
#         return (torch.zeros(1, batch_size, self.hidden_dim),
#                 torch.zeros(1, batch_size, self.hidden_dim))

class DRQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2):
        """
        Deep Recurrent Q-Network with an LSTM and Attention Layer.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of outputs to predict (e.g., `Open` and `Close`).
            hidden_dim (int): Number of units in the LSTM hidden layer.
            num_layers (int): Number of LSTM layers.
        """
        super(DRQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers to capture time dependencies
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully connected layer to map hidden state to the output (Open and Close)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state):
        """
        Forward pass of the DRQN.

        Args:
            x (Tensor): Input features, of shape (batch_size, seq_len, input_dim).
            hidden_state (Tuple[Tensor, Tensor]): LSTM hidden state and cell state.

        Returns:
            q_values (Tensor): Predicted values for the next `Open` and `Close` prices.
            hidden_state (Tuple[Tensor, Tensor]): Updated hidden and cell states.
        """
        out, hidden_state = self.lstm(x, hidden_state)
        q_values = self.fc(out[:, -1, :])  # Taking only the last time step output

        return q_values, hidden_state

    def init_hidden(self, batch_size):
        """
        Initialize hidden and cell state of the LSTM.

        Args:
            batch_size (int): Batch size for training.

        Returns:
            Tuple[Tensor, Tensor]: Initialized hidden and cell states.
        """
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))



def train_drqn(csv_paths, drqn, optimizer, criterion, episodes=50, batch_size=32):
    replay_buffer = deque(maxlen=10000)
    gamma = 0.99

    for csv_path in tqdm(csv_paths):
        # Load data using DataPreprocessor
        preprocessor = DataPreprocessor()
        try:
            data = preprocessor.load_csv(csv_path)
            preprocessor.log_csv_head()
            preprocessor.log_dataset_metrics()
        except Exception as e:
            print(f"Failed to load data from {csv_path}: {e}")
            continue

        print(f"Training on data from: {csv_path}")

        # Ensure only the desired columns are used
        data = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Open']]  # Keep only the columns we need

        # Training loop for the DRQN model
        for episode in range(episodes):
            state = data.iloc[0][['High', 'Low', 'Close', 'Adjusted_Close', 'Volume']].values  # Initial state
            hidden_state = drqn.init_hidden(batch_size=1)
            total_loss = 0
            
            for t in range(1, len(data)):
                next_state = data.iloc[t][['High', 'Low', 'Close', 'Adjusted_Close', 'Volume']].values  # Next state
                actual_values = data.iloc[t][['Open', 'Close']].values  # The actual target values to predict (Open, Close)
                done = t == len(data) - 1

                # Store (state, next_state, actual_values) in replay buffer
                replay_buffer.append((state, next_state, actual_values, done))
                state = next_state

                # Start training only if we have enough samples in the replay buffer
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, next_states, actual_values_batch, dones = zip(*batch)

                    # Convert lists to Tensors
                    states = torch.tensor(np.array(states), dtype=torch.float32)
                    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                    actual_values_batch = torch.tensor(np.array(actual_values_batch), dtype=torch.float32)

                    hidden_state_batch = drqn.init_hidden(batch_size)
                    predictions, _ = drqn(states.unsqueeze(1), hidden_state_batch)  # Predict next Open and Close prices

                    # Calculate the loss between predicted and actual values
                    loss = criterion(predictions, actual_values_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            print(f"Episode {episode + 1}/{episodes}, Loss: {total_loss:.4f}")

    # Save the trained DRQN model
    torch.save(drqn.state_dict(), 'saved_models/drqn_model.pth')

# Example usage
if __name__ == '__main__':
    # Path to the CSV files containing stock data
    csv_folder = "../data/raw/sp500/"
    csv_paths = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]
    csv_paths = csv_paths[:5]
    
    # Initialize model, optimizer, and loss criterion
    input_dim = 5  # Using 'High', 'Low', 'Close', 'Adj_Close', 'Volume' (excluding 'Date')
    output_dim = 2  # Predict 'Open' and 'Close'
    drqn = DRQN(input_dim=input_dim, output_dim=output_dim)
    optimizer = optim.Adam(drqn.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the DRQN model to predict the next Open and Close prices
    train_drqn(csv_paths, drqn, optimizer, criterion, episodes=50, batch_size=32)
