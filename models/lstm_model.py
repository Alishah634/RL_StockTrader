import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm
from collections import deque
import time
sys.path.append('../')
from data.data_loader import DataPreprocessor  # Assuming DataPreprocessor is in data/data_loader.py

# Define the LSTM model
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers to capture time dependencies
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully connected layer to map from hidden state to output (Open and Close)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Fully connected layer to output predicted values
        out = self.fc(out[:, -1, :])  # Taking only the last time step output
        return out


def train_lstm(csv_paths, lstm_model, optimizer, criterion, num_epochs=50, batch_size=32, sequence_length=10):
    model_path = 'saved_models/lstm_model.pth'

    # Check if model has already been trained
    if os.path.exists(model_path):
        try:
            lstm_model.load_state_dict(torch.load(model_path))
            print("Model already trained. Loaded the trained model.")
            return
        except Exception as e:
            print(f"Error loading model. Proceeding with training: {e}")

    replay_buffer = deque(maxlen=10000)
    preprocessor = DataPreprocessor()

    for csv_path in tqdm(csv_paths):
        try:
            # Load data using DataPreprocessor
            data = preprocessor.load_csv(csv_path)
            data = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Open']]  # Keep only relevant columns

            # Prepare training data
            features = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume']].values
            targets = data[['Open', 'Close']].values

            # Normalize features and targets using min-max scaling
            feature_max, feature_min = features.max(axis=0), features.min(axis=0)
            target_max, target_min = targets.max(axis=0), targets.min(axis=0)
            features = (features - feature_min) / (feature_max - feature_min)
            targets = (targets - target_min) / (target_max - target_min)

            # Create sequences
            X, Y = [], []
            for i in range(len(features) - sequence_length):
                X.append(features[i:i + sequence_length])
                Y.append(targets[i + sequence_length])
            
            X = np.array(X)
            Y = np.array(Y)

            # Convert to PyTorch tensors
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)

            # Training loop
            lstm_model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i + batch_size]
                    batch_Y = Y[i:i + batch_size]

                    # Forward pass
                    outputs = lstm_model(batch_X)
                    loss = criterion(outputs, batch_Y)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(X):.4f}")

        except Exception as e:
            print(f"Failed to load or process data from {csv_path}: {e}")
            continue

    # Save the trained LSTM model
    torch.save(lstm_model.state_dict(), model_path)


def evaluate_lstm(csv_paths, lstm_model, sequence_length=10):
    preprocessor = DataPreprocessor()
    print(f"Evaluating stock at: {csv_paths}:")
    time.sleep(5)
    lstm_model.eval()
    with torch.no_grad():
        for csv_path in csv_paths:
            try:
                # Load data using DataPreprocessor
                data = preprocessor.load_csv(csv_path)
                data = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Open']]  # Select the columns needed
                print(f"Stock name: {preprocessor.stock_name}\n")
                time.sleep(2)
                # Prepare evaluation data
                features = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume']].values
                targets = data[['Open', 'Close']].values
                actual_values = data[['Open', 'Close']].values
                
                # Normalize features and targets (min-max scaling)
                feature_max, feature_min = features.max(axis=0), features.min(axis=0)
                target_max, target_min = targets.max(axis=0), targets.min(axis=0)
                features = (features - feature_min) / (feature_max - feature_min)

                # Create sequences
                X, Y = [], []
                for i in range(len(features) - sequence_length):
                    X.append(features[i:i + sequence_length])
                    # Y.append(targets[i + sequence_length])
                    Y.append(actual_values[i + sequence_length])
                
                X = np.array(X)
                X = torch.tensor(X, dtype=torch.float32)
                Y = np.array(Y)

                # Predict
                predictions = lstm_model(X).detach().numpy()

                # Reverse normalization for evaluation
                predictions = predictions * (target_max - target_min) + target_min
                # actual_values = Y * (target_max - target_min) + target_min

                # Print predicted vs actual values
                for i in range(len(predictions)):
                    print(f"Predicted value: {predictions[i]} <------> Actual value: {actual_values[i]}")

            except Exception as e:
                print(f"Failed to load or process data from {csv_path}: {e}")
                continue


if __name__ == '__main__':
    # Paths to CSV files containing stock data
    csv_folder = "../data/raw/sp500/"
    csv_paths = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]
    csv_paths = csv_paths[:1]
    
    # Initialize model, optimizer, and loss function
    input_dim = 5  # Using 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume'
    hidden_dim = 128
    output_dim = 2  # Predict 'Open' and 'Close'
    lstm_model = LSTMPricePredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the LSTM model to predict the next Open and Close prices if not already trained
    train_lstm(csv_paths, lstm_model, optimizer, criterion, num_epochs=50, batch_size=32, sequence_length=10)

    # Evaluate the trained model
    evaluate_lstm(csv_paths, lstm_model, sequence_length=10)
