import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append("../")
from data.data_loader import DataPreprocessor  # Assuming DataPreprocessor is in data/data_loader.py

class LSTMPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        LSTM model for predicting next `Open` and `Close` prices.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of units in the LSTM hidden layer.
            output_dim (int): Number of outputs to predict (e.g., `Open` and `Close`).
            num_layers (int): Number of LSTM layers.
        """
        super(LSTMPricePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers to capture time dependencies
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully connected layer to map hidden state to output (Open and Close)
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


def test_lstm(csv_paths, lstm_model, criterion, sequence_length=10):
    lstm_model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for csv_path in csv_paths:
            # Load data using DataPreprocessor
            preprocessor = DataPreprocessor()
            try:
                data = preprocessor.load_csv(csv_path)
                preprocessor.log_csv_head()
                preprocessor.log_dataset_metrics()
            except Exception as e:
                print(f"Failed to load data from {csv_path}: {e}")
                continue

            print(f"Testing on data from: {csv_path}")

            # Ensure only the desired columns are used
            data = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Open']]

            # Prepare test data
            features = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume']].values
            targets = data[['Open', 'Close']].values

            # Normalize features and targets
            feature_max, feature_min = features.max(axis=0), features.min(axis=0)
            target_max, target_min = targets.max(axis=0), targets.min(axis=0)
            features = (features - feature_min) / (feature_max - feature_min)
            targets = (targets - target_min) / (target_max - target_min)

            # Create sequences for testing
            X, Y = [], []
            for i in range(len(features) - sequence_length):
                X.append(features[i:i + sequence_length])
                Y.append(targets[i + sequence_length])
            
            X = np.array(X)
            Y = np.array(Y)

            # Convert to PyTorch tensors
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)

            # Run inference
            outputs = lstm_model(X)
            losses = criterion(outputs, Y)
            total_loss += losses.item() * len(X)
            n_samples += len(X)

            # Reverse normalization for output and targets
            predictions = outputs.numpy() * (target_max - target_min) + target_min
            actual_values = Y.numpy() * (target_max - target_min) + target_min

            for i in range(len(predictions)):
                print(f"Predicted value: {predictions[i]} <------> Actual value: {actual_values[i]}")

    # Calculate average loss
    avg_loss = total_loss / n_samples if n_samples != 0 else float('inf')
    print(f"Average Test Loss (MSE): {avg_loss:.4f}")


if __name__ == '__main__':
    # Path to the CSV files containing stock data for testing
    csv_folder = "../data/raw/sp500/"
    csv_paths = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]
    csv_paths = csv_paths[5:10]  # Use different CSVs from those used for training
    
    # Initialize model
    input_dim = 5  # Using 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume' (excluding 'Date')
    hidden_dim = 128
    output_dim = 2  # Predict 'Open' and 'Close'
    lstm_model = LSTMPricePredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Load the trained model    
    model_path = '../models/saved_models/lstm_model.pth'
    try:
        lstm_model.load_state_dict(torch.load(model_path))
        print("LSTM model loaded successfully for testing.")
    except FileNotFoundError:
        print(f"Trained model file not found at {model_path}. Please train the model first.")
        exit()

    # Define the criterion for evaluating the model
    criterion = nn.MSELoss()

    # Test the LSTM model
    test_lstm(csv_paths, lstm_model, criterion, sequence_length=10)
