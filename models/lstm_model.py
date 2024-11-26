import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm
from collections import deque
<<<<<<< HEAD
import time

sys.path.append('../')
from data.data_loader import DataPreprocessor  # Assuming DataPreprocessor is in data/data_loader.py
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
=======

sys.path.append('../')
from data.data_loader import DataPreprocessor  # Assuming DataPreprocessor is in data/data_loader.py
>>>>>>> d3da7a3 (Tried many models, lstm model works best so far, not perfect but close)

# Define the LSTM model
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
<<<<<<< HEAD
=======
        """
        LSTM model for predicting next `Open` and `Close` prices.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of units in the LSTM hidden layer.
            output_dim (int): Number of outputs to predict (e.g., `Open` and `Close`).
            num_layers (int): Number of LSTM layers.
        """
>>>>>>> d3da7a3 (Tried many models, lstm model works best so far, not perfect but close)
        super(LSTMPricePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers to capture time dependencies
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

<<<<<<< HEAD
        # Attention layer
        self.attention = nn.Linear(hidden_dim, 1)

        # Fully connected layer to map from hidden state to output (Open and Close)
        self.fc = nn.Linear(hidden_dim, output_dim)

    # def forward(self, x):
    #     # Initialize LSTM hidden and cell states
    #     h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
    #     c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

    #     # LSTM forward pass
    #     out, _ = self.lstm(x, (h0, c0))

    #     # Fully connected layer to output predicted values
    #     out = self.fc(out[:, -1, :])  # Taking only the last time step output
    #     return out

    def forward(self, x):
=======
        # Fully connected layer to map from hidden state to output (Open and Close)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the LSTM.

        Args:
            x (Tensor): Input features, of shape (batch_size, seq_len, input_dim).

        Returns:
            output (Tensor): Predicted values for the next `Open` and `Close` prices.
        """
>>>>>>> d3da7a3 (Tried many models, lstm model works best so far, not perfect but close)
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM forward pass
<<<<<<< HEAD
        out, _ = self.lstm(x, (h0, c0))  # out: [batch_size, seq_length, hidden_dim]

        # Compute attention scores
        attention_scores = self.attention(out)  # [batch_size, seq_length, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_length, 1]

        # Compute context vector as weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * out, dim=1)  # [batch_size, hidden_dim]

        # Fully connected layer to output predicted values
        output = self.fc(context_vector)  # [batch_size, output_dim]
        return output


def train_lstm(csv_paths, lstm_model, optimizer, criterion, num_epochs=50, batch_size=32, sequence_length=10, model_path: str = None):
    # model_path = 'saved_models/lstm_model.pth'

    # Check if model has already been trained
    if os.path.exists(model_path):
        try:
            lstm_model.load_state_dict(torch.load(model_path))
            print("Model already trained. Loaded the trained model.")
            return
        except Exception as e:
            print(f"Error loading model. Proceeding with training: {e}")

=======
        out, _ = self.lstm(x, (h0, c0))

        # Fully connected layer to output predicted values
        out = self.fc(out[:, -1, :])  # Taking only the last time step output
        return out


def train_lstm(csv_paths, lstm_model, optimizer, criterion, num_epochs=50, batch_size=32, sequence_length=10):
>>>>>>> d3da7a3 (Tried many models, lstm model works best so far, not perfect but close)
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
<<<<<<< HEAD
            X = torch.tensor(X, dtype=torch.float32).to(device)
            Y = torch.tensor(Y, dtype=torch.float32).to(device)
=======
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)
>>>>>>> d3da7a3 (Tried many models, lstm model works best so far, not perfect but close)

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
<<<<<<< HEAD
    torch.save(lstm_model.state_dict(), model_path)


def evaluate_lstm(csv_paths, lstm_model, sequence_length=10, error_threshold=0.05):
    preprocessor = DataPreprocessor()

    result_file = open("results.txt", "w+") 
    def write_to_file_and_print(message):
        print(message)
        result_file.write(message + "\n")

    print(f"Evaluating stock at: {csv_paths}:")

    lstm_model.eval()
    with torch.no_grad():
        total_accuracy = 0
        total_samples = 0
        dataset_accuracies = {}
        dataset_accuracies_within_error = {}
        for csv_path in csv_paths:
            print(f"Loading data from {csv_path}...")
            write_to_file_and_print(f"Loading data from {csv_path}...")
            # time.sleep(1)
=======
    torch.save(lstm_model.state_dict(), 'saved_models/lstm_model.pth')


def evaluate_lstm(csv_paths, lstm_model, sequence_length=10):
    preprocessor = DataPreprocessor()
    lstm_model.eval()
    with torch.no_grad():
        for csv_path in csv_paths:
>>>>>>> d3da7a3 (Tried many models, lstm model works best so far, not perfect but close)
            try:
                # Load data using DataPreprocessor
                data = preprocessor.load_csv(csv_path)
                data = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Open']]  # Select the columns needed
<<<<<<< HEAD
                print(f"Stock name: {preprocessor.stock_name}\n")
                # Prepare evaluation data
                features = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume']].values
                targets = data[['Open', 'Close']].values
                actual_values = data[['Open', 'Close']].values
                
=======

                # Prepare evaluation data
                features = data[['High', 'Low', 'Close', 'Adjusted_Close', 'Volume']].values
                targets = data[['Open', 'Close']].values

>>>>>>> d3da7a3 (Tried many models, lstm model works best so far, not perfect but close)
                # Normalize features and targets (min-max scaling)
                feature_max, feature_min = features.max(axis=0), features.min(axis=0)
                target_max, target_min = targets.max(axis=0), targets.min(axis=0)
                features = (features - feature_min) / (feature_max - feature_min)

                # Create sequences
<<<<<<< HEAD
                X, Y = [], []
                for i in range(len(features) - sequence_length):
                    X.append(features[i:i + sequence_length])
                    # Y.append(targets[i + sequence_length])
                    Y.append(actual_values[i + sequence_length])
                
                X = np.array(X)
                X = torch.tensor(X, dtype=torch.float32).to(device)
                Y = np.array(Y)

                # Predict
                predictions = lstm_model(X).cpu().detach().numpy()

                # Reverse normalization for evaluation
                predictions = predictions * (target_max - target_min) + target_min
                # actual_values = Y * (target_max - target_min) + target_min
                # Print predicted vs actual values
                accurate_count = 0
                within_threshold = []
                for i in range(len(predictions)):
                    error = np.abs(predictions[i] - Y[i])
                    relative_error = error / (np.abs(Y[i]) + 1e-8)  # Avoid division by zero
                    is_accurate = (relative_error <= error_threshold).all()  # Check if all outputs meet the threshold
                    within_threshold.append(is_accurate)
                    accurate_count += 1 if is_accurate else 0
                    print(f"Predicted: {predictions[i]} | Actual: {Y[i]} | Accurate: {is_accurate}")

                
                # Calculate accuracy
                errors = np.abs(predictions - Y)
                relative_errors = errors / (np.abs(Y) + 1e-8)  # Add epsilon to avoid division by zero
                accuracies = 1 - relative_errors.mean(axis=1)  # Accuracy per sample
                mean_accuracy = accuracies.mean() * 100  # Convert to percentage

                # Store accuracy for the dataset
                dataset_accuracies[csv_path] = mean_accuracy
                dataset_accuracies_within_error[csv_path] = accurate_count / len(predictions) * 100
                
        #         # Print all dataset accuracies
        #         print(f"Dataset {csv_path} Accuracy: {mean_accuracy:.2f}%")
                
        #     except Exception as e:
        #         print(f"Failed to load or process data from {csv_path}: {e}")
        #         continue
            
        # # Print all dataset accuracies
        # print("Per-Dataset Accuracies:")
        # for path, accuracy in dataset_accuracies.items():
        #     print(f"{path}: Accuracy: {accuracy:.2f}%, and Accuracy {dataset_accuracies_within_error[path]:.2f}% within error threshold")
            

        # # Compute overall accuracy across all datasets
        # overall_accuracy = np.mean(list(dataset_accuracies.values())) if dataset_accuracies else 0
        # print(f"\nOverall accuracy across all datasets: {overall_accuracy:.2f}%")
        
        # overall_accuracy_within_error = np.mean(list(dataset_accuracies_within_error.values())) if dataset_accuracies_within_error else 0
        # print(f"\nOverall accuracy across all datasets within error threshold: {overall_accuracy_within_error:.2f}%")
                # Print dataset accuracy
                write_to_file_and_print(f"Dataset {csv_path} Accuracy: {mean_accuracy:.2f}%")
            except Exception as e:
                write_to_file_and_print(f"Failed to load or process data from {csv_path}: {e}")
                continue
            
        # Print all dataset accuracies
        write_to_file_and_print("Per-Dataset Accuracies:")
        for path, accuracy in dataset_accuracies.items():
            write_to_file_and_print(f"{path}: Accuracy: {accuracy:.2f}%, and Accuracy {dataset_accuracies_within_error[path]:.2f}% within error threshold")

        # Compute overall accuracy across all datasets
        overall_accuracy = np.mean(list(dataset_accuracies.values())) if dataset_accuracies else 0
        write_to_file_and_print(f"\nOverall accuracy across all datasets: {overall_accuracy:.2f}%")
        
        overall_accuracy_within_error = np.mean(list(dataset_accuracies_within_error.values())) if dataset_accuracies_within_error else 0
        write_to_file_and_print(f"\nOverall accuracy across all datasets within error threshold: {overall_accuracy_within_error:.2f}%")
        result_file.close()
=======
                X = []
                for i in range(len(features) - sequence_length):
                    X.append(features[i:i + sequence_length])
                
                X = np.array(X)
                X = torch.tensor(X, dtype=torch.float32)

                # Predict
                predictions = lstm_model(X).detach().numpy()

                # Reverse normalization
                predictions = predictions * (target_max - target_min) + target_min
                print(f"Predictions for {csv_path}:\n{predictions}")
                
            except Exception as e:
                print(f"Failed to load or process data from {csv_path}: {e}")
                continue

>>>>>>> d3da7a3 (Tried many models, lstm model works best so far, not perfect but close)

if __name__ == '__main__':
    # Paths to CSV files containing stock data
    csv_folder = "../data/raw/sp500/"
    csv_paths = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]
<<<<<<< HEAD

    # Random seed to select a subset of stocks:
    # Seed is for reproducibility:
    # random.seed(42) 
    # random.shuffle(csv_paths)
    # Trains on first 10 stocks:
    # training_paths = csv_paths[:10] 
    training_paths = csv_paths

    # Evaluates on next 20 stocks:
    # evaluation_paths = csv_paths
    evaluation_paths = ["../data/raw/sp500/DLTR.csv"]

    # Initialize model, optimizer, and loss function
    input_dim = 5  # Using 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume'
    output_dim = 2  # Predict 'Open' and 'Close'
    
    """
    -  Adjustable variables for differnet model results: 
        [hidden_dim, learning_rate, sequence_length, batch_size, num_epochs]
    
    -  Model paths are saved in order:
        f'saved_models/lstm_{hidden_dim}_{learning_rate}_{sequence_length}_{batch_size}_{num_epochs}_model.pth'
    """
    
    hidden_dim = 256
    learning_rate= 0.001
    sequence_length = 10
    batch_size = 32
    num_epochs = 50

    lstm_model = LSTMPricePredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    # Default learning rate lr= 0.001
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model_path = f'lstm_models/lstm_ALL_{hidden_dim}_{learning_rate}_{sequence_length}_{batch_size}_{num_epochs}model.pth'

    # Load the model if it exists
    if os.path.exists(model_path):
        lstm_model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded the pre-trained model.")
    else:
        print("No trained model found. Starting training.")
        train_lstm(training_paths, lstm_model, optimizer, criterion, num_epochs=num_epochs, batch_size=batch_size, sequence_length=sequence_length, model_path=model_path)

    # Evaluate the model
    evaluate_lstm(evaluation_paths, lstm_model, sequence_length=sequence_length, error_threshold=0.05)
=======
    csv_paths = csv_paths[:5]
    
    # Initialize model, optimizer, and loss function
    input_dim = 5  # Using 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume'
    hidden_dim = 128
    output_dim = 2  # Predict 'Open' and 'Close'
    lstm_model = LSTMPricePredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the LSTM model to predict the next Open and Close prices
    train_lstm(csv_paths, lstm_model, optimizer, criterion, num_epochs=50, batch_size=32, sequence_length=10)

    # Evaluate the trained model
    evaluate_lstm(csv_paths, lstm_model, sequence_length=10)
>>>>>>> d3da7a3 (Tried many models, lstm model works best so far, not perfect but close)
