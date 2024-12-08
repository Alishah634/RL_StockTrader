import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym


import sys
sys.path.append("../")
from data.data_loader import DataPreprocessor
from env.portfolio_class import Portfolio
from env.slow_market_enviroment_drqn import MarketEnvironment

# Define Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            try:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            except ValueError as e:
                print("THE FUCKING MINIBATCH:")
                # print(state)
                print(minibatch)
                print("The FUCKING STATE:\n")
                print(state)
                print("The FUCKING ACTION:\n")
                print(action)
                print("The FUCKING reward:\n")
                print(reward)
                print("The FUCKING next_state:\n")
                print(next_state)
                print("The FUCKING done:\n")
                print(done)
                print("The EXCEPTION:\n")
                print(e)
                exit(0)                

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(self.device)

            target = reward_tensor
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()
            target_f = self.model(state_tensor).detach().cpu().numpy()
            target_f[action] = target.item()
            
            target_tensor = torch.tensor(target_f, dtype=torch.float32).to(self.device)
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target_tensor)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    from config.logging_config import log_decorator
    @log_decorator(enabled=False)
    def testing_log(a = 1):
        a += 1
        print(a)
    testing_log()
    # Data Preprocessing
    csv_path = "../data/raw/sp500/DLTR.csv"
    preprocessor = DataPreprocessor()
    try:
        processed_data = preprocessor.load_csv(csv_path)
        preprocessor.log_csv_head()
        preprocessor.log_dataset_metrics()
    except Exception as e:
        print(f"Failed to load or preprocess CSV data: {e}")

    portfolio = Portfolio("John", 1000)
    env = MarketEnvironment(data=processed_data, portfolio=portfolio, initial_balance=1000)


    # Create environment and agent
    # env = gym.make('CartPole-v1')
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n

    state_dim = 6
    action_dim = 2

    agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)

    # Train the DQN agent with Experience Replay Buffer
    batch_size = 32
    num_episodes = 1000
    from tqdm import tqdm
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
