# Script for training the RL agent

from config.logging_config import logging, setup_logging, cprint, List, np
from data.data_loader import DataPreprocessor
from env.portfolio_class import Portfolio
from env.market_environment import MarketEnvironment
from config.rl_config import RL_SETTINGS
# from env.market_environment import MarketEnvironment
# from agents.base_agent import BaseAgent  # Assuming a base agent class exists for RL


# def train(episodes: int = RL_SETTINGS["episodes"], learning_rate: float = RL_SETTINGS["learning_rate"], csv_path: (None | str) = None, required_columns: (None | List[str]) = None) -> None:
#     logging.debug("Training started...")
    
#     # Compact assertions
#     assert isinstance(episodes, int) and episodes > 0, "Episodes must be an integer greater than zero."
#     assert isinstance(learning_rate, (float, int)) and 0 < learning_rate <= 1, "Learning rate must be a float or int between 0 and 1."
#     assert (csv_path is not None) or (isinstance(csv_path, str) and (csv_path.strip() != '')), f"CSV path was type {type(csv_path)} and must be a non-empty string or None."
#     assert required_columns is None or (isinstance(required_columns, list) and all(isinstance(col, str) for col in required_columns)), "Required columns must be a list of strings or None."
#     logging.debug(f"All input parameters are valid. Proceeding with training...")

#     # Placeholder for training code
#     # Initialize data preprocessor
#     preprocessor = DataPreprocessor()
#     try:
#         # Load and preprocess data from the specified CSV path
#         processed_data = preprocessor.load_csv(csv_path, required_columns)
#         preprocessor.log_csv_head()
#         preprocessor.log_dataset_metrics()
#         logging.debug("CSV data successfully loaded and preprocessed.")
#         # cprint("Data successfully preprocessed.", "green")
#     except Exception as e:
#         logging.error(f"Failed to load or preprocess CSV data: {e}")
#         return

#     # Initialize portfolio:
#     portfolio = Portfolio('John', initial_balance=1000)
#     env = MarketEnvironment(data=processed_data, portfolio=portfolio)
#     env.render()
#     # Initialize the RL agent (replace `DQNAgent` with your specific RL agent class)
#     # agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, learning_rate=learning_rate)

#     # Training loop:
    
#     # Training loop (placeholder code), where the agent when buying as much as needed:    
#     # Assuming agent's action now returns a tuple (action_type, shares_to_buy):
#     '''
#     # Assuming agent's action now returns a tuple (action_type, shares)
#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         cumulative_reward = 0

#         while not done:
#             # Let the agent choose the action and number of shares to buy/sell
#             action, shares = agent.select_action(state)

#             # Pass both action and shares to the step method
#             next_state, reward, done, _ = env.step(action, shares)

#             # Store the experience and train the agent
#             agent.store_experience(state, action, reward, next_state, done)
#             agent.learn(batch_size)

#             cumulative_reward += reward
#             state = next_state

#         print(f"Episode {episode} - Total Reward: {cumulative_reward}")
#     '''
    
#     # Training loop (placeholder code), where the agent when buying buys as much as possible:    
#     # for episode in range(episodes):
#     #     logging.debug(f"Starting episode {episode + 1}/{episodes}")
#     #     state = env.reset()  # Reset the environment and get the initial state
#     #     done = False
#     #     total_reward = 0

#     #     while not done:
#     #         # Agent selects an action based on the current state
#     #         # action = agent.select_action(state)  # Replace with your agent's action selection method
#     #         action = env.action_space.sample()  # Placeholder for random action selection

#     #         # Take the action and observe the result
#     #         next_state, reward, done, _ = env.step(action)

#     #         # Log the action taken and reward received
#     #         logging.debug(f"Action taken: {action}, Reward received: {reward}")

#     #         # Accumulate the total reward for the current episode
#     #         total_reward += reward

#     #         # Train the agent using the transition (state, action, reward, next_state)
#     #         # agent.learn(state, action, reward, next_state, done)  # Replace with your agent's learning method

#     #         # Update the current state
#     #         state = next_state

#     #     # Log the total reward at the end of the episode
#     #     logging.info(f"Episode {episode + 1} completed with total reward: {total_reward}")

#     #     # Optionally save the agent's model at checkpoints
#     #     # if (episode + 1) % save_interval == 0:
#     #     #     agent.save_model(f"saved_models/agent_checkpoint_{episode + 1}.h5")

#     logging.info("Training completed successfully.")
    
#     '''
#     Mothodology/Steps: 
#     Set up:
#         - Pre-process the stocks in data/
#         - Then load them into the Stocks class,, add them to a dictionary for easy look up
#         - Then load the data from the dictionary into the Market env, this would simulate the markt data, whose first index is at time step t = 0

#         - (Optional): Could calculate more metrics, which can be added to the Stock Class, for better informational data. Would need to update the market Env.

#         - Since the portfolio represents the state at a time t = 0, we need to intialize the protfolio with no stocks, and an initial balance in (Dollars) > 0.
     

    
#     Most likely then pipe them to the RL models ???
    
#     '''
    

# if __name__ == '__main__':
#     train(episodes= RL_SETTINGS["episodes"], learning_rate= RL_SETTINGS["learning_rate"]) 

from agents.drqn import DRQN, train_drqn
from agents.rl_agent import RLAgent  # Assume RL agent is implemented.
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from env.market_environment import MarketEnvironment
from env.portfolio_class import Portfolio

def hoeffding_bound(rewards, n, R=1, delta=0.05):
    mean_reward = np.mean(rewards)
    epsilon = R * math.sqrt(math.log(1/delta) / (2 * n))
    lower_bound = mean_reward - epsilon
    upper_bound = mean_reward + epsilon
    return lower_bound, upper_bound

def train(episodes, learning_rate, csv_path, required_columns):
    # Data Preprocessing
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.load_csv(csv_path, required_columns)

    # Portfolio and Environment Setup
    portfolio = Portfolio('John', initial_balance=1000)
    env = MarketEnvironment(data=processed_data, portfolio=portfolio)

    # Load the trained DRQN model
    drqn = DRQN(input_dim=7, action_space=3)  # 7 features (state space), 3 actions
    try:
        drqn.load_state_dict(torch.load('saved_models/drqn_model.pth'))
        print("DRQN model loaded successfully.")
    except FileNotFoundError:
        print("DRQN model not found. Training a new model...")
        train_drqn(data, drqn, optim.Adam(drqn.parameters(), lr=0.001), nn.MSELoss())



    # Initialize Reinforcement Learning Policy
    agent = RLAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, learning_rate=learning_rate)

    for episode in range(episodes):
        state = env.reset()
        hidden_state = drqn.init_hidden(batch_size=1)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)

            # Predict reward using DRQN
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            predicted_reward, hidden_state = drqn(state_tensor, hidden_state)
            predicted_reward = predicted_reward.item()

            # Validate reward with Hoeffding Bound
            lower_bound, upper_bound = hoeffding_bound([predicted_reward], n=100)
            if lower_bound <= predicted_reward <= upper_bound:
                next_state, reward, done, _ = env.step(action)

                # Store experience and train RL agent
                agent.store_experience(state, action, reward, next_state, done)
                agent.learn()

                state = next_state
                total_reward += reward

        print(f"Episode {episode} - Total Reward: {total_reward}")

    torch.save(agent.state_dict(), 'saved_models/rl_policy.pth')

if __name__ == '__main__':
    train(episodes=50, learning_rate=0.001, csv_path='path/to/your/data', required_columns=['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'])
