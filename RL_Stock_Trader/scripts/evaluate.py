# Script for evaluating the RL agent
from config.logging_config import setup_logging, cprint

from config.rl_config import RL_SETTINGS

def evaluate(episodes: (None | int) = RL_SETTINGS["episodes"]) -> None:
    cprint(f"Evaluation started...", "cyan")
    # Placeholder for evaluation code

if __name__ == '__main__':
    evaluate(episodes= RL_SETTINGS["episodes"])
