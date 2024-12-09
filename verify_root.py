import os
from termcolor import cprint

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

RootDir = os.getenv("PROJECT_ROOT")

if RootDir is None or "RL_StockTrader" not in RootDir: 
    cprint(f"Detected PROJECT_ROOT: {PROJECT_ROOT}\n", "yellow")
    if "RL_StockTrader" not in PROJECT_ROOT:
        cprint("Make sure you are running from within the RL_StockTrader project directory!!!\n", "red")
        exit(1)


    cprint ("It is very important that the Project Root is set correctly", "red")
    print("Is ", end="")
    cprint(PROJECT_ROOT, "cyan", end="")
    user_input = input(" the correct project root directory? (y/N): ").strip().lower()

    if user_input not in ["y", "yes"]:
        cprint("\nExiting script. Please re-download the submission and do not move any files around.", "red")
        exit(1)

    print ()
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT
    cprint(f"PROJECT_ROOT set to: {PROJECT_ROOT}\n", "green")