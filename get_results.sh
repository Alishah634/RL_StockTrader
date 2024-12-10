#!bin/bash


# Execute COmmnads to creade files for parsing, to obtain reuslts:
time python3 Benchmarking/PPO_LargeAction.py > ppo_large_action_evaluation.log 2>&1
echo "Completed Running PPO LARGE ACTION!" 

time python3 Benchmarking/PPO.py > ppo_small_action_evaluation.log 2>&1
echo "Completed Running PPO SMALL ACTION!" 

time python3 Benchmarking/A2C.py > a2c_evaluation.log 2>&1
echo "Completed Running A2C ACTION!"


time python3 Benchmarking/drqn_model.py > drqn_evaluation.log 2>&1
echo "Completed Running DRQN ACTION!" 

time python3 metrics/TraderResults.py 