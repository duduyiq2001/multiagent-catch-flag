import gym
import time
from gym.envs.registration import register
import argparse
import sys
import gym
import numpy as np
import logging
import tqdm
import random
import json
logging.basicConfig(filename='training.log', level=logging.INFO)
sys.path.append(r"../")
# Define the multi-agent environment.

# Training parameters
n_training_episodes = 50000 # Total training episodes
learning_rate = 0.7# Learning rate
# Evaluation parameters
n_eval_episodes = 110 # Total number of test episodes
from collections import defaultdict
# Environment parameters
max_steps = 500 # Max steps per episode
gamma = 0.95

max_epsilon = 1.0 # Exploration probability at start
min_epsilon = 0.05 # Minimum exploration probability 
decay_rate = 0.001
def default_value():
   
    return np.array([0.0, 0.0, 0.0, 0.0])
policy1 = defaultdict(default_value)
policy2 = defaultdict(default_value)







for a_prime in range(n_actions):
    if a_prime == a:
        policy[s, a_prime] += alpha * G[t] * (1 - policy[s, a_prime])
    else:
        policy[s, a_prime] -= alpha * G[t] * policy[s, a_prime]