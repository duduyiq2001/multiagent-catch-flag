import gym
import time
from gym.envs.registration import register
import argparse
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import imageio
import json
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
time.sleep(5)
sys.path.append(r"../")
with open('../savedmodels/policy4.json', 'r') as fp:
    str_policy = json.load(fp)
data = {eval(k): np.array(v) for k, v in str_policy.items()}
def default_value():
    return np.array([300.0, 300.0, 300.0, 300.0])
# Convert the normal dictionary to a defaultdict
apolicy = defaultdict(default_value, data)

def greedy_policy(Qtable, state):
    Qarray = Qtable[state]
    action = np.argmax(Qarray)
    return action
def getadvobs(obs):
    advobs = np.append(obs[3], obs[6])
    return tuple(advobs)
"""
Generate a replay video of the agent
:param env
:param Qtable: Qtable of our agent
:param out_directory
:param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we
"""
register(
        id='multigrid-collect-v1',
        entry_point='gym_multigrid.envs:CollectGame15by5',
    )
env = gym.make('multigrid-collect-v1')
state = env.reset()
state = getadvobs(state)
done = False
img = env.render(mode='human')
counter = 0
while not done:
    action = greedy_policy(apolicy, state) + 1
    actions = []
    actions.append(action)
    actions.append(0)
    actions.append(0)
    counter +=1
    state, reward, done, info = env.step(actions) # We directly put next_stat
    state = getadvobs(state)
    env.render(mode= 'human')
    if counter >= 200:
        break

