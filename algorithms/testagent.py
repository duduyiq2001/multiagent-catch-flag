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
with open('policy4.json', 'r') as fp:
    str_policy = json.load(fp)
    print(str_policy)
data = {eval(k): np.array(v) for k, v in str_policy.items()}
def default_value():
    return np.array([300.0, 300.0, 300.0, 300.0])
# Convert the normal dictionary to a defaultdict
apolicy = defaultdict(default_value, data)

def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    Qarray = Qtable[state]
    
    #print(mask_indices)
    # Find the index of the maximum value in the filtered Qarray
    action = np.argmax(Qarray)
    #print(max_index)
    # Get the corresponding index in the original Qarray
    return action
def getadvobs(obs):
    advobs = np.append(obs[3], obs[6])
    #print(advobs)
    return tuple(advobs)

"""
Generate a replay video of the agent
:param env
:param Qtable: Qtable of our agent
:param out_directory
:param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we
"""
register(
        id='multigrid-collect-v0',
        entry_point='gym_multigrid.envs:CollectGame5by5',
    )
env = gym.make('multigrid-collect-v0')
state = env.reset()
state = getadvobs(state)
print(state)
print(type(state))
done = False
img = env.render(mode='human')
counter = 0
print(apolicy[(1,4,0)])
print([i for i in apolicy.keys()])
while not done:
    # Take the action (index) that have the maximum expected future reward g
    #action, _ = policy.act(state)
    action = greedy_policy(apolicy, state) + 1
    
    actions = []
    #print(multienv)
    actions.append(action)
    print(action)
    actions.append(0)
    actions.append(0)
    #print(counter)
    counter +=1
    state, reward, done, info = env.step(actions) # We directly put next_stat
    state = getadvobs(state)
    #state = np.array(state).astype(np.float32).flatten()
    env.render(mode= 'human')
    if counter >= 200:
        break

