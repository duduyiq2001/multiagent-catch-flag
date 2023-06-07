#p1p2 vs adv

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
with open('policyp1p2.json', 'r') as fp:
    str_policy = json.load(fp)
    #print(str_policy)
with open('policy4.json', 'r') as fp:
    str_policy1 = json.load(fp)
    #print(str_policy)
data = {eval(k): np.array(v) for k, v in str_policy.items()}
data1 = {eval(k): np.array(v) for k, v in str_policy1.items()}
def default_value():
    return np.array([300.0, 300.0, 300.0, 300.0,300.0,300.0,300.0,300.0,300.0,300.0,300.0, 300.0, 300.0, 300.0,300.0,300.0])
def default_value1():
    return np.array([300.0, 300.0, 300.0, 300.0])
def getadvobs(obs):
    advobs = np.append(obs[3], obs[6])
    #print(advobs)
    return tuple(advobs)

# Convert the normal dictionary to a defaultdict
playerpolicy = defaultdict(default_value, data)
advplayer = defaultdict(default_value1,data1)

def action_look(actnum):
    player1 = actnum//4
    player2 = actnum%4
    return [player1,player2]
def actnum_look(actions):
    player1 = actions[0]
    player2 = actions[1]
    return int(player1*4 + player2)
def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    Qarray = Qtable[state]
    
    #print(mask_indices)
    # Find the index of the maximum value in the filtered Qarray
    action = np.argmax(Qarray)
    #print(max_index)
    # Get the corresponding index in the original Qarray
    return action_look(action)
def greedy_policy1(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    Qarray = Qtable[state]
    
    #print(mask_indices)
    # Find the index of the maximum value in the filtered Qarray
    action = np.argmax(Qarray)
    #print(max_index)
    # Get the corresponding index in the original Qarray
    return action
def getplayerobs(obs):
    
    p1obs = np.append(obs[4], obs[7])
    p2obs = np.append(obs[5], obs[8])
    players = np.append(p1obs,p2obs)
    #players = np.append(players,obs[9])
    #players = np.append(players,obs[10])
    #print(players)

    #print(advobs)
    return tuple(players)
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

print(state)
print(type(state))
done = False
img = env.render(mode='human')
counter = 0
#print([i for i in apolicy.keys()])
while not done:
    playerstate = getplayerobs(state)
    advstate = getadvobs(state)
    # Take the action (index) that have the maximum expected future reward g
    #action, _ = policy.act(state)
    playeractions = greedy_policy(playerpolicy, playerstate)   
    advaction = greedy_policy1(advplayer,advstate) + 1
    actions = [advaction,playeractions[0]+1,playeractions[1]+1]
    #actions = [advaction,0,0]
    print(actions)
    new_state, reward, done,info = env.step(actions)
    state = new_state
    #state = np.array(state).astype(np.float32).flatten()
    env.render(mode= 'human')
    if counter >= 200:
        break

