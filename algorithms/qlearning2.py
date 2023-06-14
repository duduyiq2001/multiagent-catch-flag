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
import tensorflow as tf
from scipy.special import softmax
logging.basicConfig(filename='training.log', level=logging.INFO)
sys.path.append(r"../")
# Define the multi-agent environment.

# Training parameters
n_training_episodes = 50000# Total training episodes
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
   
    return np.array([300.0, 300.0, 300.0, 300.0,300.0,300.0,300.0,300.0,300.0,300.0,300.0, 300.0, 300.0, 300.0,300.0,300.0])
qtable = defaultdict(default_value)


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
def epsilon_greedy_policy(Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = random.random()
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        #action = np.argmax(Qtable[state][:])
        actions = greedy_policy(Qtable, state)
        player1 = actions[0] + 1
        player2 = actions[1] + 1
        #print(f'greedy{action}')
        # else --> exploration
    else:
        '''
        print(Qtable)
        length = Qtable[state].size
        
        action = random.randint(0,length-1)
        '''
        player1 = random.randint(1,4)
        player2 = random.randint(1,4)

    return [player1,player2]
def getplayerobs(obs):
    
    p1obs = np.append(obs[4], obs[7])
    p2obs = np.append(obs[5], obs[8])
    players = np.append(p1obs,p2obs)
    players = np.append(players,obs[9])
    players = np.append(players,obs[10])
    #print(players)
    
    return tuple(players)
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    # store the training progress of this algorithm for each episode
    episode_steps = []
    episode_resolveds = []
    episode_rewards = []
    counter = 0
    for episode in tqdm.tqdm(range(n_training_episodes)):
        counter += 1
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset the environment
        new_state = env.reset()
        #print(state)
        done = False
        ep_reward = 0
        state = getplayerobs(new_state)
        # repeat
        steps = 0
        for i in range(max_steps):
            # Choose the action At using epsilon greedy policy
            steps += 1
            actions = epsilon_greedy_policy(Qtable, state, epsilon)
            #print(actions)
            #print(action)
            #print(type(action))
            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r) 
            #print(result)
            actions = [0,actions[0],actions[1]]
            new_state, reward, done,info = env.step(actions)
            new_state = getplayerobs(new_state)
            #print(new_state)
            try:
                ep_reward += reward[1] + reward[2]
                actnum = actnum_look([actions[1]-1,actions[2]-1])
                Qtable[state][actnum] += Qtable[state][actnum] + learning_rate * (
                    reward[1]+reward[2] + gamma * np.max(Qtable[new_state]) - Qtable[state][actnum]
                )
                
                if Qtable[state][actnum] < 0:
                    Qtable[state][actnum] = 0.0
                Qtable[state] = (Qtable[state] / np.sum(Qtable[state]))*1200.0

                state = new_state
            except RuntimeWarning:
                print("Overflow or invalid value encountered!")
                print("State: ", state)
                print("Action: ", action)
                print("Reward: ", reward[1] + reward[2])
                print("Gamma: ", gamma)
                print("Qtable[state]: ", Qtable[state])
                raise Exception
            # If done, finish the episode`
            if done:
            # -> store the collected rewards & number of steps in this episode 
                episode_steps.append(steps)
                episode_resolveds.append(1)
                print(f'episode {counter} has reward {ep_reward}')
                print(f'episode {counter} has length {steps}')
                print(f'episode with ep{epsilon}')
                if episode %200 == 41:
                    with tf.summary.create_file_writer("two_playerteam").as_default():
                        tf.summary.scalar('reward',sum(episode_rewards[-40:])/40.0, step=episode)
                        tf.summary.scalar('length',sum(episode_steps[-40:])/40.0, step=episode)
                episode_rewards.append(ep_reward)
                step = 0
                break
                # Our next state is the new state
        
        if steps >= max_steps:
            # -> store the collected rewards & number of steps in this episode 
            episode_steps.append(steps)
            episode_resolveds.append(0)
            episode_rewards.append(ep_reward)
            steps = 0
            print("ended")
        
            
    return Qtable, episode_rewards, episode_steps, episode_resolveds


register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame5by5',
        )
env = gym.make('multigrid-collect-v0')
policy,episode_rewards,episode_steps,episode_resolveds = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, qtable)

str_policy = {str(k): v.tolist() if isinstance(v, np.ndarray) else v for k, v in policy.items()}

with open('policyp1p2B.json', 'w') as fp:
    json.dump(str_policy, fp)

# Load the JSON and convert keys back to tuples.
with open('policyp1p2B.json', 'r') as fp:
    str_policy = json.load(fp)

apolicy = {eval(k): np.array(v) if isinstance(v, list) else v for k, v in str_policy.items()}

print(apolicy)  # {(1, 2): 'value'}
