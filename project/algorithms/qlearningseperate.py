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
   
    return np.array([300.0, 300.0, 300.0, 300.0])
qtable1 = defaultdict(default_value)
qtable2 = defaultdict(default_value)


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    Qarray = Qtable[state]
    
    #print(mask_indices)
    # Find the index of the maximum value in the filtered Qarray
    action = np.argmax(Qarray)
    #print(max_index)
    # Get the corresponding index in the original Qarray
    return action
def epsilon_greedy_policy(Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = random.random()
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        #action = np.argmax(Qtable[state][:])
        action = greedy_policy(Qtable, state) +1
       
        #print(f'greedy{action}')
        # else --> exploration
    else:
        '''
        print(Qtable)
        length = Qtable[state].size
        
        action = random.randint(0,length-1)
        '''
        action = random.randint(1,4)
    
    return action
def getplayerobs(obs):
    
    p1obs = np.append(obs[4], obs[7])
    p2obs = np.append(obs[5], obs[8])
    players = np.append(p1obs,p2obs)
    players = np.append(players,obs[9])
    players = np.append(players,obs[10])
    #print(players)
    
    return tuple(players)
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable1,Qtable2):
    # store the training progress of this algorithm for each episode
    episode_steps = []
    episode_resolveds = []
    episode_rewards1 = []
    episode_rewards2 = []
    counter = 0
    for episode in tqdm.tqdm(range(n_training_episodes)):
        counter += 1
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset the environment
        new_state = env.reset()
        #print(state)
        done = False
        ep_reward1 = 0
        ep_reward2 = 0
        state = getplayerobs(new_state)
        # repeat
        steps = 0
        for i in range(max_steps):
            # Choose the action At using epsilon greedy policy
            steps += 1
            action1 = epsilon_greedy_policy(Qtable1,state,epsilon)
            action2 = epsilon_greedy_policy(Qtable2,state,epsilon)
            #print(actions)
            #print(action)
            #print(type(action))
            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r) 
            #print(result)
            actions = [0,action1,action2]
            #print(actions)
            new_state, reward, done,info = env.step(actions)
            new_state = getplayerobs(new_state)
            #print(new_state)
            try:
                #update player1
                ep_reward1 += reward[1] 
                Qtable1[state][action1-1] += Qtable1[state][action1-1] + learning_rate * (
                    reward[1] + gamma * np.max(Qtable1[new_state]) - Qtable1[state][action1-1]
                )
                if Qtable1[state][action1-1] < 0:
                    Qtable1[state][action1-1] = 0.0
                Qtable1[state] = (Qtable1[state] / np.sum(Qtable1[state]))*1200.0
                #update player2
                ep_reward2 += reward[2] 
                Qtable2[state][action2-1] += Qtable2[state][action2-1] + learning_rate * (
                    reward[2] + gamma * np.max(Qtable2[new_state]) - Qtable2[state][action2-1]
                )
                if Qtable2[state][action2-1] < 0:
                    Qtable2[state][action2-1] = 0.0
                Qtable2[state] = (Qtable2[state] / np.sum(Qtable2[state]))*1200.0

                state = new_state
            except RuntimeWarning:
                print("Overflow or invalid value encountered!")
                print("State: ", state)
                print("Action: ", action1)
                print("Reward: ", reward[1])
                print("Gamma: ", gamma)
                print("Qtable[state]: ", Qtable1[state])
                raise Exception
            # If done, finish the episode`
            if done:
            # -> store the collected rewards & number of steps in this episode 
                episode_steps.append(steps)
                episode_resolveds.append(1)
                print(f'episode {counter} has player1 reward {ep_reward1}')
                print(f'episode {counter} has player2 reward {ep_reward2}')
                print(f'episode {counter} has length {steps}')
                print(f'episode with ep{epsilon}')
                if episode %200 == 41:
                    with tf.summary.create_file_writer("two_playerteamsep").as_default():
                        tf.summary.scalar('reward1',sum(episode_rewards1[-40:])/40.0, step=episode)
                        tf.summary.scalar('reward2',sum(episode_rewards2[-40:])/40.0, step=episode)
                        tf.summary.scalar('length',sum(episode_steps[-40:])/40.0, step=episode)
                episode_rewards1.append(ep_reward1)
                episode_rewards2.append(ep_reward2)
                #print(episode_rewards1)
                #print(episode_rewards2)
                step = 0
                break
                # Our next state is the new state
        
        if steps >= max_steps:
            # -> store the collected rewards & number of steps in this episode 
            episode_steps.append(steps)
            episode_resolveds.append(0)
            episode_rewards1.append(ep_reward1)
            episode_rewards2.append(ep_reward2)
            steps = 0
            print("ended")
        
            
    return Qtable1,Qtable2, episode_rewards1,episode_rewards2, episode_steps, episode_resolveds


register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame5by5',
        )
env = gym.make('multigrid-collect-v0')
policy1,policy2,episode_rewards1,episode_rewards2,episode_steps,episode_resolveds = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, qtable1,qtable2)

str_policy1 = {str(k): v.tolist() if isinstance(v, np.ndarray) else v for k, v in policy1.items()}
str_policy2 = {str(k): v.tolist() if isinstance(v, np.ndarray) else v for k, v in policy2.items()}

with open('policyp1A.json', 'w') as fp:
    json.dump(str_policy1, fp)

# Load the JSON and convert keys back to tuples.
with open('policyp2A.json', 'w') as fp1:
    json.dump(str_policy2, fp1)
    
    

with open('policyp1A.json', 'r') as fp2:
    str_policy1 = json.load(fp2)
with open('policyp2A.json', 'r') as fp3:
    str_policy2 = json.load(fp3)

apolicy1 = {eval(k): np.array(v) if isinstance(v, list) else v for k, v in str_policy1.items()}

print(apolicy1)  # {(1, 2): 'value'}
