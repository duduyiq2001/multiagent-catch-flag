
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
n_training_episodes = 25000 # Total training episodes
learning_rate = 0.7# Learning rate
# Evaluation parameters
n_eval_episodes = 110 # Total number of test episodes
from collections import defaultdict
# Environment parameters
max_steps = 300 # Max steps per episode
gamma = 0.95

max_epsilon = 1.0 # Exploration probability at start
min_epsilon = 0.05 # Minimum exploration probability 
decay_rate = 0.001
def default_value():
    return np.array([300.0, 300.0, 300.0, 300.0])
qtable = defaultdict(default_value)


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
        action = greedy_policy(Qtable, state) + 1
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
def getadvobs(obs):
    
    advobs = np.append(obs[3], obs[6])
    #print(advobs)
    return tuple(advobs)
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
        state = getadvobs(new_state)
        # repeat
        steps = 0
        for i in range(max_steps):
            # Choose the action At using epsilon greedy policy
            steps += 1
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            #print(action)
            #print(type(action))
            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r) 
            #print(result)
            actions = [action,0,0]
            new_state, reward, done,info = env.step(actions)
            new_state = getadvobs(new_state)
            try:
                ep_reward += reward[0]
                
                Qtable[state][action-1] += Qtable[state][action-1] + learning_rate * (
                    reward[0] + gamma * np.max(Qtable[new_state]) - Qtable[state][action-1]
                )
                if Qtable[state][action-1] < 0.0:
                    Qtable[state][action-1] = 0.0
               

                # Normalize the array by dividing each element by the sum
                Qtable[state] = (Qtable[state] / np.sum(Qtable[state]))*1200.0

                state = new_state
            except RuntimeWarning:
                print("Overflow or invalid value encountered!")
                print("State: ", state)
                print("Action: ", action)
                print("Reward: ", reward[0])
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
                episode_rewards.append(ep_reward)
                step = 0
                break
                # Our next state is the new state
        
        if steps != 0:
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

with open('policy5.json', 'w') as fp:
    json.dump(str_policy, fp)

# Load the JSON and convert keys back to tuples.
with open('policy5.json', 'r') as fp:
    str_policy = json.load(fp)

apolicy = {eval(k): np.array(v) if isinstance(v, list) else v for k, v in str_policy.items()}

print(apolicy)  # {(1, 2): 'value'}