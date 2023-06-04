
import gym
import time
from gym.envs.registration import register
import argparse
import sys
import gym
import numpy as np
import logging
logging.basicConfig(filename='training.log', level=logging.INFO)
sys.path.append(r"../")
# Define the multi-agent environment.

# Training parameters
n_training_episodes = 25000 # Total training episodes
learning_rate = 0.7 # Learning rate
# Evaluation parameters
n_eval_episodes = 100 # Total number of test episodes
from collections import defaultdict
# Environment parameters
max_steps = 99 # Max steps per episode
gamma = 0.95

max_epsilon = 1.0 # Exploration probability at start
min_epsilon = 0.05 # Minimum exploration probability 
decay_rate = 0.005 
def default_value():
    return np.array([0.0, 0.0, 0.0, 0.0])
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
        # else --> exploration
    else:
        '''
        print(Qtable)
        length = Qtable[state].size
        
        action = random.randint(0,length-1)
        '''
        action = np.random.choice(Qtable[state]) + 1
    return action
def getadvobs(obs):
    advobs = []
    advobs.append(obs[3])
    advobs[0].append(obs[6])
    return advobs
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    # store the training progress of this algorithm for each episode
    episode_steps = []
    episode_resolveds = []
    episode_rewards = []
    counter = 0
    for episode in tqdm(range(n_training_episodes)):
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
        for step in range(max_steps):
            `# Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r) 
            #print(result)
            actions = [action,0,0]
            new_state, reward, done,info = env.step(actions)
            state = getadvobs(new_state)
            ep_reward += reward[0]
            Qtable[state][action-1] += Qtable[state][action-1] + learning_rate * (
                reward[0] + gamma * np.max(Qtable[state]) - Qtable[state][action-1]
            )
            # If done, finish the episode`
            if done or truncated:
            # -> store the collected rewards & number of steps in this episode 
                episode_steps.append(step)
                episode_resolveds.append(1)
                print(f'episode {counter} has reward {ep_reward}')
                print(f'episode {counter} has length {step}')
                episode_rewards.append(ep_reward)
                step = 0
                break
                # Our next state is the new state
            if step != 0:
                # -> store the collected rewards & number of steps in this episode 
                episode_steps.append(step)
                episode_resolveds.append(0)
                episode_rewards.append(ep_reward)
                step = 0
            
    return Qtable, episode_rewards, episode_steps, episode_resolved


register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame5by5',
        )
env = gym.make('multigrid-collect-v0')
policy = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, qtable)
import json


with open('policy1.json', 'w') as fp:
    json.dump(policy, fp)

# Load a dictionary back from the json file.
with open('policy1.json', 'r') as fp:
    example_dict = json.load(fp)

print(example_dict)