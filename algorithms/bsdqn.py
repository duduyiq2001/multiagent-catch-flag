import gym
import time
from gym.envs.registration import register
import argparse
import sys
import gym
import numpy as np
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
logging.basicConfig(filename='training.log', level=logging.INFO)
sys.path.append(r"../")
# Define the multi-agent environment.
class SingleAgentWrapper(gym.Env):
    def __init__(self,superenv):
        self.superenv = superenv
        self.action_space = self.superenv.action_space
        self.observation_space = self.superenv.observation_space
    def step(self, action):
        actions = [action]
        actions.append(self.superenv.action_space.sample())
        actions.append(self.superenv.action_space.sample())
        obs, rewards, done, info = self.superenv.step(actions)
        return obs[0],rewards[0], done, info

    def reset(self):
        return self.superenv.reset()[0]


register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGamefullobs',
        )
multienv = gym.make('multigrid-collect-v0')
env = SingleAgentWrapper(multienv)
check_env(env)
model = DQN(policy="MlpPolicy", env=env, verbose=1,learning_rate=0.5,train_freq=(3,"episode"),batch_size=100,gamma=0.7,exploration_initial_eps=0.2,exploration_final_eps=0.01,stats_window_size=20,buffer_size=10000,gradient_steps=30)
model.learn(total_timesteps=300000,progress_bar=True,log_interval=20)
model.save("bsdpnfulladv")