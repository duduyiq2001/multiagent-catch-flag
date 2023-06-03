import gym
import time
from gym.envs.registration import register
import argparse
import sys
import gym
import numpy as np
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


sys.path.append(r"../")
class SingleAgentWrapper(gym.Env):
    def __init__(self,superenv):
        self.superenv = superenv
        self.action_space = self.superenv.action_space
        self.observation_space = gym.spaces.Box(low=0, high=8, shape=(12,), dtype=np.float32)
    def step(self, action):
        actions = [action]
        actions.append(0)
        actions.append(0)
        obs, rewards, done, info = self.superenv.step(actions)
        return np.array(obs).astype(np.float32).flatten(),rewards[0], done, info

    def reset(self):
        return np.array(self.superenv.reset()).astype(np.float32).flatten()

# Load the trained model
register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGamefullobs',
        )
multienv = gym.make('multigrid-collect-v0')
env = SingleAgentWrapper(multienv)
check_env(env)
model = DQN(policy="MlpPolicy", env=env, verbose=1,tensorboard_log= "./")
model = DQN.load("bsdpnfulladv")

# Evaluate the model
mean_reward, meanlen = evaluate_policy(model, env, deterministic= True,n_eval_episodes=10,return_episode_rewards= True)

print(f"Mean reward: {mean_reward} +/- {meanlen}")