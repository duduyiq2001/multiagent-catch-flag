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
        self.observation_space = gym.spaces.Box(low=0, high=8, shape=(12,), dtype=np.float32)
    def step(self, action):
        actions = [action]
        actions.append(0)
        actions.append(0)
        obs, rewards, done, info = self.superenv.step(actions)
        return np.array(obs).astype(np.float32).flatten(),rewards[0], done, info

    def reset(self):
        return np.array(self.superenv.reset()).astype(np.float32).flatten()


register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGamefullobs',
        )
multienv = gym.make('multigrid-collect-v0')
env = SingleAgentWrapper(multienv)
check_env(env)
model = DQN(policy="MlpPolicy", env=env, verbose=1,tensorboard_log= "./")
#model = DQN.load("bsdpnfulladv",env=env,exploration_initial_eps=0.05, exploration_final_eps=0.03)
model.learn(total_timesteps=800000,progress_bar=True,log_interval=5)
import imageio
def record_video(out_directory,env,model, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we
    """
    images = []
    state = env.reset()
    state = np.array(state).astype(np.float32).flatten()
    done = False
    img = env.render(mode='rgb_array')
    images.append(img)
    counter = 0
    while not done:
        # Take the action (index) that have the maximum expected future reward g
        #action, _ = policy.act(state)
        actions = []
        print(model.predict(state)[0])
        #print(multienv)
        actions.append(model.predict(state)[0])
        actions.append(0)
        actions.append(0)
        #print(counter)
        counter +=1
        state, reward, done, info = env.step(actions) # We directly put next_stat
        state = np.array(state).astype(np.float32).flatten()
        img = env.render(mode='rgb_array')
        env.render(mode= 'human')
        images.append(img)
        imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)])
        if counter >= 200:
            break

record_video('./replay2.mp4',env=multienv,model=model)


model.save("bsdpnfulladv1")