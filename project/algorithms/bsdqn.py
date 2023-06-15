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
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space =  gym.spaces.Box(low=-2, high=12, shape=(10, 10, 6), dtype=np.float32)
    def step(self, action):
        actions = [action+1]
        actions.append(0)
        actions.append(0)
        obs, rewards, done, info = self.superenv.step(actions)
        return np.array(obs[0]).astype(np.float32),rewards[0], done, info

    def reset(self):
        return np.array(self.superenv.reset()[0]).astype(np.float32)
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
    done = False
    img = env.render(mode='rgb_array')
    images.append(img)
    counter = 0
    while not done:
        # Take the action (index) that have the maximum expected future reward g
        #action, _ = policy.act(state)
        actions = []
        print(model.predict(np.array(state[0]).astype(np.float32)))
        #print(multienv)
        actions.append(model.predict(np.array(state[0]).astype(np.float32))[0]+1)
        actions.append(0)
        actions.append(0)
        #print(counter)
        counter +=1
        state, reward, done, info = env.step(actions) # We directly put next_stat
        #state = np.array(state).astype(np.float32).flatten()
        img = env.render(mode='rgb_array')
        env.render(mode= 'human')
        images.append(img)
        imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)])
        if counter >= 200:
            break


register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGamefullobs',
        )
multienv = gym.make('multigrid-collect-v0')
env = SingleAgentWrapper(multienv)
check_env(env)
#model = DQN(policy="MlpPolicy", env=env, verbose=1,tensorboard_log= "./")
model = DQN.load("bsdpnfulladvk0",env=env)
for i in range(10):
    model.learn(total_timesteps=300000,progress_bar=True,log_interval=5)

    record_video(f'./replay{i}.mp4',env=multienv,model=model)


    model.save(f'bsdpnfulladvk{i}')