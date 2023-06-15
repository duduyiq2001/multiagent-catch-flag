import imageio
import gym
import time
from gym.envs.registration import register
import sys
import numpy as np
sys.path.append(r"../")
def record_video(out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we
    """
    images = []
    done = False
    register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
    env = gym.make('multigrid-collect-v0')
    state = env.reset()
    img = env.render(mode='rgb_array')
    images.append(img)
    counter = 0
    while not done:
        # Take the action (index) that have the maximum expected future reward g
        #action, _ = policy.act(state)
        actions = [env.action_space.sample() for i in range(3)]
        print(counter)
        counter +=1
        state, reward, done, info = env.step(actions) # We directly put next_stat
        img = env.render(mode='rgb_array')
        images.append(img)
        imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)])
        if counter >= 200:
            break
record_video('./replay2.mp4')